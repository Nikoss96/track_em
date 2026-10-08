"""OpenAI-бэкенд без сети: httpx.MockTransport имитирует /v1/models и /v1/chat/completions."""

import json

import httpx
import pytest
import soundfile as sf

from app.analyzers import llm_client
from app.analyzers.llm_client import LLMUsage, OpenAIClient, strict_schema
from app.analyzers.speech import backends as asr_backends
from app.analyzers.text.analyzer import TextAnalysisOut
from app.analyzers.transactional.analyzer import TransactionalOut
from tests.fakes import FakeASR, FakeOllama
from tests.harness import Harness


class FakeOpenAI:
    def __init__(self, fail_429: int = 0, reject_temperature: bool = False) -> None:
        self.fail_429 = fail_429
        self.reject_temperature = reject_temperature
        self.bodies: list[dict] = []
        self.ollama = FakeOllama()  # переиспользуем генерацию контента
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer sk-test"
        if req.url.path.startswith("/v1/models/"):
            return httpx.Response(200, json={"id": req.url.path.rsplit("/", 1)[-1], "object": "model"})
        body = json.loads(req.content)
        self.bodies.append(body)
        if self.fail_429 > 0:
            self.fail_429 -= 1
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"type": "rate_limit"}})
        if self.reject_temperature and "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0"}})
        rf = body["response_format"]["json_schema"]
        assert rf["strict"] is True and body["store"] is False
        # конвертируем запрос в формат FakeOllama, чтобы получить JSON по нужной схеме
        fake_req = httpx.Request("POST", "http://127.0.0.1:11434/api/chat", json={
            "messages": body["messages"], "stream": False, "format": {"title": rf["name"]},
            "options": {"temperature": 0}})
        content = json.loads(self.ollama.handle(fake_req).content)["message"]["content"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content.replace("<think>x</think>", "")}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}},
        })


CFG = {"backend": "openai", "allow_external": True, "openai_model": "gpt-4.1-mini", "timeout_s": 10,
       "pricing": {"gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60}}}


def test_strict_schema_shape():
    s = strict_schema(TransactionalOut.model_json_schema())
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    rep = s["$defs"]["ReplicaOut"]
    assert rep["additionalProperties"] is False and "evidence" in rep["required"]
    assert "maxLength" not in json.dumps(s) and "default" not in json.dumps(s)


async def test_consent_and_key_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("app.config.read_secrets", lambda: {})
    with pytest.raises(llm_client.ModelUnavailable, match="allow_external"):
        OpenAIClient({**CFG, "allow_external": False})
    with pytest.raises(llm_client.ModelUnavailable, match="ключа"):
        OpenAIClient(CFG)


async def test_cost_retry_temperature_and_budget(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = FakeOpenAI(fail_429=2, reject_temperature=True)
    monkeypatch.setattr(llm_client, "TRANSPORT", fake.transport)
    usage = LLMUsage(budget_usd=0.0015)
    c = OpenAIClient(CFG, usage)
    await c.check()
    res, meta = await c.chat_json("sys", "АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ: «Мама мыла раму.»", TextAnalysisOut, module="text")
    assert res.sentiment == "positive"
    # 600 свежих × 0.40 + 400 кэш × 0.10 + 200 вых × 1.60 = (240 + 40 + 320)/1e6
    assert abs(meta["cost_usd"] - 0.0006) < 1e-9
    assert len(fake.bodies) == 4 and "temperature" not in fake.bodies[-1]  # 2×429 + отказ temperature + успех
    await c.chat_json("sys", "АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ: «Мама мыла раму.»", TextAnalysisOut, module="text")
    await c.chat_json("sys", "АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ: «Мама мыла раму.»", TextAnalysisOut, module="text")
    with pytest.raises(llm_client.BudgetExceeded):
        await c.chat_json("sys", "x", TextAnalysisOut)
    s = usage.summary()
    assert s["calls"] == 3 and s["input_tokens"] == 3000 and abs(s["cost_usd"] - 0.0018) < 1e-9 and s["cost_complete"]


async def test_session_modules_with_openai(fixtures, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = FakeOpenAI()
    monkeypatch.setattr(llm_client, "TRANSPORT", fake.transport)
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    FakeASR.reference, FakeASR.total_ms, FakeASR.corrupt_every = ref, sf.info(str(fixtures / "reading_tts.wav")).duration * 1000, 0
    monkeypatch.setitem(asr_backends.BACKENDS, "fake", FakeASR)
    h = Harness(["speech", "text", "transactional"], mode="reading", reference_text=ref,
                overrides={"llm": CFG, "modules": {"speech": {"backend": "fake", "partial_interval_ms": 0}}})
    await h.run_file(fixtures / "reading_tts.wav")
    text = [r for r in h.by("text") if r.status.value == "ok"]
    assert text and text[0].model_id == "openai/gpt-4.1-mini"
    assert text[0].payload["llm"]["cost_usd"] > 0 and text[0].payload["llm"]["backend"] == "openai"
    assert any(r.status.value == "ok" for r in h.by("transactional"))
    assert h.models and h.models[0]["license"].startswith("OpenAI")
