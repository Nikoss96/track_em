"""LLM-адаптеры со structured output: Ollama (локально) и OpenAI API (внешний сервис).

Общее для бэкендов:
- temperature 0 (если модель поддерживает), лимит выходных токенов, таймаут;
- ответ — JSON по схеме Pydantic, валидируется; одна повторная попытка при невалидном JSON;
- учёт токенов и стоимости каждого вызова (LLMUsage, общий на сессию), лимит бюджета сессии.

OpenAI — ВНЕШНИЙ сервис: в него уходит текст (эталонные фрагменты / транскрипты / вопрос; не аудио и не видео).
Включается только при `llm.allow_external: true`. Ключ — переменная окружения OPENAI_API_KEY или configs/secrets.yaml.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import threading
import time
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from app.analyzers.base import ModelUnavailable

T = TypeVar("T", bound=BaseModel)
_THINK = re.compile(r"<think>.*?</think>", re.S)
# Подмена HTTP-транспорта (только тесты: поддельные Ollama/OpenAI без сети).
TRANSPORT: httpx.AsyncBaseTransport | None = None


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


class LLMUsage:
    """Счётчик токенов/стоимости LLM на сессию (общий для TextAnalyzer и TransactionalAnalyzer)."""

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd if budget_usd and budget_usd > 0 else None
        self._lock = threading.Lock()
        self.calls = 0
        self.failed = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.priced = True
        self.by_module: dict[str, dict[str, float]] = {}

    def check_budget(self) -> None:
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(f"бюджет LLM на сессию исчерпан: ${self.cost_usd:.4f} ≥ ${self.budget_usd:.2f}")

    def add(self, module: str, usage: dict[str, int], cost: float | None) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.cached_input_tokens += usage.get("cached_input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.reasoning_tokens += usage.get("reasoning_tokens", 0)
            if cost is None:
                self.priced = False
            else:
                self.cost_usd += cost
            m = self.by_module.setdefault(module, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["input_tokens"] += usage.get("input_tokens", 0)
            m["output_tokens"] += usage.get("output_tokens", 0)
            m["cost_usd"] = round(m["cost_usd"] + (cost or 0.0), 6)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls, "failed": self.failed, "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens, "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens, "cost_usd": round(self.cost_usd, 6),
            "cost_complete": self.priced, "budget_usd": self.budget_usd, "by_module": self.by_module,
        }


def price_cost(pricing: dict[str, Any] | None, usage: dict[str, int]) -> float | None:
    """USD по тарифу за 1M токенов: {input, cached_input, output}. None — тариф не задан."""
    if not pricing:
        return None
    cached = usage.get("cached_input_tokens", 0)
    fresh = usage.get("input_tokens", 0) - cached
    return (fresh * float(pricing["input"]) + cached * float(pricing.get("cached_input", pricing["input"]))
            + usage.get("output_tokens", 0) * float(pricing["output"])) / 1e6


class BaseLLMClient:
    backend = "base"
    external = False

    def __init__(self, cfg: dict[str, Any], usage: LLMUsage | None) -> None:
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.timeout = float(cfg.get("timeout_s", 90))
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0))
        self.usage = usage or LLMUsage()
        self.version = "unknown"
        self.digest: str | None = None
        self.pricing = (cfg.get("pricing") or {}).get(self.model)

    async def check(self) -> None:  # pragma: no cover - переопределяется
        raise NotImplementedError

    async def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> tuple[str, dict[str, int]]:
        raise NotImplementedError

    async def chat_json(self, system: str, user: str, schema: type[T], retries: int = 1, module: str = "llm") -> tuple[T, dict[str, Any]]:
        self.usage.check_budget()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        meta: dict[str, Any] = {"backend": self.backend, "model": self.model, "attempts": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        last_err = ""
        for attempt in range(retries + 1):
            meta["attempts"] = attempt + 1
            t = time.perf_counter()
            try:
                content, usage = await self._call(messages, schema)
            except LLMError:
                self.usage.failed += 1
                raise
            meta["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
            cost = price_cost(self.pricing, usage) if self.external else 0.0
            self.usage.add(module, usage, cost)
            meta["input_tokens"] += usage.get("input_tokens", 0)
            meta["output_tokens"] += usage.get("output_tokens", 0)
            meta["cost_usd"] = round(meta["cost_usd"] + (cost or 0.0), 6) if cost is not None else None
            content = _THINK.sub("", content).strip()
            try:
                return schema.model_validate(json.loads(content)), meta
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = str(e)[:300]
                messages = messages + [
                    {"role": "assistant", "content": content[:2000]},
                    {"role": "user", "content": f"Ответ не прошёл проверку схемы: {last_err}. Верни только валидный JSON по схеме."},
                ]
        self.usage.failed += 1
        raise LLMError(f"невалидный JSON после {retries + 1} попыток: {last_err}")


class OllamaClient(BaseLLMClient):
    backend = "ollama"

    def __init__(self, cfg: dict[str, Any], usage: LLMUsage | None = None) -> None:
        super().__init__(cfg, usage)
        self.url = cfg.get("url", "http://127.0.0.1:11434").rstrip("/")
        host = urlparse(self.url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ModelUnavailable(f"Ollama URL {self.url} не локальный — запрещено политикой приватности")
        self.model = cfg.get("model", "qwen3:8b")
        self.num_ctx = int(cfg.get("num_ctx", 4096))
        self.think = bool(cfg.get("think", False))

    async def check(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5, transport=TRANSPORT) as c:
                v = await c.get(f"{self.url}/api/version")
                self.version = v.json().get("version", "unknown")
                tags = (await c.get(f"{self.url}/api/tags")).json().get("models", [])
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"Ollama не отвечает на {self.url}: {type(e).__name__}. Установите Ollama и выполните `ollama pull {self.model}`") from e
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        want = {self.model, f"{self.model}:latest"}
        hit = [m for m in tags if m.get("name") in want or m.get("model") in want]
        if not hit:
            raise ModelUnavailable(f"модель {self.model} не загружена в Ollama (есть: {sorted(n for n in names if n)}). Выполните `ollama pull {self.model}`")
        self.digest = hit[0].get("digest")

    async def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> tuple[str, dict[str, int]]:
        body = {
            "model": self.model, "messages": messages, "stream": False, "format": schema.model_json_schema(), "think": self.think,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx, "seed": 0},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=TRANSPORT) as c:
                r = await c.post(f"{self.url}/api/chat", json=body)
                r.raise_for_status()
                data = r.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"таймаут LLM ({self.timeout:.0f} c)") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"ошибка LLM: {type(e).__name__}: {e}") from e
        usage = {"input_tokens": int(data.get("prompt_eval_count") or 0), "output_tokens": int(data.get("eval_count") or 0)}
        return data.get("message", {}).get("content", ""), usage


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic JSON Schema → подмножество для OpenAI Structured Outputs (strict):
    все поля required, additionalProperties=false, без default/title и ограничений длины/диапазона
    (ограничения проверяются локально Pydantic после ответа)."""
    drop = {"default", "title", "maxLength", "minLength", "maxItems", "minItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in drop}
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"].keys())
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(copy.deepcopy(schema))


def read_openai_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    from app.config import read_secrets

    return (read_secrets().get("openai_api_key") or "").strip() or None


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions + Structured Outputs (response_format=json_schema, strict)."""

    backend = "openai"
    external = True

    def __init__(self, cfg: dict[str, Any], usage: LLMUsage | None = None) -> None:
        if not cfg.get("allow_external", False):
            raise ModelUnavailable("OpenAI — внешний сервис: включите llm.allow_external (Настройки) после согласования с участниками исследования")
        super().__init__(cfg, usage)
        self.url = (cfg.get("openai_url") or "https://api.openai.com/v1").rstrip("/")
        self.model = cfg.get("openai_model", "gpt-4.1-mini")
        self.pricing = (cfg.get("pricing") or {}).get(self.model)
        self.max_tokens = int(cfg.get("openai_max_tokens", 1200))
        self.reasoning_effort = cfg.get("openai_reasoning_effort")  # для reasoning-моделей: minimal|low|...
        self.max_retries = int(cfg.get("openai_max_retries", 4))
        self.key = read_openai_key()
        if not self.key:
            raise ModelUnavailable("нет ключа OpenAI: задайте OPENAI_API_KEY или сохраните ключ в «Настройках»")
        self._no_temperature = False  # часть моделей (reasoning) не принимает temperature

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    async def check(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=15, transport=TRANSPORT) as c:
                r = await c.get(f"{self.url}/models/{self.model}", headers=self._headers())
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"OpenAI API недоступен: {type(e).__name__}") from e
        if r.status_code == 401:
            raise ModelUnavailable("OpenAI: неверный API-ключ (401)")
        if r.status_code == 404:
            raise ModelUnavailable(f"OpenAI: модель {self.model} недоступна для этого ключа (404)")
        if r.status_code >= 400:
            raise ModelUnavailable(f"OpenAI: HTTP {r.status_code}: {r.text[:200]}")
        self.version = r.json().get("id", self.model)
        if not self.pricing:
            self.usage.priced = False  # стоимость не посчитать без тарифа — видно в сводке

    async def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> tuple[str, dict[str, int]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "strict": True, "schema": strict_schema(schema.model_json_schema())}},
            "seed": 0,
            "store": False,
        }
        if not self._no_temperature:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, transport=TRANSPORT) as c:
                    r = await c.post(f"{self.url}/chat/completions", headers=self._headers(), json=body)
            except httpx.TimeoutException as e:
                raise LLMError(f"таймаут OpenAI ({self.timeout:.0f} c)") from e
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"ошибка сети OpenAI: {type(e).__name__}: {e}") from e
            if r.status_code == 400 and "temperature" in r.text and not self._no_temperature:
                self._no_temperature = True
                body.pop("temperature", None)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                if r.status_code == 429 and "insufficient_quota" in r.text:
                    raise LLMError("OpenAI: исчерпана квота/баланс аккаунта (insufficient_quota)")
                ra = r.headers.get("retry-after")
                await asyncio.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else delay)
                delay = min(delay * 2, 20)
                continue
            if r.status_code >= 400:
                raise LLMError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            ch = data["choices"][0]
            msg = ch.get("message", {})
            if msg.get("refusal"):
                raise LLMError(f"OpenAI отказ: {msg['refusal'][:200]}")
            if ch.get("finish_reason") == "length":
                raise LLMError("OpenAI: ответ обрезан лимитом токенов (увеличьте openai_max_tokens)")
            u = data.get("usage") or {}
            usage = {
                "input_tokens": int(u.get("prompt_tokens") or 0),
                "cached_input_tokens": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "output_tokens": int(u.get("completion_tokens") or 0),
                "reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            }
            return msg.get("content") or "", usage
        raise LLMError("OpenAI: превышено число повторов (429/5xx)")


def llm_identity(llm: BaseLLMClient) -> tuple[str, str, dict[str, Any]]:
    """(model_id, model_version, запись реестра моделей) для результатов и реестра."""
    if llm.backend == "openai":
        reg = {"model_id": llm.model, "version": llm.version, "backend": "openai api", "source": llm.url,
               "license": "OpenAI API Terms (внешний сервис)", "languages": "multilingual (ru)",
               "notes": f"тариф $/1M: {llm.pricing}" if llm.pricing else "тариф не задан в llm.pricing"}
        return f"openai/{llm.model}", f"openai api; {llm.version}", reg
    reg = {"model_id": llm.model, "version": str(llm.digest), "backend": f"ollama {llm.version}", "source": "ollama library",
           "license": "см. карточку модели (Qwen3 — Apache-2.0)", "languages": "multilingual (ru)"}
    return f"ollama/{llm.model}", f"ollama {llm.version}; digest {str(llm.digest)[:12]}", reg


def make_llm(cfg: dict[str, Any], usage: LLMUsage | None = None) -> BaseLLMClient:
    backend = cfg.get("backend", "ollama")
    if backend == "ollama":
        return OllamaClient(cfg, usage)
    if backend == "openai":
        return OpenAIClient(cfg, usage)
    raise ModelUnavailable(f"LLM backend {backend} не поддерживается")


def grounded(quote: str, text: str) -> bool:
    """Цитата действительно присутствует во входном тексте (после нормализации)."""
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[«»\"'“”„—–\-.,!?…:;()]", " ", s.lower().replace("ё", "е"))).strip()  # noqa: E731
    q = norm(quote)
    return bool(q) and q in norm(text)
