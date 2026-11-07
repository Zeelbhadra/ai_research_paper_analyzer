"""Replaceable LLM provider abstraction with ordered failover.

Providers: OpenAI (key from OPENAI_KEY, else OPENAI_API_KEY; never logged), a local HF instruct model, and an
extractive 'provider' that needs no LLM at all. `ProviderChain` tries them in order and records which one answered.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger(__name__)


class ProviderError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cost_usd: float | None = None
    attempts: list[str] = field(default_factory=list)       # providers that failed before this one


class LLMProvider(ABC):
    name = "base"
    model = ""

    @abstractmethod
    def generate(self, messages: list[dict], max_tokens: int = 500, temperature: float = 0.0) -> LLMResponse: ...

    def stream(self, messages: list[dict], max_tokens: int = 500, temperature: float = 0.0) -> Iterator[str]:
        yield self.generate(messages, max_tokens, temperature).text      # default: single chunk

    def available(self) -> bool:
        return True


# USD per 1M tokens (input, output); used only for the cost estimate, update if pricing changes.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}


_ENV_LOADED = False


def _load_project_env() -> None:
    """Make keys from the project's .env visible without requiring the caller to load them (values are never logged)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        except Exception:
            pass
        _ENV_LOADED = True


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        _load_project_env()
        self.model = model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
        self._key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self._client = None

    def available(self) -> bool:
        return bool(self._key)

    def _c(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._key, timeout=self.timeout, max_retries=1)
        return self._client

    def generate(self, messages, max_tokens=500, temperature=0.0):
        if not self._key:
            raise ProviderError("no OpenAI key configured (OPENAI_KEY / OPENAI_API_KEY)")
        t = time.perf_counter()
        try:
            r = self._c().chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                  temperature=temperature)
        except Exception as e:                       # includes rate-limit / no-credit / network errors
            raise ProviderError(f"openai: {type(e).__name__}: {str(e)[:200]}") from None
        u = r.usage
        pin, pout = PRICES.get(self.model, (None, None))
        cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6 if pin is not None else None
        return LLMResponse(r.choices[0].message.content or "", self.name, self.model, u.prompt_tokens, u.completion_tokens,
                           time.perf_counter() - t, cost)

    def stream(self, messages, max_tokens=500, temperature=0.0):
        if not self._key:
            raise ProviderError("no OpenAI key configured")
        try:
            for ev in self._c().chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                        temperature=temperature, stream=True):
                if ev.choices and ev.choices[0].delta.content:
                    yield ev.choices[0].delta.content
        except Exception as e:
            raise ProviderError(f"openai: {type(e).__name__}: {str(e)[:200]}") from None


class LocalProvider(LLMProvider):
    """Local HF instruct model (CPU is slow: ~20 prompt tokens/s for a 1.5B model on this machine)."""
    name = "local"

    def __init__(self, checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model = checkpoint
        self._llm = None

    def generate(self, messages, max_tokens=500, temperature=0.0):
        from backend.benchmarking.llm_local import LocalLLM
        self._llm = self._llm or LocalLLM(self.model)
        t = time.perf_counter()
        try:
            text = self._llm.generate(messages, max_tokens)
            n_in = len(self._llm._prompt_ids(messages)["input_ids"][0])
        except Exception as e:
            raise ProviderError(f"local: {type(e).__name__}: {e}") from None
        return LLMResponse(text, self.name, self.model, n_in, len(self._llm._tok.encode(text)), time.perf_counter() - t, 0.0)


class ProviderChain(LLMProvider):
    """Try providers in order; the first that answers wins. Raises ProviderError if all fail."""
    name = "chain"

    def __init__(self, providers: list[LLMProvider]):
        self.providers = [p for p in providers if p.available()]
        self.model = ",".join(p.name for p in self.providers)

    def available(self) -> bool:
        return bool(self.providers)

    def generate(self, messages, max_tokens=500, temperature=0.0):
        failed = []
        for p in self.providers:
            try:
                r = p.generate(messages, max_tokens, temperature)
                r.attempts = failed
                return r
            except ProviderError as e:
                log.warning("provider %s failed: %s", p.name, e)
                failed.append(f"{p.name}: {e}")
        raise ProviderError("all providers failed: " + " | ".join(failed))

    def stream(self, messages, max_tokens=500, temperature=0.0):
        failed = []
        for p in self.providers:
            started = False
            try:
                for tok in p.stream(messages, max_tokens, temperature):
                    started = True
                    yield tok
                return
            except ProviderError as e:
                if started:              # cannot restart a half-delivered answer on another provider
                    raise
                failed.append(f"{p.name}: {e}")
        raise ProviderError("all providers failed: " + " | ".join(failed))


def default_chain(use_local: bool = False) -> ProviderChain:
    """OpenAI first when configured; optionally the local model as backup. Extractive answering is handled by the
    generator itself when this chain is empty or fails."""
    ps: list[LLMProvider] = [OpenAIProvider()]
    if use_local or os.getenv("RAG_USE_LOCAL_LLM") == "1":
        ps.append(LocalProvider())
    return ProviderChain(ps)
