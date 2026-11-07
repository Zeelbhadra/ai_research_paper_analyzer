"""Production retrieval service: serves the registry's production model, falls back to TF-IDF on failure,
recovers automatically, and swaps production models safely.

State machine
  PRIMARY  --(failure_threshold consecutive failures, or failed health probe)-->  FALLBACK
  FALLBACK --(recovery_checks consecutive healthy probes, spaced by exponential cooldown)--> PRIMARY
A failing request is always retried on the fallback immediately, so the caller still gets results.
If the production model IS the fallback model (TF-IDF) there is no independent fallback and failures surface
as RetrievalUnavailable.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from backend.monitoring.health_checks import probe
from backend.monitoring.logger import EventLog
from backend.retrieval.retriever import Retriever, make_filter
from backend.vectorstore.base import Hit

log = logging.getLogger(__name__)


class RetrievalUnavailable(Exception):
    pass


@dataclass
class ServiceConfig:
    failure_threshold: int = 2          # consecutive primary failures before switching mode
    recovery_checks: int = 2            # consecutive healthy probes before reactivating primary
    cooldown_base_s: float = 5.0        # delay before first recovery attempt, doubles per failed attempt
    cooldown_max_s: float = 300.0


@dataclass
class RetrievalResult:
    hits: list[Hit]
    mode: str                           # 'primary' | 'fallback'
    retriever: str
    kind: str                           # dense | sparse | hybrid
    fallback_active: bool
    latency_ms: float
    error: str | None = None            # why the primary was bypassed for THIS request


@dataclass
class ServiceStatus:
    mode: str = "primary"
    fallback_active: bool = False
    primary: str | None = None
    primary_version: str | None = None
    fallback: str | None = None
    reason: str | None = None
    since: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    fallback_requests: int = 0
    recovery_attempts: int = 0
    healthy_probes: int = 0
    next_recovery_at: float | None = None
    last_probe: dict | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RetrievalService:
    def __init__(self, primary_factory: Callable[[], Retriever], fallback_factory: Callable[[], Retriever] | None,
                 primary_name: str, primary_version: str, fallback_name: str | None = None,
                 config: ServiceConfig | None = None, events: EventLog | None = None,
                 clock: Callable[[], float] = time.time):
        self.cfg, self.events, self.clock = config or ServiceConfig(), events or EventLog(), clock
        self._primary_factory, self._fallback_factory = primary_factory, fallback_factory
        self._primary: Retriever | None = None
        self._fallback: Retriever | None = None
        self._lock = threading.RLock()
        self.status = ServiceStatus(primary=primary_name, primary_version=primary_version, fallback=fallback_name, since=clock())
        if fallback_factory is None:
            self.status.reason = "no independent fallback (production model is the lexical baseline)"
        self._ensure_primary(initial=True)

    # ---- component management ----
    def _ensure_primary(self, initial: bool = False) -> Retriever | None:
        if self._primary is None:
            try:
                self._primary = self._primary_factory()
            except Exception as e:
                self._record_failure(f"primary failed to load: {type(e).__name__}: {e}", force_fallback=True)
        return self._primary

    def _get_fallback(self) -> Retriever | None:
        if self._fallback is None and self._fallback_factory is not None:
            try:
                self._fallback = self._fallback_factory()
            except Exception as e:
                log.error("fallback failed to load: %s", e)
                self.events.emit("fallback_load_failed", error=str(e))
        return self._fallback

    # ---- state transitions ----
    def _record_failure(self, reason: str, force_fallback: bool = False) -> None:
        st = self.status
        st.consecutive_failures += 1
        st.total_failures += 1
        self.events.emit("primary_failure", reason=reason, consecutive=st.consecutive_failures)
        log.error("primary retrieval failure (%d): %s", st.consecutive_failures, reason)
        if (force_fallback or st.consecutive_failures >= self.cfg.failure_threshold) and st.mode == "primary" \
                and self._fallback_factory is not None:
            st.mode, st.fallback_active, st.reason, st.since = "fallback", True, reason, self.clock()
            st.recovery_attempts, st.healthy_probes = 0, 0
            st.next_recovery_at = self.clock() + self.cfg.cooldown_base_s
            self.events.emit("fallback_activated", reason=reason, fallback=st.fallback)
            log.warning("FALLBACK ACTIVATED -> %s (%s)", st.fallback, reason)

    def _activate_primary(self) -> None:
        st = self.status
        self.events.emit("primary_restored", after_s=round(self.clock() - st.since, 1), attempts=st.recovery_attempts)
        log.warning("primary %s restored", st.primary)
        st.mode, st.fallback_active, st.reason, st.since = "primary", False, None, self.clock()
        st.consecutive_failures, st.healthy_probes, st.next_recovery_at = 0, 0, None

    # ---- serving ----
    def retrieve(self, query: str, k: int = 10, allow: Callable[[str], bool] | None = None) -> RetrievalResult:
        """The lock guards only state (mode, counters, lazy loading). The search itself runs outside it so concurrent
        requests execute in parallel (FAISS reads, TF-IDF scoring and torch inference are safe to call concurrently)."""
        t0 = time.perf_counter()
        st = self.status
        with self._lock:
            st.total_requests += 1
            prim = self._ensure_primary() if st.mode == "primary" else None
            use_primary = prim is not None and st.mode == "primary"
            err = None if use_primary else (st.reason or ("primary unavailable" if st.mode == "primary" else "fallback active"))
        if use_primary:
            try:
                hits = prim.retrieve(query, k, allow)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                with self._lock:
                    self._record_failure(err)
            else:
                with self._lock:
                    st.consecutive_failures = 0
                return RetrievalResult(hits, "primary", prim.name, prim.kind, False, (time.perf_counter() - t0) * 1000)
        with self._lock:
            fb = self._get_fallback()
            if fb is not None:
                st.fallback_requests += 1
        if fb is None:
            raise RetrievalUnavailable(f"primary unavailable ({err}) and no fallback available")
        hits = fb.retrieve(query, k, allow)
        return RetrievalResult(hits, "fallback", fb.name, fb.kind, True, (time.perf_counter() - t0) * 1000, err)

    # ---- monitoring / recovery (called by a scheduler or on demand) ----
    def check_health(self, force: bool = False) -> dict:
        """Probe the primary. In PRIMARY mode a failed probe counts as a failure; in FALLBACK mode this is a
        recovery attempt (respecting the cooldown unless force=True)."""
        with self._lock:
            st = self.status
            if st.mode == "primary":
                prim = self._ensure_primary()
                if prim is None:
                    return {"ok": False, "detail": "primary not loaded", "mode": st.mode}
                rep = probe(prim)
                st.last_probe = {"ok": rep.ok, "detail": rep.detail, "latency_ms": round(rep.latency_ms, 1), "at": self.clock()}
                if rep.ok:
                    st.consecutive_failures = 0
                else:
                    self._record_failure(f"health probe failed: {rep.detail}")
                return {"ok": rep.ok, "detail": rep.detail, "mode": st.mode}
            if not force and st.next_recovery_at and self.clock() < st.next_recovery_at:
                return {"ok": False, "detail": "cooling down", "mode": st.mode, "retry_in_s": round(st.next_recovery_at - self.clock(), 1)}
            st.recovery_attempts += 1
            try:
                if self._primary is None:
                    self._primary = self._primary_factory()          # retry loading
                rep = probe(self._primary)
            except Exception as e:
                self._primary = None
                rep = type("R", (), {"ok": False, "detail": f"{type(e).__name__}: {e}", "latency_ms": 0.0})()
            st.last_probe = {"ok": rep.ok, "detail": rep.detail, "latency_ms": round(rep.latency_ms, 1), "at": self.clock()}
            if rep.ok:
                st.healthy_probes += 1
                self.events.emit("recovery_probe_ok", n=st.healthy_probes, needed=self.cfg.recovery_checks)
                if st.healthy_probes >= self.cfg.recovery_checks:
                    self._activate_primary()
                    return {"ok": True, "detail": "primary restored", "mode": st.mode}
                st.next_recovery_at = self.clock() + self.cfg.cooldown_base_s
            else:
                st.healthy_probes = 0
                delay = min(self.cfg.cooldown_base_s * (2 ** st.recovery_attempts), self.cfg.cooldown_max_s)
                st.next_recovery_at = self.clock() + delay
                self.events.emit("recovery_probe_failed", detail=rep.detail, next_in_s=delay)
            return {"ok": rep.ok, "detail": rep.detail, "mode": st.mode}

    # ---- safe model switching ----
    def switch_primary(self, new_factory: Callable[[], Retriever], name: str, version: str) -> tuple[bool, str]:
        """Build and validate the new primary BEFORE swapping; on any failure the working one is kept."""
        try:
            new = new_factory()
            rep = probe(new)
        except Exception as e:
            self.events.emit("model_switch_failed", to=name, error=str(e))
            return False, f"could not load {name}: {e}"
        if not rep.ok:
            self.events.emit("model_switch_failed", to=name, error=rep.detail)
            return False, f"{name} failed validation: {rep.detail}"
        with self._lock:
            old = self.status.primary
            self._primary, self._primary_factory = new, new_factory
            self.status.primary, self.status.primary_version = name, version
            self._activate_primary() if self.status.mode == "fallback" else None
            self.status.consecutive_failures = 0
            self.events.emit("model_switched", old=old, new=name, version=version)
        return True, f"switched {old} -> {name}"

    def reload(self) -> tuple[bool, str]:
        """Rebuild the primary (e.g. after the indexes were updated) and drop the cached fallback; the new primary must pass
        a probe before it replaces the old one."""
        ok, msg = self.switch_primary(self._primary_factory, self.status.primary, self.status.primary_version)
        if ok:
            with self._lock:
                self._fallback = None
        return ok, msg

    def evidence_signal(self, query: str, chunk_ids: list[str]) -> tuple[float | None, str | None]:
        """(signal, primary_version) from the PRIMARY model only; (None, None) when it is unavailable or has no dense part."""
        with self._lock:
            if self.status.mode != "primary" or self._primary is None:
                return None, None
            try:
                return self._primary.evidence_signal(query, chunk_ids), self.status.primary_version
            except Exception as e:
                log.warning("evidence signal unavailable: %s", e)
                return None, None

    def snapshot(self) -> dict:
        with self._lock:
            return self.status.to_dict()
