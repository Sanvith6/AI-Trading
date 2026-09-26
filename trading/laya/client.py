"""
Laya Local Client & Bounded Concurrency Transport.

Enforces Directives 5, 6, 9, and 21:
- Bounded concurrency: max workers capped (default 2). One slow candidate cannot stall the scanner.
- Latency budget tracking (P50, P95, P99) across:
    1. feature formatting
    2. Laya request
    3. Laya inference
    4. response validation
    5. total decision path
- Fail-safe fallback integration with circuit breaker and GPU monitor.
- Exact cache integration for historical reproducibility.
"""

from typing import Dict, Any, List, Optional
import urllib.request
import json
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from .schemas import LayaMarketState, LayaDecision, ResponseValidator
from .tasks import LayaTasks
from .cache import laya_cache
from .fallback import laya_circuit_breaker, LayaFailSafeEngine
from .gpu_monitor import gpu_monitor
from .config import laya_config

logger = logging.getLogger(__name__)


class LayaClient:
    """Thread-safe bounded client communicating with local Laya SystemOne service."""

    def __init__(self, config=None):
        self.config = config or laya_config
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_workers)
        self.latency_metrics: Dict[str, List[float]] = {
            "feature_formatting": [],
            "laya_request": [],
            "laya_inference": [],
            "response_validation": [],
            "total_decision_path": [],
        }

    def record_latency(self, stage: str, duration_ms: float) -> None:
        """Records stage latency for percentiles calculation."""
        if stage in self.latency_metrics:
            self.latency_metrics[stage].append(duration_ms)
            # Limit memory footprint
            if len(self.latency_metrics[stage]) > 2000:
                self.latency_metrics[stage] = self.latency_metrics[stage][-1000:]

    def get_latency_percentiles(self, stage: str = "total_decision_path") -> Dict[str, float]:
        """Calculates P50, P95, P99 latency percentiles for the specified pipeline stage (Directive 6)."""
        vals = self.latency_metrics.get(stage, [])
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}

        arr = np.array(vals)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "count": len(arr),
        }

    def predict_tasks(
        self,
        market_state: LayaMarketState,
        tasks: List[str],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, LayaDecision]:
        """
        Executes inference for the requested tasks on a single candidate.
        Checks cache -> Checks circuit breaker -> Checks GPU health -> Executes HTTP -> Validates -> Caches.
        """
        t_total_start = time.perf_counter()
        state_hash = market_state.compute_state_hash()
        decisions: Dict[str, LayaDecision] = {}
        pending_tasks = []

        # 1. Check Decision Cache (Directive 4)
        for task in tasks:
            cached = laya_cache.get(state_hash, task)
            if cached is not None:
                decisions[task] = cached
            else:
                pending_tasks.append(task)

        if not pending_tasks:
            total_duration_ms = (time.perf_counter() - t_total_start) * 1000
            self.record_latency("total_decision_path", total_duration_ms)
            return decisions

        # 2. Check Circuit Breaker & GPU Safety Guardrails (Directives 9 & 21)
        if not laya_circuit_breaker.can_attempt():
            for t in pending_tasks:
                decisions[t] = LayaFailSafeEngine.create_fallback_decision(
                    t, state_hash, "Circuit breaker OPEN: Laya service currently paused"
                )
            return decisions

        if not gpu_monitor.is_safe_for_inference():
            for t in pending_tasks:
                decisions[t] = LayaFailSafeEngine.create_fallback_decision(
                    t, state_hash, "GPU resources constrained: degrading safely to baseline"
                )
            return decisions

        # 3. Format Request Payload
        t_format_start = time.perf_counter()
        narrative = market_state.to_narrative_context()
        task_payload = LayaTasks.get_task_payload(pending_tasks)
        payload = {
            "state": narrative,
            "questions": task_payload,
        }
        self.record_latency("feature_formatting", (time.perf_counter() - t_format_start) * 1000)

        # 4. Invoke Service with strict timeout
        timeout = timeout_seconds or self.config.request_timeout_seconds
        t_req_start = time.perf_counter()
        raw_response = None
        req_latency_ms = 0.0

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.endpoint,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_bytes = response.read()
                raw_response = json.loads(resp_bytes.decode("utf-8"))

            req_latency_ms = (time.perf_counter() - t_req_start) * 1000
            self.record_latency("laya_request", req_latency_ms)
            self.record_latency("laya_inference", req_latency_ms)
            laya_circuit_breaker.record_success()

        except Exception as e:
            req_latency_ms = (time.perf_counter() - t_req_start) * 1000
            err_msg = f"Laya inference error: {e}"
            logger.warning(err_msg)
            laya_circuit_breaker.record_failure(err_msg)

            for t in pending_tasks:
                decisions[t] = LayaFailSafeEngine.create_fallback_decision(
                    t, state_hash, err_msg, latency_ms=req_latency_ms
                )
            return decisions

        # 5. Response Validation & Caching (Directives 6, 8, 9)
        t_val_start = time.perf_counter()
        answers = raw_response.get("answers", {}) if isinstance(raw_response, dict) else {}

        # Check Latency Budget
        is_over_budget = req_latency_ms > self.config.latency_budget_ms
        if is_over_budget:
            logger.warning(
                f"Latency budget breach: {req_latency_ms:.1f}ms exceeds limit {self.config.latency_budget_ms:.1f}ms"
            )

        for t in pending_tasks:
            t_key = t.lower()
            raw_answer = answers.get(t_key)
            if not raw_answer:
                decisions[t] = LayaFailSafeEngine.create_fallback_decision(
                    t, state_hash, f"Missing answer key '{t_key}' in response", latency_ms=req_latency_ms
                )
                continue

            decision = ResponseValidator.validate(
                task=t,
                raw_answer=raw_answer,
                model_version=self.config.model_version,
                state_hash=state_hash,
                latency_ms=req_latency_ms,
            )

            if is_over_budget:
                decision.validation_error = f"Latency budget exceeded ({req_latency_ms:.1f}ms)"

            # Cache successful decision
            if decision.is_valid:
                laya_cache.put(decision)

            decisions[t] = decision

        self.record_latency("response_validation", (time.perf_counter() - t_val_start) * 1000)
        self.record_latency("total_decision_path", (time.perf_counter() - t_total_start) * 1000)

        return decisions

    def evaluate_candidates_bounded(
        self,
        candidates_with_states: List[LayaMarketState],
        tasks: List[str],
    ) -> List[Dict[str, LayaDecision]]:
        """
        Executes bounded parallel inference over Top N candidates.
        Ensures bounded concurrency so slow candidates never hang the entire pipeline.
        """
        # Enforce Top N limit (Directive 5)
        top_candidates = candidates_with_states[: self.config.max_candidates_per_scan]
        results = [None] * len(top_candidates)

        futures_map = {}
        for idx, state in enumerate(top_candidates):
            f = self.executor.submit(self.predict_tasks, state, tasks)
            futures_map[f] = idx

        for f in as_completed(futures_map):
            idx = futures_map[f]
            try:
                dec = f.result(timeout=self.config.request_timeout_seconds + 1.0)
                results[idx] = dec
            except Exception as e:
                state = top_candidates[idx]
                state_hash = state.compute_state_hash()
                logger.error(f"Bounded execution timeout/error on candidate {state.symbol}: {e}")
                err_decisions = {
                    t: LayaFailSafeEngine.create_fallback_decision(t, state_hash, f"Worker exception: {e}")
                    for t in tasks
                }
                results[idx] = err_decisions

        return results


# Global singleton client
laya_client = LayaClient()
