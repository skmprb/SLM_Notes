"""
Observability (Phase 13) - Tracing, metrics, token/cost tracking.

Three pillars:
    1. Tracing: Follow a request through all components (correlation IDs)
    2. Metrics: Latency, token usage, error rates, cache hit rates
    3. Cost tracking: Per-request, per-user, per-model cost accounting

Production concerns:
    - Correlation IDs across all log lines
    - Structured logging (JSON format for log aggregation)
    - Metric aggregation (P50, P95, P99 latency)
    - Cost budgets and alerts
    - Dashboard-ready data export

LangChain equivalent:
    - LangSmith tracing (traces, runs, feedback)
    - CallbackHandler for metrics
    - OpenTelemetry integration
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from contextlib import contextmanager

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================
# TRACING
# ============================================================

@dataclass
class Span:
    """A single operation within a trace."""
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return None

    def finish(self, status: str = "ok", **metadata):
        self.end_time = time.time()
        self.status = status
        self.metadata.update(metadata)


class Tracer:
    """Simple request tracer with spans."""

    def __init__(self):
        self._traces: dict[str, list[Span]] = {}

    @contextmanager
    def span(self, name: str, trace_id: Optional[str] = None, **metadata):
        """Context manager for creating a traced span."""
        tid = trace_id or str(uuid.uuid4())[:12]
        s = Span(name=name, trace_id=tid, metadata=metadata)

        if tid not in self._traces:
            self._traces[tid] = []
        self._traces[tid].append(s)

        try:
            yield s
        except Exception as e:
            s.finish(status="error", error=str(e))
            raise
        else:
            s.finish()

    def get_trace(self, trace_id: str) -> list[dict]:
        spans = self._traces.get(trace_id, [])
        return [
            {"name": s.name, "duration_ms": s.duration_ms, "status": s.status, "metadata": s.metadata}
            for s in spans
        ]


# ============================================================
# METRICS COLLECTOR
# ============================================================

@dataclass
class MetricPoint:
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates platform metrics."""

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def record(self, name: str, value: float) -> None:
        """Record a value in a histogram (for percentile calculations)."""
        self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def get_summary(self) -> dict:
        """Get all metrics as a summary dict."""
        summary = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
        }
        for name, values in self._histograms.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                summary["histograms"][name] = {
                    "count": n,
                    "min": sorted_v[0],
                    "max": sorted_v[-1],
                    "avg": round(sum(sorted_v) / n, 2),
                    "p50": sorted_v[int(n * 0.5)],
                    "p95": sorted_v[int(n * 0.95)] if n >= 20 else sorted_v[-1],
                    "p99": sorted_v[int(n * 0.99)] if n >= 100 else sorted_v[-1],
                }
        return summary


# ============================================================
# COST TRACKER
# ============================================================

# Pricing per 1M tokens (approximate, as of 2024)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


class CostTracker:
    """Track token usage and costs per model/user."""

    def __init__(self):
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._cost_by_model: dict[str, float] = defaultdict(float)
        self._requests: int = 0

    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
        """Record token usage and return cost for this request."""
        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 2.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._cost_by_model[model] += cost
        self._requests += 1

        return round(cost, 6)

    def get_summary(self) -> dict:
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_requests": self._requests,
            "cost_by_model": {k: round(v, 4) for k, v in self._cost_by_model.items()},
            "avg_cost_per_request": round(self._total_cost / self._requests, 6) if self._requests else 0,
        }


# ============================================================
# GLOBAL INSTANCES
# ============================================================

_tracer: Optional[Tracer] = None
_metrics: Optional[MetricsCollector] = None
_cost_tracker: Optional[CostTracker] = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
