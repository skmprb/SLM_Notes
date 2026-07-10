"""
Phase 13: Observability — LangChain Comparison

| Our Code                  | LangChain Equivalent                          |
|---------------------------|-----------------------------------------------|
| Tracer / Span             | LangSmith tracing / OpenTelemetry spans       |
| MetricsCollector          | CallbackHandler + custom metrics              |
| CostTracker               | get_openai_callback() token tracking          |
| MODEL_PRICING dict        | LangSmith cost tracking                       |
| get_trace(trace_id)       | LangSmith run tree view                       |
| P50/P95/P99 latency      | Prometheus histograms / Grafana               |
| Correlation IDs           | RunnableConfig metadata                       |
"""
