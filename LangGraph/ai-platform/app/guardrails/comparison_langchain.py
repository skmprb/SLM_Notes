"""
Phase 12: Guardrails — LangChain Comparison

| Our Code                    | LangChain / Industry Equivalent               |
|-----------------------------|-----------------------------------------------|
| GuardrailPipeline           | NeMo Guardrails / Guardrails AI               |
| PIIDetector                 | Presidio / custom regex                       |
| PromptInjectionDetector     | Rebuff / LLM-based detection                  |
| ContentLengthGuard          | max_tokens validation                         |
| check_input() / check_output() | RunnablePassthrough with validation        |
| GuardrailAction.REDACT      | Presidio anonymize()                          |
| GuardrailAction.BLOCK       | Raise exception / return error                |
"""
