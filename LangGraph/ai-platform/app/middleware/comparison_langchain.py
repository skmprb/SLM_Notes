"""
Phase 14: Rate Limiting & Multi-tenancy — LangChain Comparison

| Our Code                  | LangChain / Industry Equivalent               |
|---------------------------|-----------------------------------------------|
| RateLimiter (TokenBucket) | No built-in (infra: API Gateway, Kong, nginx) |
| TenantManager             | LangGraph Platform assistants / custom        |
| TenantConfig (tiers)      | LangSmith organizations / workspaces          |
| check_access()            | RBAC middleware                                |
| Per-tenant rate limits    | API Gateway per-key throttling                |
| Budget tracking           | LangSmith usage limits                        |
| TIER_DEFAULTS             | Stripe subscription tiers                     |
"""
