# Research Paper: Amazon Bedrock AgentCore — Features, Capabilities & Comparison

## Abstract

Amazon Bedrock AgentCore is an enterprise-grade agentic platform for building, deploying, and operating AI agents securely at scale. Unlike traditional managed agent services, AgentCore is framework-agnostic, model-agnostic, and protocol-agnostic — enabling developers to use any open-source framework (LangGraph, CrewAI, Strands, LlamaIndex), any foundation model (Bedrock, OpenAI, Gemini), and any protocol (MCP, A2A) while gaining enterprise security, observability, and governance. This paper provides an in-depth analysis of all 12 core services and compares AgentCore with alternative approaches.

---

## 1. Introduction

### 1.1 What is AgentCore?

Amazon Bedrock AgentCore is a modular platform that solves the "last mile" problem of taking AI agents from prototype to production. It provides:

- **Infrastructure**: Serverless runtime with microVM session isolation
- **Tools**: Gateway, Browser, Code Interpreter for agent capabilities
- **Memory**: Short-term and long-term context management
- **Security**: Identity management, policy enforcement, credential handling
- **Operations**: Observability, evaluations, registry for production monitoring
- **Payments**: Microtransaction support for paid APIs via x402 protocol

### 1.2 Design Philosophy

- **Framework-agnostic**: Works with CrewAI, LangGraph, LlamaIndex, Google ADK, OpenAI Agents SDK, Strands Agents, or custom code
- **Model-agnostic**: Any LLM — Amazon Bedrock, OpenAI, Google Gemini, Anthropic Claude, Meta Llama, Mistral
- **Protocol-native**: First-class support for Model Context Protocol (MCP) and Agent-to-Agent (A2A)
- **Modular**: Each service works independently or together
- **Consumption-based pricing**: No upfront commitments, pay only for resources consumed

---

## 2. Core Services — Detailed Analysis

### 2.1 AgentCore Runtime

The foundational hosting layer for AI agents and tools.

| Feature | Details |
|---------|---------|
| **Architecture** | Serverless, microVM-based session isolation |
| **Session Isolation** | Dedicated microVM per user session (CPU, memory, filesystem) |
| **Execution Time** | Real-time (15 min sync) + Long-running (up to 8 hours async) |
| **Payload Size** | Up to 100 MB per request/response |
| **Streaming** | HTTP API + WebSocket bidirectional streaming |
| **Persistent Filesystem** | Files, packages, artifacts survive session stop/resume |
| **Hardware per Session** | 2 vCPU / 8 GB RAM |
| **Storage per Session** | Up to 1 GB |
| **Docker Image Size** | Up to 2 GB |
| **Cold Start** | Fast cold starts optimized for real-time interactions |
| **Protocols** | MCP, A2A, AG-UI |
| **Deployment** | Container image OR direct code deployment (250 MB compressed) |

**Key Differentiators:**
- Memory sanitization after session termination (deterministic security for non-deterministic AI)
- CPU billing aligned with active processing — no charges during I/O wait (e.g., waiting for LLM responses)
- Shell command execution within sessions
- Versioning and endpoint (alias) management

### 2.2 AgentCore Harness

A managed agent loop — define and invoke agents with a single API call.

| Feature | Details |
|---------|---------|
| **Purpose** | Zero-infrastructure agent orchestration |
| **Input** | Model + System Prompt + Tools (inline) |
| **Handles** | Orchestration, tool execution, memory management, response generation |
| **Isolation** | Each session in isolated microVM with filesystem + shell access |
| **Custom Environments** | Bring your own container image |
| **Model Support** | Amazon Bedrock, OpenAI, Google Gemini, any OpenAI-compatible provider |
| **Integrations** | Memory, Gateway, Browser, Code Interpreter, Observability |
| **Tool Types** | Remote MCP servers, inline functions, custom containers |

**Use Cases:** Code generation, data analysis, deep research, document processing

### 2.3 AgentCore Memory

Context-aware agents with full control over what they remember.

| Feature | Details |
|---------|---------|
| **Short-term Memory** | Multi-turn conversation context within a session |
| **Long-term Memory** | Persists across sessions, learns from experiences |
| **Shared Memory** | Memory stores shareable across multiple agents |
| **Memory Strategies** | Up to 6 per resource (extraction, consolidation, custom) |
| **Max Resources** | 150 per Region per account |
| **Event Size** | Up to 10 MB per event |
| **Messages per Event** | Up to 100 |
| **Token Limit** | 150,000 TPM for long-term extraction |
| **Expiration** | 7–365 days configurable |
| **Framework Support** | LangGraph, LangChain, Strands, LlamaIndex |

### 2.4 AgentCore Gateway

Convert APIs into MCP-compatible tools for agents.

| Feature | Details |
|---------|---------|
| **Purpose** | Transform APIs, Lambda functions, services into MCP tools |
| **Protocol** | Model Context Protocol (MCP) compatible endpoints |
| **Sources** | Any API, Lambda, OpenAPI specs, existing MCP servers |
| **Integrations** | Salesforce, Zoom, JIRA, Slack, GitHub, and more |
| **Authorization** | SigV4, OAuth, API keys |
| **Policy Integration** | Intercepts every tool call for policy enforcement |
| **Gateways per Account** | 1,000 |
| **Endpoint Format** | `https://{gateway-Id}.gateway.bedrock-agentcore.{Region}.amazonaws.com` |

### 2.5 AgentCore Identity

Centralized identity management for AI agents (non-human identities).

| Feature | Details |
|---------|---------|
| **Purpose** | Agent identity, access, and authentication management |
| **Authentication** | OAuth 2.0, SigV4, API keys, JWT authorizers |
| **Identity Providers** | Amazon Cognito, Okta, Microsoft Entra ID, Auth0, any IdP |
| **Workload Identities** | Up to 1,000 per account per Region |
| **OAuth2 Providers** | Up to 50 per account |
| **API Key Providers** | Up to 50 per account |
| **Inbound Auth** | End users authenticate into agents they have access to |
| **Outbound Auth** | Agents securely access third-party services (Slack, GitHub, Zoom) |
| **Modes** | On behalf of user OR autonomous agent operation |

### 2.6 AgentCore Code Interpreter

Isolated sandbox for agents to execute code.

| Feature | Details |
|---------|---------|
| **Purpose** | Enhance agent accuracy by executing code for complex tasks |
| **Languages** | Python, JavaScript, TypeScript |
| **Runtimes** | Node.js, Deno |
| **Isolation** | Sandboxed execution environment |
| **Integration** | Strands framework, Boto3 SDK, Bedrock Runtime |
| **Setup** | Console, CLI, or SDK |

### 2.7 AgentCore Browser

Cloud-based browser for web interaction.

| Feature | Details |
|---------|---------|
| **Purpose** | Enable agents to interact with web applications |
| **Capabilities** | Navigate websites, fill forms, extract information, click elements |
| **CAPTCHA** | Reduction support |
| **Extensions** | Browser extension support |
| **Session Profiles** | Configurable browser profiles |
| **Proxy Support** | Custom proxy configuration |
| **Root CA** | Custom certificate authority configuration |
| **OS-Level Actions** | Supported |
| **Frameworks** | Playwright, BrowserUse |
| **Model Support** | Any foundation model |

### 2.8 AgentCore Observability

Production monitoring and debugging for agents.

| Feature | Details |
|---------|---------|
| **Purpose** | Trace, debug, and monitor agent performance |
| **Format** | OpenTelemetry (OTEL) compatible |
| **Visualization** | Detailed step-by-step agent workflow views |
| **Capabilities** | Inspect execution paths, audit outputs, debug bottlenecks |
| **Integration** | Amazon CloudWatch, ADOT SDK, Transaction Search |
| **Scope** | Runtime, Memory, Gateway, Built-in Tools, Identity |
| **External Agents** | Supports agents hosted outside AgentCore |
| **Custom Headers** | Enhanced observability via custom headers |

### 2.9 AgentCore Evaluations

Automated agent quality assessment.

| Feature | Details |
|---------|---------|
| **Purpose** | Measure agent task execution, edge case handling, output reliability |
| **Input** | Sessions, traces, spans from Strands or LangGraph |
| **Instrumentation** | OpenTelemetry or OpenInference |
| **Output** | Measurable quality signals, structured insights |
| **Integration** | CloudWatch via AgentCore Observability |
| **Timing** | Pre-deployment and post-deployment evaluation |

### 2.10 AgentCore Policy

Deterministic governance for agent behavior.

| Feature | Details |
|---------|---------|
| **Purpose** | Ensure agents operate within defined boundaries |
| **Authoring** | Natural language OR Cedar (AWS open-source policy language) |
| **Enforcement** | Intercepts every tool call before execution via Gateway |
| **Controls** | Which tools agents can access, what actions they perform, under what conditions |
| **Speed** | Deterministic control without slowing agents down |

### 2.11 AgentCore Payments

Microtransaction support for AI agents.

| Feature | Details |
|---------|---------|
| **Purpose** | Enable agents to access paid APIs, MCP servers, content |
| **Protocol** | x402 |
| **Wallet Providers** | Coinbase CDP, Stripe (Privy) |
| **Features** | Configurable spending limits, end-to-end observability |
| **Integration** | Gateway, Strands Agents, Identity, Observability |

### 2.12 AgentCore Registry

Centralized catalog for agent resources.

| Feature | Details |
|---------|---------|
| **Purpose** | Discover and manage agents, MCP servers, tools, skills |
| **Search** | Hybrid semantic + keyword search |
| **Governance** | Publish → Review → Approve workflow |
| **Resources** | MCP Servers, Agents, Skills, Custom Resources |
| **Deployment** | AWS, On-Prem, or any other cloud environment |

---

## 3. Service Quotas Summary

### 3.1 Runtime Quotas

| Resource | Limit | Adjustable |
|----------|-------|------------|
| Active sessions per account | 1,000 (us-east-1, us-west-2) / 500 (other) | Yes |
| Agents per account | 1,000 | Yes |
| Versions per agent | 1,000 | Yes |
| Endpoints per agent | 10 | Yes |
| InvokeAgentRuntime rate | 25 TPS per agent | Yes |
| Sync request timeout | 15 minutes | No |
| Async job duration | 8 hours | No |
| Payload size | 100 MB | No |
| Streaming duration | 60 minutes | No |
| Session storage | 1 GB | No |
| Hardware per session | 2 vCPU / 8 GB | No |
| Idle session timeout | 15 minutes | Yes (API) |
| Max session duration | 8 hours | Yes (API) |

### 3.2 Memory Quotas

| Resource | Limit | Adjustable |
|----------|-------|------------|
| Memory resources per Region | 150 | Yes |
| Strategies per resource | 6 | No |
| Long-term extraction TPM | 150,000 | Yes |
| Event size | 10 MB | No |
| Messages per event | 100 | No |
| Event expiration | 7–365 days | No |

### 3.3 Identity Quotas

| Resource | Limit |
|----------|-------|
| Workload identities | 1,000 per Region |
| OAuth2 credential providers | 50 per Region |
| API key credential providers | 50 per Region |

---

## 4. Comparison: AgentCore vs Bedrock Agents vs Other Platforms

### 4.1 AgentCore vs Amazon Bedrock Agents

| Feature | AgentCore | Bedrock Agents |
|---------|-----------|----------------|
| **Framework** | Any (LangGraph, CrewAI, Strands, custom) | Bedrock-native only |
| **Model** | Any LLM (Bedrock, OpenAI, Gemini, etc.) | Bedrock models only |
| **Protocol** | MCP, A2A, AG-UI | Proprietary |
| **Session Isolation** | microVM per session | Shared infrastructure |
| **Execution Time** | Up to 8 hours | Minutes (with timeouts) |
| **Memory** | Short-term + Long-term + Shared | Conversation history only |
| **Code Execution** | Code Interpreter (Python, JS, TS) | Limited code interpretation |
| **Browser** | Full browser automation | Not available |
| **Identity** | Full IdP integration (Okta, Entra, Cognito) | IAM only |
| **Policy** | Cedar-based fine-grained policies | Guardrails (content filtering) |
| **Observability** | OpenTelemetry native | CloudWatch + Trace |
| **Payments** | x402 microtransactions | Not available |
| **Registry** | Centralized catalog | Not available |
| **Deployment** | Container or direct code | Managed (no custom deployment) |
| **Pricing** | Consumption-based (active CPU) | Per-invocation + token-based |
| **Best For** | Complex, long-running, multi-framework agents | Simple, quick-to-deploy agents |

### 4.2 AgentCore vs Self-Managed (EC2/ECS/Lambda)

| Feature | AgentCore | Self-Managed |
|---------|-----------|--------------|
| **Infrastructure** | Fully managed, serverless | You manage everything |
| **Session Isolation** | Automatic microVM | Must implement yourself |
| **Scaling** | Automatic | Manual or auto-scaling config |
| **Cold Start** | Optimized | Depends on implementation |
| **Memory Management** | Built-in service | Build your own |
| **Identity/Auth** | Built-in service | Implement with Cognito/custom |
| **Observability** | Built-in OTEL | Instrument manually |
| **Policy Enforcement** | Built-in Cedar | Implement custom logic |
| **MCP Support** | Native | Implement yourself |
| **Cost Model** | Pay for active CPU only | Pay for allocated resources |
| **Time to Production** | Hours/Days | Weeks/Months |

### 4.3 AgentCore vs Third-Party Platforms

| Feature | AgentCore | LangSmith/LangServe | Vercel AI SDK |
|---------|-----------|--------------------:|---------------|
| **Hosting** | AWS serverless microVM | Cloud/self-hosted | Vercel Edge |
| **Isolation** | microVM per session | Process-level | Request-level |
| **Max Execution** | 8 hours | Varies | Seconds/minutes |
| **Memory** | Built-in (short+long) | LangMem (separate) | None built-in |
| **Browser** | Built-in | Not available | Not available |
| **Identity** | Enterprise IdP integration | Basic auth | Vercel auth |
| **Policy** | Cedar-based governance | Not available | Not available |
| **Payments** | x402 native | Not available | Not available |
| **AWS Integration** | Native | Via SDKs | Via SDKs |
| **Compliance** | SOC, HIPAA eligible | Varies | Limited |

---

## 5. Architecture Patterns

### 5.1 Single Agent Pattern

```
User → AgentCore Runtime (microVM) → FM (any model)
                ↓
        AgentCore Gateway → Tools (APIs, MCP servers)
                ↓
        AgentCore Memory → Context persistence
```

### 5.2 Multi-Agent Pattern

```
User → Supervisor Agent (Runtime)
            ├── Worker Agent 1 (Runtime) → Specialized tools
            ├── Worker Agent 2 (Runtime) → Knowledge retrieval
            └── Worker Agent 3 (Runtime) → Code execution
                    ↓
            AgentCore Registry (discover agents/tools)
            AgentCore Policy (enforce boundaries)
            AgentCore Observability (trace all)
```

### 5.3 Enterprise Platform Pattern

```
Developer Teams → AgentCore Registry (publish/discover)
                        ↓
                AgentCore Gateway (centralized tool access)
                        ↓
                AgentCore Identity (SSO + credential management)
                        ↓
                AgentCore Policy (governance)
                        ↓
                AgentCore Runtime (deploy agents)
                        ↓
                AgentCore Observability (monitor)
                AgentCore Evaluations (quality gates)
```

---

## 6. Security Model

| Layer | Mechanism |
|-------|-----------|
| **Session Isolation** | Dedicated microVM per session, memory sanitized on termination |
| **Authentication** | OAuth 2.0, SigV4, JWT, API keys via AgentCore Identity |
| **Authorization** | Cedar policies via AgentCore Policy |
| **Network** | VPC integration, private endpoints |
| **Encryption** | TLS in transit, KMS at rest |
| **Credential Management** | Centralized via Identity service, no hardcoded secrets |
| **Audit** | Full trace via Observability + CloudTrail |
| **Compliance** | GovCloud support, FIPS endpoints available |

---

## 7. Pricing Model

AgentCore uses consumption-based pricing:

| Component | Pricing Basis |
|-----------|---------------|
| **Runtime** | Active CPU time (no charge during I/O wait) |
| **Memory** | Per API call + storage |
| **Gateway** | Per request |
| **Identity** | Per authentication event |
| **Code Interpreter** | Per execution |
| **Browser** | Per session |
| **Observability** | CloudWatch pricing |
| **Evaluations** | Per evaluation run |

No upfront commitments or minimum fees.

---

## 8. Supported Regions

AgentCore is available in multiple AWS Regions including:
- US East (N. Virginia) — us-east-1
- US West (Oregon) — us-west-2
- Europe (Frankfurt, Ireland, London)
- Asia Pacific (Tokyo, Sydney, Singapore, Mumbai)
- AWS GovCloud (US-East, US-West)

---

## 9. Conclusion

Amazon Bedrock AgentCore represents a paradigm shift from "managed agents" to "agent infrastructure as a service." By being framework-agnostic, model-agnostic, and protocol-native, it eliminates vendor lock-in while providing enterprise-grade security (microVM isolation), governance (Cedar policies), and operations (OpenTelemetry observability). The 12 modular services can be adopted incrementally — from just using Runtime for hosting, to building a full enterprise agent platform with Registry, Policy, Identity, and Evaluations. For organizations building production AI agents that need security, compliance, and operational excellence, AgentCore provides the most comprehensive platform available on AWS.

---

## References

1. AgentCore Overview: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html
2. AgentCore Runtime: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
3. AgentCore Quotas: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html
4. AgentCore Identity: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-overview.html
5. AgentCore Pricing: https://aws.amazon.com/bedrock/agentcore/pricing/
6. AWS Prescriptive Guidance - AgentCore: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/amazon-bedrock-agentcore.html

---

*Paper generated based on AWS official documentation. Features and availability are subject to change.*
