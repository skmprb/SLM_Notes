# Research Paper: Amazon Bedrock — Build Section

## Abstract

Amazon Bedrock provides a comprehensive suite of "Build" tools that enable developers to create production-ready generative AI applications. This paper covers the eight core Build capabilities: Agents, Flows, Knowledge Bases, Automated Reasoning, Guardrails, Prompt Management, Advanced Prompt Optimization, and Data Automation.

---

## 1. Introduction

The Build section of Amazon Bedrock provides the building blocks to go beyond simple model inference. These tools allow you to orchestrate multi-step AI workflows, ground responses in proprietary data, enforce safety policies, validate accuracy with formal logic, and automate document processing — all without managing infrastructure.

---

## 2. Amazon Bedrock Agents

### 2.1 Overview

Amazon Bedrock Agents enables you to build autonomous AI agents that can plan, reason, and take actions to complete tasks on behalf of users. Agents orchestrate interactions between foundation models, knowledge bases, and external APIs.

### 2.2 Architecture

Agents consist of two main API sets:
- **Build-time APIs**: Create, configure, and manage agents and their resources
- **Runtime APIs**: Invoke agents with user input and initiate orchestration

### 2.3 Core Components

| Component | Description |
|-----------|-------------|
| **Foundation Model** | The FM that interprets user input and drives orchestration decisions |
| **Instructions** | Natural language description of what the agent is designed to do |
| **Action Groups** | Define actions the agent can perform (via OpenAPI schema or function detail schema) |
| **Knowledge Bases** | Associated KBs the agent queries for additional context |
| **Prompt Templates** | 4 base templates: Pre-processing, Orchestration, KB Response Generation, Post-processing |

### 2.4 Action Groups

Action groups define what an agent can do. Each action group requires:
- **Schema**: OpenAPI schema (for API operations) OR Function detail schema (for parameter elicitation)
- **Lambda Function** (optional): Handles the API invocation and returns responses
- **Return Control**: Alternative to Lambda — returns parameters to the calling application

### 2.5 Runtime Process (Orchestration Loop)

```
User Input → Pre-processing → Orchestration Loop → Post-processing → Response
                                    ↓
                         [Interpret with FM]
                                    ↓
                    [Generate Rationale & Predict Action]
                                    ↓
                  [Invoke Action Group OR Query Knowledge Base]
                                    ↓
                         [Generate Observation]
                                    ↓
                    [Loop until task complete or need more info]
```

1. **Pre-processing**: Contextualizes/categorizes user input, validates it
2. **Orchestration** (iterative loop):
   - FM interprets input → generates rationale
   - Predicts which action to invoke or KB to query
   - Invokes Lambda function or returns control
   - Generates observation from action/KB results
   - Determines if more iterations needed
3. **Post-processing**: Formats final response (disabled by default)

### 2.6 Advanced Features

- **Multi-Agent Collaboration**: Supervisor-collaborator hierarchies with domain specialist sub-agents
- **Custom Orchestration**: Override default ReAct pattern with custom Lambda-based orchestration
- **Conversation History**: Automatically preserved across InvokeAgent requests within a session
- **Trace**: Track agent's rationale, actions, queries, and observations at each step
- **Code Interpretation**: Execute code within agent workflows
- **Guardrails Integration**: Apply safety filters to agent responses
- **Provisioned Throughput**: Use dedicated capacity for agent aliases

### 2.7 Orchestration Strategies

| Strategy | Description |
|----------|-------------|
| **Default (ReAct)** | Reason-Act-Observe loop using FM reasoning |
| **Custom Orchestration** | Lambda function controls the orchestration logic |
| **Single KB Optimization** | Optimized latency path for agents with one knowledge base |

---

## 3. Amazon Bedrock Flows

### 3.1 Overview

Amazon Bedrock Flows provides a visual builder to create end-to-end generative AI workflows by linking foundation models, prompts, knowledge bases, and AWS services into connected pipelines.

### 3.2 Key Capabilities

- **Visual Builder**: Drag-and-drop interface for designing workflows
- **Node-Based Architecture**: Connect different node types via data and conditional links
- **Immutable Versioning**: Publish snapshots of flows for production deployment
- **Alias-Based Deployment**: Route traffic to specific flow versions
- **Async Execution**: Long-running flows supporting up to 24-hour runtimes
- **Multi-Turn Conversations**: Converse with flows interactively

### 3.3 Node Types

| Node Type | Purpose |
|-----------|---------|
| **Input** | Entry point — receives user input |
| **Output** | Exit point — returns final response |
| **Prompt** | Invokes an FM with a configured prompt |
| **Knowledge Base** | Queries a knowledge base for information |
| **Lambda** | Executes an AWS Lambda function |
| **Condition** | Routes flow based on conditional logic |
| **Iterator** | Loops over a collection of items |
| **Collector** | Aggregates results from iterator |
| **Lex** | Integrates with Amazon Lex bots |
| **Agent** | Invokes a Bedrock Agent |

### 3.4 Workflow Lifecycle

```
Create → Design (add nodes/connections) → Prepare → Test → Version → Alias → Deploy
```

1. **Create**: Specify name, description, IAM permissions
2. **Design**: Add nodes, configure them, create connections
3. **Prepare**: Apply latest changes to working draft
4. **Test**: Invoke with sample inputs, validate outputs
5. **Version**: Publish immutable snapshot
6. **Alias**: Create alias pointing to a version
7. **Deploy**: Application calls `InvokeFlow` on the alias

### 3.5 Example Use Cases

- **Email automation**: Prompt → KB lookup → Lambda (send email)
- **Troubleshooting**: Error input → Documentation KB → System logs → Fix configuration
- **Report generation**: Metrics query → Aggregation → Summary → Publish
- **Data ingestion**: Dataset filtering → Processing → Failure reporting

### 3.6 Pricing

Flows pricing is based on the underlying resources used (model invocations, Lambda executions, KB queries, etc.). No separate Flows charge.

---

## 4. Amazon Bedrock Knowledge Bases

### 4.1 Overview

Knowledge Bases implement Retrieval Augmented Generation (RAG) to integrate proprietary data into generative AI applications. They search your data to find relevant information and use it to improve the accuracy and relevancy of model responses.

### 4.2 How It Works

```
User Query → Embedding → Vector Search → Retrieve Relevant Chunks → Augment Prompt → Generate Response
```

1. **Ingestion**: Documents are chunked, embedded into vectors, and stored in a vector database
2. **Retrieval**: User queries are embedded and matched against stored vectors
3. **Generation**: Retrieved context augments the FM prompt for accurate responses

### 4.3 Data Source Types

| Type | Description | Query Method |
|------|-------------|--------------|
| **Unstructured** | Documents (PDF, text, HTML, etc.) in S3 | Vector similarity search |
| **Structured** | Databases (SQL-compatible) | Natural language → SQL conversion |
| **Multimodal** | Documents with images and visual content | Image + text retrieval |
| **Graph** | Amazon Neptune Analytics graphs | Graph-based queries |
| **Kendra GenAI Index** | Amazon Kendra integration | Kendra search |

### 4.4 Supported Vector Stores

- Amazon OpenSearch Serverless (auto-created via console)
- Amazon Aurora PostgreSQL (pgvector)
- Amazon Neptune Analytics
- Amazon S3 Vectors
- Pinecone
- Redis Enterprise Cloud
- MongoDB Atlas

### 4.5 Key Features

- **Citations**: Responses include references to source documents
- **Multimodal Queries**: Search using images or text+image combinations
- **Reranking**: Use reranking models to improve retrieval quality
- **Real-time Sync**: Update data sources and ingest changes immediately
- **Agent Integration**: Attach knowledge bases to Bedrock Agents
- **Chat with Document**: Zero-setup document Q&A
- **Cross-Region Inference**: Use inference profiles for KB response generation

### 4.6 APIs

| API | Purpose |
|-----|---------|
| `Retrieve` | Return relevant sources without generating a response |
| `RetrieveAndGenerate` | Retrieve sources AND generate a natural language response |

### 4.7 Supported Embedding Models

- Amazon Titan Text Embeddings V2
- Amazon Titan Embeddings G1 - Text
- Amazon Titan Multimodal Embeddings G1
- Amazon Nova Multimodal Embeddings
- Cohere Embed models

---

## 5. Automated Reasoning

### 5.1 Overview

Automated Reasoning checks use mathematical formal logic to validate LLM responses against policies you define. Unlike content filters that simply block/allow content, Automated Reasoning provides structured feedback about *why* a response is correct or incorrect.

### 5.2 What It Does

- **Detects factually incorrect statements** by mathematically proving contradictions against policy rules
- **Highlights unstated assumptions** where responses are incomplete
- **Provides mathematically verifiable explanations** citing specific policy rules that support conclusions

### 5.3 End-to-End Workflow

```
Source Document → Extracted Policy (formal logic) → Testing → Deployment (guardrail) → Integration (validate responses)
```

1. **Create Policy**: Upload source document → AR extracts formal logic rules + variable schema → Fidelity report generated
2. **Test & Refine**: Create scenario tests and QnA tests → Validate rule correctness and NL-to-logic translation
3. **Deploy**: Save immutable version → Attach to guardrail → Automate via CloudFormation/CI/CD
4. **Integrate**: Runtime validation via `Converse`, `InvokeModel`, `InvokeAgent`, `RetrieveAndGenerate`, or `ApplyGuardrail` APIs

### 5.4 Key Concepts

| Concept | Description |
|---------|-------------|
| **Policy** | Set of formal logic rules extracted from a source document |
| **Rules** | Individual logical statements that define valid/invalid conditions |
| **Variables** | Parameters extracted from the policy that are evaluated at runtime |
| **Fidelity Report** | Measures accuracy of extracted policy vs. source document (coverage + accuracy scores) |
| **Findings** | Runtime validation results: VALID, INVALID, or UNSTATED_ASSUMPTION |
| **Confidence Thresholds** | Configurable sensitivity for validation |

### 5.5 When to Use

- Regulated industries (healthcare, finance, HR)
- Complex rule sets (mortgage approvals, insurance eligibility, zoning laws)
- Compliance scenarios requiring auditable AI responses
- Customer-facing applications where incorrect guidance erodes trust

### 5.6 Limitations

- English (US) only
- No streaming support — must validate complete responses
- No prompt injection protection (use Content Filters for that)
- No off-topic detection (use Topic Policies for that)
- Source documents limited to 5 MB / 50,000 characters
- Operates in **detect mode only** — returns findings, does not block content

### 5.7 Availability

US East (N. Virginia), US West (Oregon), US East (Ohio), EU (Frankfurt), EU (Paris), EU (Ireland)

---

## 6. Amazon Bedrock Guardrails

### 6.1 Overview

Guardrails provide configurable safeguards to detect and filter undesirable content, protect sensitive information, and enforce safety policies across all foundation models. They evaluate both user inputs and model responses.

### 6.2 Safeguard Components

| Filter | What It Does | Action |
|--------|-------------|--------|
| **Content Filters** | Detects harmful content: Hate, Insults, Sexual, Violence, Misconduct, Prompt Attack | Block/Detect |
| **Denied Topics** | Blocks user-defined undesirable topics | Block |
| **Word Filters** | Blocks custom words/phrases (exact match) + profanity | Block |
| **Sensitive Information Filters** | Detects/masks PII (SSN, DOB, Address, etc.) + custom regex patterns | Block/Mask |
| **Contextual Grounding Checks** | Detects hallucinations — responses not grounded in source or irrelevant to query | Block/Flag |
| **Automated Reasoning Checks** | Validates accuracy against formal logic rules | Detect (feedback) |

### 6.3 Content Filter Categories

| Category | Description | Configurable Strength |
|----------|-------------|----------------------|
| Hate | Discriminatory language based on identity | NONE / LOW / MEDIUM / HIGH |
| Insults | Demeaning or offensive language | NONE / LOW / MEDIUM / HIGH |
| Sexual | Sexually explicit content | NONE / LOW / MEDIUM / HIGH |
| Violence | Violent or threatening content | NONE / LOW / MEDIUM / HIGH |
| Misconduct | Criminal activity, harmful instructions | NONE / LOW / MEDIUM / HIGH |
| Prompt Attack | Jailbreaks, prompt injections, leakage | NONE / LOW / MEDIUM / HIGH |

### 6.4 Tiers

| Tier | Coverage |
|------|----------|
| **Classic** | Standard content filtering across predefined categories |
| **Standard** | Extended protection including harmful content within code elements (comments, variable names, string literals) |

### 6.5 Runtime Actions

| Action | Behavior |
|--------|----------|
| **Block** | Prevents content from being returned; shows configured blocked message |
| **Mask** | Replaces sensitive information with placeholders (e.g., `[PII]`) |
| **Detect** | Flags content but still returns it (for logging/monitoring) |

### 6.6 Integration Points

- `InvokeModel` / `InvokeModelWithResponseStream` — specify guardrail ID + version
- `Converse` / `ConverseStream` — specify guardrail configuration
- `ApplyGuardrail` — standalone API (no FM invocation needed)
- `InvokeAgent` — apply to agent responses
- `RetrieveAndGenerate` — apply to KB-generated responses
- Amazon Bedrock Flows — include guardrails in flow nodes

### 6.7 Deployment Workflow

1. Create guardrail (working draft)
2. Configure filters with appropriate strengths
3. Test with built-in test window
4. Create versioned snapshot
5. Apply to model invocations via guardrail ID + version

### 6.8 Key Features

- **Cross-Region distribution**: Distribute guardrail inference across AWS Regions
- **Cross-Account enforcement**: Apply guardrails across AWS accounts
- **Selective evaluation**: Use input tags to evaluate only specific sections of prompts
- **Custom blocked messages**: Configure user-facing messages for violations

---

## 7. Prompt Management

### 7.1 Overview

Prompt Management enables you to create, edit, version, and deploy reusable prompts. It provides a centralized way to manage prompts across different workflows, models, and applications.

### 7.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Prompt** | An input provided to a model to guide response generation |
| **Variable** | Placeholder in the prompt (e.g., `{{customer_name}}`) for runtime substitution |
| **Prompt Variant** | Alternative configuration — different message, model, or inference settings |
| **Prompt Builder** | Visual console tool for creating, editing, and testing prompts |
| **Version** | Immutable snapshot of a prompt configuration |

### 7.3 Workflow

1. **Create**: Define prompt with variables for flexibility
2. **Configure**: Select model/inference profile, set inference parameters
3. **Test**: Fill in variable values, run prompt, compare variant outputs
4. **Compare**: Test multiple variants side-by-side to find the best one
5. **Version**: Save immutable snapshot when satisfied
6. **Deploy**: Integrate via model inference or as a prompt node in Flows

### 7.4 Integration Options

- **Direct inference**: Specify the prompt when calling `InvokeModel` or `Converse`
- **Flows**: Add a prompt node to a Bedrock Flow
- **Agents**: Use in agent prompt templates

### 7.5 Key Features

- Version comparison with visual diff (green/red highlighting)
- Support for multiple models per prompt variant
- Inference parameter configuration per variant
- API access via `CreatePrompt`, `GetPrompt`, `ListPrompts`, `CreatePromptVersion`

---

## 8. Advanced Prompt Optimization (APO)

### 8.1 Overview

Advanced Prompt Optimization automatically rewrites and improves your prompts to get better results from foundation models. It uses AI-driven techniques to enhance prompt structure, clarity, and effectiveness.

### 8.2 How It Works

1. Submit a prompt for optimization via the `OptimizePrompt` API or console
2. The system analyzes the prompt and generates an optimized version
3. Compare original vs. optimized outputs
4. Use the optimized prompt in your application

### 8.3 Optimization Techniques

- Restructuring prompts for clarity
- Adding specificity and constraints
- Improving instruction formatting
- Optimizing for the target model's strengths
- Reducing ambiguity

### 8.4 Access Methods

- **Console**: Playground or Prompt Management interface
- **API**: `OptimizePrompt` request (returns `analyzePromptEvent` stream)
- **Prompt Management**: Optimize variants directly within the prompt builder

### 8.5 Quotas

- Maximum active APO jobs per account: 20 (adjustable)
- Maximum inactive APO jobs per account: 5,000 (adjustable)

---

## 9. Data Automation

### 9.1 Overview

Amazon Bedrock Data Automation extracts, transforms, and processes information from unstructured documents (PDFs, images, videos, audio) using foundation models. It automates document understanding workflows that traditionally required manual processing.

### 9.2 Key Capabilities

- **Document extraction**: Extract text, tables, key-value pairs from PDFs and images
- **Video understanding**: Analyze video content and extract insights
- **Audio processing**: Transcribe and analyze audio content
- **Custom blueprints**: Define extraction schemas tailored to your document types
- **Projects**: Organize blueprints and configurations for specific use cases

### 9.3 Architecture

| Component | Description |
|-----------|-------------|
| **Blueprints** | Define what information to extract and how to structure it |
| **Projects** | Group blueprints and configurations for a use case |
| **Runtime** | Execute extraction on documents using blueprints/projects |

### 9.4 APIs

| Endpoint | Purpose |
|----------|---------|
| `bedrock-data-automation.{region}.amazonaws.com` | Build-time: Create blueprints and projects |
| `bedrock-data-automation-runtime.{region}.amazonaws.com` | Runtime: Invoke extraction on files |

### 9.5 Supported Regions

US East (N. Virginia), US East (Ohio), US West (Oregon), Asia Pacific (Mumbai, Sydney, Tokyo), Canada (Central), Europe (Frankfurt, Ireland, London, Spain), AWS GovCloud (US-West)

### 9.6 Use Cases

- Invoice and receipt processing
- Contract analysis and extraction
- Medical document understanding
- Insurance claim processing
- Compliance document review
- Media content analysis

---

## 10. Build Section — Feature Comparison Matrix

| Feature | Agents | Flows | Knowledge Bases | Guardrails | Automated Reasoning | Prompt Mgmt | APO | Data Automation |
|---------|--------|-------|-----------------|------------|--------------------:|-------------|-----|-----------------|
| **Primary Purpose** | Autonomous task execution | Workflow orchestration | RAG / Data retrieval | Safety & compliance | Accuracy validation | Prompt lifecycle | Prompt improvement | Document processing |
| **Uses FM** | ✅ | ✅ | ✅ (generation) | ✅ (classification) | ✅ (NL→logic) | ✅ | ✅ | ✅ |
| **Visual Builder** | Console | ✅ (drag-drop) | Console | Console | Console | ✅ | Console | Console |
| **Versioning** | Aliases | ✅ (immutable) | ❌ | ✅ | ✅ (policy versions) | ✅ | ❌ | ❌ |
| **Cross-Region** | ❌ | ❌ | ✅ (via profiles) | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Works with Agents** | — | ✅ (agent node) | ✅ (attached) | ✅ (applied) | ✅ (via guardrail) | ✅ (templates) | ✅ | ❌ |
| **Streaming** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 11. How Build Components Work Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION                               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   BEDROCK AGENT    │ ◄── Prompt Management (templates)
                    │  (Orchestration)   │ ◄── Advanced Prompt Optimization
                    └──┬──────┬──────┬──┘
                       │      │      │
            ┌──────────▼┐  ┌─▼────┐ ┌▼──────────────┐
            │  ACTION    │  │ KB   │ │  GUARDRAILS   │
            │  GROUPS    │  │(RAG) │ │  (Safety)     │
            │ (Lambda/   │  │      │ │  + Automated  │
            │  APIs)     │  │      │ │  Reasoning    │
            └────────────┘  └──────┘ └───────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │            BEDROCK FLOWS                      │
                    │  (End-to-end workflow orchestration)          │
                    │  Nodes: Prompt | KB | Lambda | Agent | Lex   │
                    └──────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │         DATA AUTOMATION                       │
                    │  (Document/Video/Audio processing)            │
                    │  Feeds extracted data into KBs or Flows       │
                    └──────────────────────────────────────────────┘
```

**Typical Integration Pattern:**
1. **Data Automation** processes raw documents → feeds structured data into **Knowledge Bases**
2. **Knowledge Bases** store and index data for retrieval
3. **Agents** orchestrate tasks using **Action Groups** + **Knowledge Bases**
4. **Prompt Management** provides reusable, versioned prompts for agents and flows
5. **APO** optimizes those prompts for better performance
6. **Guardrails** + **Automated Reasoning** validate all inputs/outputs for safety and accuracy
7. **Flows** chain everything together into deployable end-to-end workflows

---

## 12. Conclusion

The Amazon Bedrock Build section provides a complete toolkit for creating production-grade generative AI applications. Agents handle autonomous task execution with multi-step reasoning. Flows enable visual workflow orchestration. Knowledge Bases ground responses in proprietary data via RAG. Guardrails enforce safety and privacy policies. Automated Reasoning provides mathematical verification of response accuracy. Prompt Management and APO streamline prompt engineering. Data Automation handles unstructured document processing. Together, these components form a cohesive platform where each piece integrates seamlessly with the others.

---

## References

1. Amazon Bedrock Agents: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html
2. Amazon Bedrock Flows: https://docs.aws.amazon.com/bedrock/latest/userguide/flows.html
3. Amazon Bedrock Knowledge Bases: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
4. Amazon Bedrock Guardrails: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
5. Automated Reasoning Checks: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html
6. Prompt Management: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
7. Data Automation: https://docs.aws.amazon.com/bedrock/latest/userguide/data-automation.html
8. Multi-Agent Collaboration: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-multi-agent-collaboration.html

---

*Paper generated based on AWS official documentation. Features and availability are subject to change. Visit the AWS Bedrock console for the latest information.*
