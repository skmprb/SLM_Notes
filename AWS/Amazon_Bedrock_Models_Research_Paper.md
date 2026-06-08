# Research Paper: Amazon Bedrock Foundation Models

## Abstract

Amazon Bedrock is a fully managed service by AWS that provides access to hundreds of top foundation models (FMs) from leading AI companies through a single API. This paper provides a comprehensive overview of all model providers available on Amazon Bedrock, with detailed analysis of Anthropic Claude models (Chat LLMs) and Amazon Titan models (Text Generation & Embeddings).

---

## 1. Introduction

Amazon Bedrock enables developers to build and scale generative AI applications without managing infrastructure. It offers the flexibility to swap models in and out without rewriting code, making it ideal for enterprises seeking to innovate rapidly. Models are accessed via the `InvokeModel` and `InvokeModelWithResponseStream` APIs, or the unified `Converse` API.

---

## 2. Complete List of Model Providers on Amazon Bedrock

| # | Provider | Model Categories | Key Models |
|---|----------|-----------------|------------|
| 1 | **AI21 Labs** | Text Generation | Jamba 1.5 Large, Jamba 1.5 Mini |
| 2 | **Amazon** | Text, Embeddings, Image, Video, Speech | Nova (Micro, Lite, Pro, Premier, Canvas, Reel, Sonic), Titan (Text, Embeddings, Image, Multimodal Embeddings) |
| 3 | **Anthropic** | Chat LLM, Reasoning, Coding | Claude 4.x (Opus, Sonnet, Haiku), Claude 3.x, Claude Mythos Preview |
| 4 | **Cohere** | Text, Embeddings, Reranking | Command R, Command R+, Embed English, Embed Multilingual, Embed v4, Rerank 3.5 |
| 5 | **DeepSeek** | Text Generation, Reasoning | DeepSeek-R1, DeepSeek-V3.1, DeepSeek V3.2 |
| 6 | **Google** | Text Generation | Gemma 3 4B IT, Gemma 3 12B IT, Gemma 3 27B PT |
| 7 | **Meta** | Text Generation, Multimodal | Llama 4 (Maverick, Scout), Llama 3.3, Llama 3.2, Llama 3.1, Llama 3 |
| 8 | **MiniMax** | Text Generation | MiniMax M2, MiniMax M2.1, MiniMax M2.5 |
| 9 | **Mistral AI** | Text Generation, Vision, Speech | Mistral Large 3, Mistral Small, Mixtral 8x7B, Pixtral Large, Devstral 2, Voxtral |
| 10 | **Moonshot AI** | Text Generation, Reasoning | Kimi K2.5, Kimi K2 Thinking |
| 11 | **NVIDIA** | Text Generation | Nemotron Nano 9B/12B/30B, Nemotron 3 Super 120B |
| 12 | **OpenAI** | Text Generation, Safety | GPT OSS 20B/120B, GPT OSS Safeguard 20B/120B |
| 13 | **Qwen** | Text Generation, Vision, Coding | Qwen3 32B, Qwen3 235B, Qwen3 VL, Qwen3 Coder series |
| 14 | **Stability AI** | Image Generation & Editing | Stable Image (Upscale, Inpaint, Outpaint, Style Transfer, etc.) |
| 15 | **TwelveLabs** | Video Embeddings & Understanding | Marengo Embed 3.0, Marengo Embed v2.7, Pegasus v1.2 |
| 16 | **Writer** | Text Generation | Palmyra X4, Palmyra X5, Palmyra Vision 7B |
| 17 | **Z.AI** | Text Generation | GLM 4.7, GLM 4.7 Flash, GLM 5 |

---

## 3. Anthropic Claude Models (Chat LLM) — Detailed Analysis

### 3.1 Overview

Anthropic is an AI safety company that develops the Claude family of models. Claude models on Amazon Bedrock are designed for conversational AI, reasoning, coding, agentic tasks, and multimodal understanding. They are accessed via the **Messages API** and support the **Converse API**.

### 3.2 Available Models

#### Claude 4.x Series (Latest Generation)

| Model | Description | Key Strengths |
|-------|-------------|---------------|
| **Claude Opus 4.8** | Optimized for coding, agents, and deeper reasoning in enterprise workflows | Deep reasoning, enterprise-grade |
| **Claude Opus 4.7** | Built for coding, enterprise workflows, and long-running agentic tasks | Long-running agents |
| **Claude Opus 4.6** | Flagship model — plans carefully, sustains agentic tasks longer, operates in massive codebases | Massive codebase navigation |
| **Claude Sonnet 4.6** | Full upgrade of mid-tier model with 1M token context window | Improved coding, computer use, long-context reasoning, agent planning |
| **Claude Opus 4.5** | Coding, agents, and computer use with improvements for spreadsheets and long-running chats | Spreadsheets, long chats |
| **Claude Sonnet 4.5** | Optimized for agents, coding, and computer use | Significant benchmark improvements |
| **Claude Haiku 4.5** | Lightweight model optimized for speed and efficiency | Strong coding and agent performance at low cost |
| **Claude Opus 4.1** | Improved coding, reasoning, and agentic task capabilities | Upgraded reasoning |
| **Claude Sonnet 4** | Balanced model with extended thinking and tool use | Instruction following, coding, reasoning |

#### Claude 3.x Series

| Model | Description | Key Strengths |
|-------|-------------|---------------|
| **Claude 3.5 Haiku** | Next-generation fast model | Improved coding and reasoning over Claude 3 Haiku at same speed |
| **Claude 3 Haiku** | Fastest and most compact Claude 3 model | Near-instant responses, cost-efficient |

#### Preview Models

| Model | Description |
|-------|-------------|
| **Claude Mythos Preview** | Gated research preview — new class of intelligence for cybersecurity, autonomous coding, and long-running agents |

### 3.3 Key Capabilities

- **Conversational AI**: Build virtual assistants, coaching applications, and chatbots
- **Extended Thinking**: Models can reason step-by-step before responding
- **Tool Use / Function Calling**: Invoke external tools, APIs, and functions
- **Computer Use (Beta)**: GUI automation capabilities
- **PDF Document Processing**: Process and cite information from PDF documents
- **Citations**: Provide references to source information in responses
- **Memory Tool (Beta)**: Persistent memory across conversations
- **System Prompts**: Supported on Claude 2.1 and later
- **XML Tag Structuring**: Use XML tags to structure prompts for optimal results
- **Streaming**: Real-time response streaming via InvokeModelWithResponseStream

### 3.4 API & Inference Parameters

Claude models use the **Messages API** format:

```json
{
  "anthropic_version": "bedrock-2023-05-31",
  "max_tokens": 4096,
  "messages": [
    {"role": "user", "content": "Your prompt here"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 250,
  "stop_sequences": ["\\n\\nHuman:"]
}
```

Key parameters:
- **max_tokens**: Maximum tokens to generate in the response
- **temperature** (0.0–1.0): Controls randomness; lower = more deterministic
- **top_p** (0.0–1.0): Nucleus sampling threshold
- **top_k**: Number of top tokens to consider
- **stop_sequences**: Sequences that stop generation
- **system**: System prompt for role/behavior definition

### 3.5 Model Tiers

| Tier | Models | Use Case | Cost |
|------|--------|----------|------|
| **Opus** (Highest) | Opus 4.8, 4.7, 4.6, 4.5, 4.1 | Complex reasoning, long-running agents, enterprise coding | Highest |
| **Sonnet** (Mid) | Sonnet 4.6, 4.5, 4 | Balanced performance/cost, general-purpose | Medium |
| **Haiku** (Fast) | Haiku 4.5, 3.5 Haiku, 3 Haiku | Speed-critical, high-volume, cost-sensitive | Lowest |

---

## 4. Amazon Titan Models — Detailed Analysis

### 4.1 Overview

Amazon Titan is Amazon's own family of foundation models, built for enterprise use cases. The Titan family includes text generation models (Chat LLMs), text embedding models, multimodal embedding models, and image generation models.

### 4.2 Titan Text Models (Chat LLM)

#### Available Models

| Model | Model ID | Max Output Tokens | Use Cases |
|-------|----------|-------------------|-----------|
| **Titan Text Lite** | `amazon.titan-text-lite-v1` | 4,096 | Lightweight tasks, summarization, simple Q&A |
| **Titan Text Express** | `amazon.titan-text-express-v1` | 8,192 | General-purpose text generation, conversations |
| **Titan Text Premier** | `amazon.titan-text-premier-v1:0` | 3,072 | High-quality generation, complex tasks |

#### Inference Parameters

```json
{
  "inputText": "User: What is Amazon Bedrock?\nBot:",
  "textGenerationConfig": {
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokenCount": 4096,
    "stopSequences": []
  }
}
```

| Parameter | Default | Min | Max | Description |
|-----------|---------|-----|-----|-------------|
| **temperature** | 0.7 | 0.0 | 1.0 | Controls randomness |
| **topP** | 0.9 | >0 (e.g., 0.0001) | 1.0 | Nucleus sampling |
| **maxTokenCount** | 512 | 0 | Model-dependent | Max output tokens |
| **stopSequences** | [] | — | — | Stop generation triggers |

#### Response Format

```json
{
  "inputTextTokenCount": 25,
  "results": [{
    "tokenCount": 150,
    "outputText": "\nAmazon Bedrock is...\n",
    "completionReason": "FINISHED"
  }]
}
```

Completion reasons: `FINISHED`, `LENGTH`, `STOP_CRITERIA_MET`, `CONTENT_FILTERED`

#### Conversational Format

For chat-style interactions, use:
```
"inputText": "User: <prompt>\nBot:"
```

### 4.3 Amazon Titan Text Embeddings Models

#### Available Embedding Models

| Model | Model ID | Max Tokens | Max Characters | Output Dimensions | Languages |
|-------|----------|------------|----------------|-------------------|-----------|
| **Titan Text Embeddings V2** | `amazon.titan-embed-text-v2:0` | 8,192 | 50,000 | 1024 (default), 512, 256 | English + 100+ languages |
| **Titan Embeddings G1 - Text** | `amazon.titan-embed-text-v1` | 8,192 | — | 1,536 | English |
| **Titan Multimodal Embeddings G1** | `amazon.titan-embed-image-v1` | — | — | 1024, 384, 256 | Image + Text |
| **Amazon Nova Multimodal Embeddings** | — | — | — | — | Multimodal |

#### Titan Text Embeddings V2 — Key Details

- **Model ID**: `amazon.titan-embed-text-v2:0`
- **Max Input**: 8,192 tokens / 50,000 characters
- **Character-to-Token Ratio**: ~4.7 characters per token (English average)
- **Output Vector Sizes**: 1,024 (default), 512, 256
- **Inference Types**: On-Demand, Provisioned Throughput
- **Throttling**: Requests Per Minute (RPM), not Tokens Per Minute
- **Optimization**: Optimized for text retrieval; also supports semantic similarity, clustering, classification

#### Supported Use Cases

- Retrieval-Augmented Generation (RAG)
- Document search and retrieval
- Semantic similarity
- Text classification
- Clustering
- Reranking

#### Embedding API Request

```json
{
  "inputText": "Your text to embed here",
  "dimensions": 1024,
  "normalize": true,
  "embeddingTypes": ["float"]
}
```

#### Embedding API Response

```json
{
  "embedding": [0.123, -0.456, ...],
  "inputTextTokenCount": 5,
  "embeddingsByType": {
    "float": [0.123, -0.456, ...]
  }
}
```

#### Multilingual Support (V2)

Titan Text Embeddings V2 supports 100+ languages including: English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Arabic, Hindi, Russian, and many more. Cross-language queries return sub-optimal results — it's recommended to keep knowledge base and queries in the same language.

#### Best Practices for Embeddings

1. **Segment documents** into logical paragraphs or sections for retrieval tasks
2. **Use appropriate dimensions** — 256 for speed, 1024 for accuracy
3. **Normalize vectors** for cosine similarity comparisons
4. **Plan capacity** using RPM quotas, not token-based quotas

### 4.4 Titan Multimodal Embeddings G1

- Supports both **image and text** inputs
- Configurable vector sizes: 1024, 384, 256
- Use cases: Image search, recommendation systems, personalization
- Supports fine-tuning with custom datasets

---

## 5. Comparison: Anthropic Claude vs Amazon Titan (Chat LLM)

| Feature | Anthropic Claude | Amazon Titan Text |
|---------|-----------------|-------------------|
| **API Format** | Messages API (conversational) | inputText with textGenerationConfig |
| **Context Window** | Up to 1M tokens (Sonnet 4.6) | Model-dependent (up to 8K output) |
| **Multimodal** | Vision, PDF, Documents | Text only |
| **Tool Use** | Yes (function calling) | No |
| **Extended Thinking** | Yes | No |
| **System Prompts** | Yes (v2.1+) | No dedicated field |
| **Streaming** | Yes | Yes |
| **Fine-tuning** | Limited | Supported |
| **Best For** | Complex reasoning, coding, agents | Cost-effective text generation, enterprise |

---

## 6. Embedding Models Comparison

| Feature | Titan Text Embeddings V2 | Titan Embeddings G1 | Titan Multimodal Embeddings | Cohere Embed | Nova Multimodal Embeddings |
|---------|--------------------------|--------------------|-----------------------------|--------------|---------------------------|
| **Dimensions** | 256/512/1024 | 1,536 | 256/384/1024 | 1024 | — |
| **Max Tokens** | 8,192 | 8,192 | — | 512 | — |
| **Modality** | Text | Text | Image + Text | Text | Multimodal |
| **Languages** | 100+ | English | — | 100+ | — |
| **Fine-tuning** | No | No | Yes | No | — |

---

## 7. Code Example: Invoking Anthropic Claude on Bedrock (Python)

```python
import boto3
import json

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7
})

response = bedrock.invoke_model(
    body=body,
    modelId="anthropic.claude-sonnet-4-20250514-v1:0",
    accept="application/json",
    contentType="application/json"
)

result = json.loads(response.get("body").read())
print(result["content"][0]["text"])
```

---

## 8. Code Example: Amazon Titan Text Embeddings (Python)

```python
import boto3
import json

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

body = json.dumps({
    "inputText": "Amazon Bedrock is a fully managed service for foundation models.",
    "dimensions": 1024,
    "normalize": True
})

response = bedrock.invoke_model(
    body=body,
    modelId="amazon.titan-embed-text-v2:0",
    accept="application/json",
    contentType="application/json"
)

result = json.loads(response.get("body").read())
embedding = result["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

---

## 9. Model Limits & Token Quotas

### 9.1 Context Window Limits (Input + Output)

#### Anthropic Claude Models

| Model | Context Window (Input) | Max Output Tokens | Notes |
|-------|----------------------|-------------------|-------|
| **Claude Sonnet 4.6** | 1,000,000 tokens | 64,000 | Largest context window |
| **Claude Opus 4.8/4.7/4.6/4.5** | 200,000 tokens | 32,000 | Enterprise-grade |
| **Claude Sonnet 4.5/4** | 200,000 tokens | 64,000 | Balanced tier |
| **Claude Haiku 4.5** | 200,000 tokens | 16,000 | Fast tier |
| **Claude 3.5 Haiku** | 200,000 tokens | 8,192 | Legacy fast tier |
| **Claude 3 Haiku** | 200,000 tokens | 4,096 | Compact model |

#### Amazon Titan Text Models

| Model | Context Window (Input) | Max Output Tokens |
|-------|----------------------|-------------------|
| **Titan Text Lite** | 4,096 tokens | 4,096 |
| **Titan Text Express** | 8,192 tokens | 8,192 |
| **Titan Text Premier** | 32,000 tokens | 3,072 |

#### Amazon Titan Embedding Models

| Model | Max Input Tokens | Max Input Characters | Output Dimensions |
|-------|-----------------|---------------------|-------------------|
| **Titan Text Embeddings V2** | 8,192 | 50,000 | 256 / 512 / 1,024 |
| **Titan Embeddings G1 - Text** | 8,192 | — | 1,536 |
| **Titan Multimodal Embeddings G1** | — | — | 256 / 384 / 1,024 |

### 9.2 Tokens Per Minute (TPM) Quotas

Amazon Bedrock enforces per-model, per-Region token-based quotas on the `bedrock-runtime` endpoint:

| Quota Type | Scope | Description |
|-----------|-------|-------------|
| **On-demand InvokeModel TPM** | Per model, per Region | Max tokens/minute (input + output combined) for on-demand single-Region invocation |
| **Cross-Region InvokeModel TPM** | Per model, per Region | Max tokens/minute when invoked through a cross-Region inference profile |
| **Model invocation max tokens per day (TPD)** | Per model, per Region | Max tokens/day. Default = TPM × 24 × 60. New accounts may have reduced quotas |

> **Important**: Amazon Bedrock **no longer enforces Requests Per Minute (RPM)** quotas on the `bedrock-runtime` endpoint. Throttling is governed entirely by token-based quotas.

> **Exception**: Embedding models are still throttled by **Requests Per Minute (RPM)**, not Tokens Per Minute (TPM).

### 9.3 Token Burndown Rate

Output tokens consume quota at different rates depending on the model:

| Model Family | Burndown Rate | Example |
|-------------|---------------|----------|
| **Anthropic Claude 3.7 and later** | **5x** for output tokens | 1 output token = 5 tokens from quota |
| **All other models** | **1:1** | 1 output token = 1 token from quota |

**Quota Deduction Formula:**

- **At request start**: `Input tokens + CacheWriteInputTokens + max_tokens` (deducted upfront)
- **At request end (final)**: `InputTokenCount + CacheWriteInputTokens + (OutputTokenCount × burndown rate)`
- **CacheReadInputTokens** do NOT count toward quota
- You are only **billed** for actual token usage, not the quota reservation

**Example** (Claude Sonnet 4):
- Input: 1,000 tokens, Output: 100 tokens
- Quota consumed: 1,000 + (100 × 5) = **1,500 tokens**
- Billed for: **1,100 tokens** (actual usage)

### 9.4 Optimizing max_tokens for Throughput

- The `max_tokens` value is deducted from your quota at the **beginning** of each request
- Setting `max_tokens` too high reduces concurrency (quota fills up faster)
- **Best practice**: Set `max_tokens` close to your expected output size
- Use CloudWatch `OutputTokenCount` metrics to determine optimal values

### 9.5 Default Quota Values (Representative)

| Model | On-Demand TPM (Default) | Cross-Region TPM (Default) | Adjustable |
|-------|------------------------|---------------------------|------------|
| Claude Sonnet 4 | ~100,000–1,000,000 | Higher (varies) | Yes |
| Claude Haiku 4.5 | ~100,000–1,000,000 | Higher (varies) | Yes |
| Titan Text Express | ~100,000–500,000 | Higher (varies) | Yes |
| Titan Embeddings V2 | RPM-based (~2,000 RPM) | — | Yes |

> **Note**: Exact default values vary by Region and account age. Check the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas) for your account's current limits.

### 9.6 Requesting Quota Increases

1. Open the **Service Quotas console** → Select **Amazon Bedrock**
2. Search for the specific model quota
3. Request increase for **Cross-Region InvokeModel TPM** (AWS will offer to increase On-Demand TPM and TPD together)
4. Priority is given to customers already consuming their existing quota allocation
5. Quota increases are **not granted** for models in Legacy or Deprecated lifecycle status

---

## 10. Inference Profiles — On-Demand, Provisioned Throughput & Cross-Region

### 10.1 Overview

Inference profiles are resources in Amazon Bedrock that define a model and one or more Regions to which inference requests can be routed. They enable cost tracking, usage metrics, and cross-Region load distribution.

### 10.2 Inference Types Comparison

| Feature | On-Demand | Provisioned Throughput | Cross-Region Inference |
|---------|-----------|----------------------|------------------------|
| **Pricing** | Pay-per-token | Fixed hourly rate per Model Unit (MU) | Pay-per-token (source Region pricing) |
| **Capacity** | Shared, subject to TPM quotas | Dedicated, guaranteed throughput | Distributed across Regions |
| **Commitment** | None | No commitment, 1-month, or 6-month | None |
| **Best For** | Variable/unpredictable workloads | Consistent high-volume workloads | Burst traffic, high availability |
| **Custom Models** | Not required | Required for custom models | Not supported |
| **Quota Type** | TPM-based | Model Unit-based | TPM-based (per profile) |

### 10.3 On-Demand Inference

- **Default mode** — no setup required beyond model access
- Pay only for tokens consumed (input + output)
- Subject to per-model TPM and TPD quotas
- Suitable for development, testing, and variable production workloads
- Accessed via `InvokeModel`, `InvokeModelWithResponseStream`, `Converse`, `ConverseStream` APIs

### 10.4 Provisioned Throughput

Provisioned Throughput provides dedicated capacity at a fixed hourly cost:

**Key Concepts:**
- **Model Unit (MU)**: A unit of dedicated throughput capacity
  - Specifies max input tokens processable per minute across all requests
  - Specifies max output tokens generatable per minute across all requests
- **Commitment Terms**:
  - No commitment — delete anytime (highest hourly rate)
  - 1-month commitment — locked for 1 month (discounted)
  - 6-month commitment — locked for 6 months (most discounted)
- **Billing**: Continuous hourly billing until Provisioned Throughput is deleted

**When to Use Provisioned Throughput:**
- Consistent, predictable high-volume inference workloads
- Custom/fine-tuned models (required)
- Need guaranteed throughput without throttling
- Cost optimization for sustained high usage

**Limitations:**
- Inference profiles do NOT support Provisioned Throughput
- Must be purchased per model, per Region
- Contact AWS account manager for MU specifications and pricing

### 10.5 Cross-Region Inference

Cross-Region inference distributes requests across multiple AWS Regions to increase throughput and handle traffic bursts:

#### Types of Cross-Region Inference

| Feature | Geographic Cross-Region | Global Cross-Region |
|---------|------------------------|--------------------|
| **Data Residency** | Within geographic boundaries (US, EU, APAC) | Any supported AWS commercial Region worldwide |
| **Throughput** | Higher than single-region | Highest available |
| **Cost** | Standard pricing | ~10% savings |
| **SCP Requirements** | Allow all destination Regions in profile | Allow `"aws:RequestedRegion": "unspecified"` |
| **Best For** | Organizations with data residency regulations | Maximum performance and cost optimization |

#### Key Characteristics:
- **No additional routing cost** — priced based on source Region
- **Data stays on AWS network** — never traverses public internet
- **Encrypted in transit** between Regions
- **CloudTrail logging** in source Region (check `additionalEventData.inferenceRegion`)
- Does NOT require manual Region enablement
- Does NOT support Provisioned Throughput

### 10.6 Inference Profile Types

| Profile Type | Created By | Purpose |
|-------------|-----------|----------|
| **Cross-Region (System-Defined)** | AWS (predefined) | Route requests across multiple Regions for a model |
| **Application Inference Profile** | User-created | Track costs/usage for a model in one or multiple Regions |

**Application Inference Profiles support:**
- Model inference (InvokeModel, Converse APIs)
- Knowledge base vector embedding and response generation
- Model evaluation jobs
- Prompt management
- Flows (prompt nodes)

### 10.7 Supported Features by Inference Type

| Feature | On-Demand | Provisioned | Cross-Region |
|---------|-----------|-------------|---------------|
| InvokeModel API | ✅ | ✅ | ✅ |
| Converse API | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ |
| Batch Inference | ✅ | ❌ | ❌ |
| Knowledge Bases | ✅ | ✅ | ✅ |
| Agents | ✅ | ✅ | ✅ |
| Fine-tuned Models | ❌ | ✅ (required) | ❌ |
| Cost Tagging | Via inference profiles | Direct | Via inference profiles |
| CloudWatch Metrics | ✅ | ✅ | ✅ |

---

## 11. Service Endpoints & Regional Availability

### 11.1 Primary Endpoints

| Endpoint | Purpose |
|----------|----------|
| `bedrock.{region}.amazonaws.com` | Control plane (model management, training, deployment) |
| `bedrock-runtime.{region}.amazonaws.com` | Inference (InvokeModel, Converse) |
| `bedrock-agent.{region}.amazonaws.com` | Agents build-time (create/manage agents & KBs) |
| `bedrock-agent-runtime.{region}.amazonaws.com` | Agents runtime (invoke agents, query KBs) |

### 11.2 Key Regions for Bedrock Runtime

| Region | Endpoint | FIPS Available |
|--------|----------|----------------|
| US East (N. Virginia) | bedrock-runtime.us-east-1.amazonaws.com | ✅ |
| US East (Ohio) | bedrock-runtime.us-east-2.amazonaws.com | ✅ |
| US West (Oregon) | bedrock-runtime.us-west-2.amazonaws.com | ✅ |
| Europe (Frankfurt) | bedrock-runtime.eu-central-1.amazonaws.com | ❌ |
| Europe (Ireland) | bedrock-runtime.eu-west-1.amazonaws.com | ❌ |
| Asia Pacific (Tokyo) | bedrock-runtime.ap-northeast-1.amazonaws.com | ❌ |
| Asia Pacific (Sydney) | bedrock-runtime.ap-southeast-2.amazonaws.com | ❌ |
| Asia Pacific (Mumbai) | bedrock-runtime.ap-south-1.amazonaws.com | ❌ |
| Canada (Central) | bedrock-runtime.ca-central-1.amazonaws.com | ✅ |

---

## 12. Conclusion

Amazon Bedrock provides a unified platform to access foundation models from 17 leading AI providers. Anthropic Claude models represent the state-of-the-art in conversational AI, reasoning, and agentic tasks with models ranging from the fast Haiku tier to the powerful Opus tier. Amazon Titan models offer cost-effective, enterprise-ready text generation and industry-leading embedding capabilities supporting 100+ languages. Together, they cover the full spectrum of generative AI needs — from simple text generation to complex multi-step reasoning, and from semantic search to multimodal understanding.

---

## References

1. AWS Documentation - Amazon Bedrock User Guide: https://docs.aws.amazon.com/bedrock/latest/userguide/
2. Amazon Bedrock Models at a Glance: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
3. Anthropic Claude Models: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
4. Amazon Titan Text Models: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
5. Amazon Titan Text Embeddings: https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
6. Anthropic Documentation: https://docs.anthropic.com/en/docs/welcome
7. Bedrock Runtime Quotas: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-runtime.html
8. How Tokens Are Counted: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html
9. Inference Profiles: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html
10. Cross-Region Inference: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
11. Provisioned Throughput: https://docs.aws.amazon.com/bedrock/latest/userguide/prov-throughput.html
12. Amazon Bedrock Endpoints & Quotas: https://docs.aws.amazon.com/general/latest/gr/bedrock.html

---

*Paper generated based on AWS official documentation. Model availability and specifications are subject to change. Visit the AWS Bedrock console for the latest information.*
