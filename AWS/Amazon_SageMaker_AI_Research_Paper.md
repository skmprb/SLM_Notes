# Research Paper: Amazon SageMaker AI — A Story-Driven Guide to ML at Scale

## Abstract

Amazon SageMaker AI is a fully managed machine learning service that enables data scientists and developers to build, train, and deploy ML models at any scale. This paper takes a story-driven approach — following a user's journey from "I need to train a model" through every decision point, explaining which SageMaker feature solves each problem, comparing compute options, and providing configuration examples.

---

## 1. What is Amazon SageMaker AI?

Amazon SageMaker AI (renamed from Amazon SageMaker in December 2024) is a fully managed ML service that covers the entire machine learning lifecycle:

**Prepare** → **Build** → **Train** → **Tune** → **Deploy** → **Monitor**

It eliminates the undifferentiated heavy lifting of ML infrastructure — you focus on your model, SageMaker handles the rest.

### 1.1 Why Use SageMaker?

| Challenge | Without SageMaker | With SageMaker |
|-----------|-------------------|----------------|
| Infrastructure | Manually provision EC2, GPUs, storage | Fully managed, auto-provisioned |
| Scaling | Build custom distributed training | Built-in distributed training libraries |
| Experimentation | Track experiments manually | SageMaker Experiments auto-tracks |
| Deployment | Build API servers, load balancers | One-click endpoint deployment |
| Monitoring | Custom monitoring pipelines | Model Monitor detects drift automatically |
| Cost | Pay for idle GPUs | Managed Spot (up to 90% savings), Warm Pools |
| Collaboration | Shared drives, Git | Studio with shared spaces, notebooks |

---

## 2. The SageMaker Story: "I Need to Train a Model"

### 📖 Chapter 1: "I have data but no ML experience"

> *"I'm a business analyst. I have a CSV of customer data and want to predict churn. I don't know Python or ML."*

**→ Use: SageMaker Canvas**

Canvas is a no-code/low-code AutoML tool. You upload data, Canvas automatically selects the best algorithm, trains the model, and gives you predictions.

```
You bring: CSV data
Canvas does: Data analysis → Algorithm selection → Training → Evaluation → Predictions
You get: A working model with accuracy metrics
```

**Configuration**: None needed — just upload data in the Canvas UI.

---

### 📖 Chapter 2: "I know Python and want to train a standard ML model"

> *"I'm a data scientist. I want to train an XGBoost model on my tabular data with hyperparameter tuning."*

**→ Use: SageMaker Built-in Algorithms + Python SDK**

```python
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Use built-in XGBoost algorithm
xgb = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", session.boto_region_name, "1.7-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",       # CPU instance for tabular data
    output_path=f"s3://{session.default_bucket()}/output",
    sagemaker_session=session
)

# Set hyperparameters
xgb.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2
)

# Train
xgb.fit({"train": "s3://my-bucket/train.csv", "validation": "s3://my-bucket/val.csv"})
```

**Compute Choice**: `ml.m5.xlarge` (CPU) — tabular data doesn't need GPUs.

---

### 📖 Chapter 3: "I want to fine-tune a foundation model (LLM)"

> *"I want to fine-tune Llama 3 on my company's domain-specific data for a chatbot."*

**→ Use: SageMaker JumpStart + GPU Instances**

```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id="meta-textgeneration-llama-3-8b",
    instance_type="ml.g5.12xlarge",     # 4x NVIDIA A10G GPUs
    instance_count=1,
    environment={"accept_eula": "true"}
)

# Fine-tune on your data
estimator.fit({"training": "s3://my-bucket/fine-tune-data/"})

# Deploy
predictor = estimator.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1
)
```

**Compute Choice**: `ml.g5.12xlarge` for training (4 GPUs), `ml.g5.2xlarge` for inference (1 GPU).

---

### 📖 Chapter 4: "I need to train a massive model across multiple GPUs"

> *"I'm pre-training a 70B parameter model. I need distributed training across many GPUs."*

**→ Use: SageMaker HyperPod + Distributed Training**

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=8,                    # 8 nodes
    instance_type="ml.p4d.24xlarge",     # 8x NVIDIA A100 GPUs per node = 64 GPUs total
    framework_version="2.0",
    py_version="py310",
    distribution={
        "torch_distributed": {
            "enabled": True
        }
    },
    hyperparameters={
        "model_name": "llama-70b",
        "epochs": 3,
        "batch_size": 32
    }
)

estimator.fit({"training": "s3://my-bucket/pretrain-data/"})
```

**Compute Choice**: `ml.p4d.24xlarge` × 8 = 64 A100 GPUs with 400 Gbps EFA networking.

For even larger scale, use **SageMaker HyperPod** — persistent clusters with thousands of accelerators, automatic fault recovery, and Slurm workload management.

---

### 📖 Chapter 5: "I want the cheapest training possible"

> *"My training job takes 4 hours. I want to minimize cost."*

**→ Use: Managed Spot Training (up to 90% savings)**

```python
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    use_spot_instances=True,             # Enable Spot!
    max_wait=7200,                       # Max wait time (seconds)
    max_run=14400,                       # Max run time (seconds)
    checkpoint_s3_uri="s3://my-bucket/checkpoints/"  # Save progress
)
```

**Cost Savings**: Up to 90% off On-Demand pricing. Checkpointing ensures no work is lost if interrupted.

---

### 📖 Chapter 6: "My model is trained. How do I deploy it?"

> *"I have a trained model. I need to serve predictions to my application."*

**→ Choose your deployment option:**

| Option | Best For | Latency | Cost Model | Payload |
|--------|----------|---------|------------|---------|
| **Real-time Endpoint** | Interactive apps, low latency | Milliseconds | Pay per instance-hour | Up to 6 MB |
| **Serverless Endpoint** | Sporadic traffic, no infra management | Cold start possible | Pay per inference | Up to 6 MB |
| **Asynchronous Endpoint** | Large payloads, long processing | Minutes | Pay per instance-hour | Up to 1 GB |
| **Batch Transform** | Bulk predictions, no endpoint needed | Hours | Pay per job | Unlimited |

**Real-time Endpoint Example:**

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://my-bucket/output/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py"
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",         # GPU for inference
    initial_instance_count=1,
    endpoint_name="my-chatbot-endpoint"
)

# Get predictions
result = predictor.predict({"text": "What is your return policy?"})
```

**Serverless Endpoint (zero infrastructure):**

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10
)

predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="my-serverless-endpoint"
)
```

---

### 📖 Chapter 7: "My model is in production. How do I monitor it?"

> *"My model has been running for 3 months. How do I know if it's still accurate?"*

**→ Use: SageMaker Model Monitor**

```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20
)

# Create baseline from training data
monitor.suggest_baseline(
    baseline_dataset="s3://my-bucket/train.csv",
    dataset_format=DatasetFormat.csv(header=True)
)

# Schedule monitoring (runs hourly)
monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-model-monitor",
    endpoint_input=predictor.endpoint_name,
    schedule_cron_expression="cron(0 * ? * * *)"
)
```

Detects: Data drift, model quality degradation, bias drift, feature attribution drift.

---

## 3. Compute Instance Comparison

### 3.1 Training Instances

| Instance Family | GPU | GPU Memory | vCPUs | RAM | Network | Best For | Cost (approx/hr) |
|----------------|-----|-----------|-------|-----|---------|----------|-------------------|
| **ml.m5.xlarge** | None | — | 4 | 16 GB | Up to 10 Gbps | Tabular data, XGBoost, sklearn | $0.23 |
| **ml.c5.4xlarge** | None | — | 16 | 32 GB | Up to 10 Gbps | CPU-intensive preprocessing | $0.68 |
| **ml.g4dn.xlarge** | 1× T4 | 16 GB | 4 | 16 GB | Up to 25 Gbps | Small model training, inference | $0.53 |
| **ml.g5.xlarge** | 1× A10G | 24 GB | 4 | 16 GB | Up to 25 Gbps | Medium model fine-tuning | $1.01 |
| **ml.g5.12xlarge** | 4× A10G | 96 GB | 48 | 192 GB | 40 Gbps | LLM fine-tuning (7B–13B) | $5.67 |
| **ml.g5.48xlarge** | 8× A10G | 192 GB | 192 | 768 GB | 100 Gbps | Large model training | $16.29 |
| **ml.p4d.24xlarge** | 8× A100 | 320 GB | 96 | 1.1 TB | 400 Gbps EFA | Foundation model pre-training | $32.77 |
| **ml.p5.48xlarge** | 8× H100 | 640 GB | 192 | 2 TB | 3200 Gbps EFA | Largest models, fastest training | ~$98 |
| **ml.trn1.32xlarge** | 16× Trainium | 512 GB | 128 | 512 GB | 800 Gbps EFA | Cost-effective large model training | $21.50 |
| **ml.trn2.48xlarge** | 16× Trainium2 | 1.5 TB | 192 | 1.5 TB | 3200 Gbps EFA | Next-gen efficient training | ~$45 |

### 3.2 Decision Matrix: Which Instance to Choose?

| Your Task | Model Size | Recommended Instance | Why |
|-----------|-----------|---------------------|-----|
| Tabular ML (XGBoost, Random Forest) | Small | `ml.m5.xlarge` | CPU is sufficient, cheapest |
| Image classification (ResNet) | Medium | `ml.g4dn.xlarge` | 1 GPU enough for CNN training |
| Fine-tune 7B LLM | 7B params | `ml.g5.12xlarge` | 4× A10G = 96 GB GPU memory |
| Fine-tune 13B LLM | 13B params | `ml.g5.48xlarge` | 8× A10G = 192 GB GPU memory |
| Fine-tune 70B LLM | 70B params | `ml.p4d.24xlarge` | 8× A100 = 320 GB + model parallelism |
| Pre-train foundation model | 100B+ params | `ml.p5.48xlarge` × N | H100 cluster with EFA |
| Cost-optimized large training | 70B+ params | `ml.trn1.32xlarge` | AWS Trainium — 50% cheaper than GPUs |

### 3.3 Inference Instances

| Instance | GPU | Best For | Latency | Cost |
|----------|-----|----------|---------|------|
| `ml.t3.medium` | None | Tiny models, testing | High | $0.05/hr |
| `ml.c5.xlarge` | None | CPU inference (tabular) | Low | $0.17/hr |
| `ml.g4dn.xlarge` | 1× T4 | Small model inference | Low | $0.53/hr |
| `ml.g5.xlarge` | 1× A10G | Medium model inference | Very Low | $1.01/hr |
| `ml.g5.2xlarge` | 1× A10G | LLM inference (7B) | Very Low | $1.21/hr |
| `ml.p4d.24xlarge` | 8× A100 | Large LLM inference (70B) | Low | $32.77/hr |
| `ml.inf2.xlarge` | AWS Inferentia2 | Cost-optimized inference | Low | $0.76/hr |

---

## 4. Complete Feature Map: "If You Need X, Use Y"

| If You Need... | Use This SageMaker Feature | Why |
|----------------|---------------------------|-----|
| No-code model building | **Canvas** | Drag-and-drop, auto-selects algorithm |
| Pre-trained foundation models | **JumpStart** | 400+ models, 1-click deploy |
| Custom training scripts | **Training Jobs (Script Mode)** | Bring your PyTorch/TF code |
| Distributed GPU training | **Distributed Training Libraries** | Data parallel + model parallel |
| Persistent GPU clusters | **HyperPod** | Always-on, fault-tolerant, Slurm |
| Hyperparameter optimization | **Automatic Model Tuning** | Bayesian optimization |
| Data labeling | **Ground Truth** | Human + ML labeling |
| Data preparation | **Data Wrangler** | 300+ built-in transforms |
| Feature engineering | **Feature Store** | Online + Offline store |
| Bias detection | **Clarify** | Pre-training + post-training bias |
| Experiment tracking | **Experiments** | Auto-logs metrics, params, artifacts |
| ML pipelines (CI/CD) | **Pipelines** | DAG-based workflow orchestration |
| Model versioning | **Model Registry** | Approval workflows, lineage |
| Real-time predictions | **Real-time Endpoints** | Low-latency, auto-scaling |
| Serverless predictions | **Serverless Endpoints** | Zero infra, pay-per-use |
| Large payload inference | **Async Endpoints** | Up to 1 GB, queued processing |
| Bulk predictions | **Batch Transform** | No endpoint needed |
| Production monitoring | **Model Monitor** | Data drift, quality, bias |
| Edge deployment | **Neo + Edge Manager** | Compile for edge hardware |
| Cost savings (training) | **Managed Spot Instances** | Up to 90% off |
| Faster iterative training | **Warm Pools** | Keep instances warm between jobs |
| Mixed instance training | **Heterogeneous Clusters** | CPU + GPU in same job |
| Compute reservation | **Training Plans** | Guaranteed GPU access |
| IDE environment | **Studio** | JupyterLab, Code Editor, RStudio |

---

## 5. End-to-End Example: Customer Churn Prediction

### The Story

> A telecom company wants to predict which customers will churn. They have 100K rows of customer data (demographics, usage, billing). They want a model in production with monitoring.

### Step 1: Prepare Data (Data Wrangler)

```python
# In SageMaker Studio → Data Wrangler
# Import from S3, apply transforms:
# - Handle missing values
# - Encode categorical features
# - Normalize numerical features
# - Export to S3 as train/test split
```

### Step 2: Train Model (Built-in XGBoost)

```python
from sagemaker.estimator import Estimator

xgb = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", "us-east-1", "1.7-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://telecom-ml/output/",
    use_spot_instances=True,
    max_wait=3600,
    max_run=3600
)

xgb.set_hyperparameters(
    objective="binary:logistic",
    num_round=200,
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb.fit({
    "train": "s3://telecom-ml/data/train.csv",
    "validation": "s3://telecom-ml/data/test.csv"
})
```

### Step 3: Tune Hyperparameters

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name="validation:auc",
    hyperparameter_ranges={
        "max_depth": IntegerParameter(3, 10),
        "eta": ContinuousParameter(0.01, 0.3),
        "num_round": IntegerParameter(100, 500)
    },
    max_jobs=20,
    max_parallel_jobs=4
)

tuner.fit({"train": "s3://telecom-ml/data/train.csv", "validation": "s3://telecom-ml/data/test.csv"})
```

### Step 4: Deploy Best Model

```python
best_estimator = tuner.best_estimator()
predictor = best_estimator.deploy(
    instance_type="ml.c5.xlarge",
    initial_instance_count=1,
    endpoint_name="churn-predictor"
)
```

### Step 5: Monitor in Production

```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(role=role, instance_type="ml.m5.xlarge")
monitor.suggest_baseline(baseline_dataset="s3://telecom-ml/data/train.csv")
monitor.create_monitoring_schedule(
    endpoint_input="churn-predictor",
    schedule_cron_expression="cron(0 */6 ? * * *)"  # Every 6 hours
)
```

**Total Cost**: ~$5–15 for training (Spot), ~$0.17/hr for inference endpoint.

---

## 6. SageMaker vs Alternatives

| Feature | SageMaker | Google Vertex AI | Azure ML | Self-Managed (EC2) |
|---------|-----------|-----------------|----------|-------------------|
| Managed Training | ✅ | ✅ | ✅ | ❌ (manual) |
| Built-in Algorithms | 17+ | Limited | Limited | ❌ |
| Foundation Model Hub | JumpStart (400+) | Model Garden | Model Catalog | ❌ |
| No-Code ML | Canvas | AutoML Tables | Designer | ❌ |
| Distributed Training | Native libraries | Vertex Training | Distributed | Manual setup |
| Spot Training | ✅ (90% savings) | Preemptible VMs | Low-priority VMs | Manual |
| HyperPod (persistent clusters) | ✅ | ❌ | ❌ | Manual |
| Serverless Inference | ✅ | ✅ | ✅ | ❌ |
| Model Monitoring | Model Monitor | Model Monitoring | ❌ (basic) | ❌ |
| Feature Store | ✅ | ✅ | ❌ | ❌ |
| ML Pipelines | Pipelines | Vertex Pipelines | ML Pipelines | Airflow/custom |
| AWS Trainium Support | ✅ | ❌ | ❌ | ❌ |
| Edge Deployment | Neo | Edge | ❌ | Manual |

---

## 7. Cost Optimization Strategies

| Strategy | Savings | How |
|----------|---------|-----|
| **Managed Spot Training** | Up to 90% | Use interruptible instances with checkpointing |
| **Warm Pools** | Reduce startup time | Keep instances warm between iterative jobs |
| **Heterogeneous Clusters** | 30–40% | Mix CPU + GPU instances in same job |
| **Training Plans** | Predictable cost | Reserve capacity for large jobs |
| **Serverless Inference** | Pay-per-use | No idle instance costs |
| **Auto-scaling** | Match demand | Scale endpoints based on traffic |
| **AWS Trainium** | 50% vs GPUs | Use Trainium chips for large model training |
| **AWS Inferentia** | 70% vs GPUs | Use Inferentia chips for inference |
| **SageMaker Neo** | 2× throughput | Compile models for target hardware |

---

## 8. What Are Inference Instances? (Explained Simply)

### 8.1 Training vs Inference — The Two Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TRAINING                    PHASE 2: INFERENCE            │
│  (Teaching the model)                 (Using the model)             │
│                                                                     │
│  Input: Data + Algorithm              Input: New data (user query)  │
│  Output: Trained model file           Output: Prediction/Answer     │
│  Duration: Hours/Days                 Duration: Milliseconds        │
│  Frequency: Once (or periodically)    Frequency: Every user request │
│  Compute: Heavy (many GPUs)           Compute: Lighter (fewer GPUs) │
└─────────────────────────────────────────────────────────────────────┘
```

**Training** = Teaching a student (expensive, takes time, done once)
**Inference** = The student answering questions (cheap per question, happens constantly)

### 8.2 What Are Inference Instances?

Inference instances are the **servers that host your trained model and serve predictions** to users in real-time. When someone asks your chatbot a question, the inference instance:

1. Receives the request
2. Loads the input into the model
3. Runs the model forward pass (no backpropagation — that's training)
4. Returns the prediction/response

**Why different instances for inference vs training?**

| Aspect | Training Instance | Inference Instance |
|--------|-------------------|--------------------|
| **Purpose** | Learn patterns from data | Serve predictions to users |
| **GPU needs** | Maximum (parallel computation) | Moderate (single forward pass) |
| **Memory needs** | Very high (model + gradients + optimizer states) | Lower (model weights only) |
| **Duration** | Hours/days | Always running (or serverless) |
| **Cost priority** | Speed (finish faster = less cost) | Latency + cost per request |
| **Scaling** | Fixed (1 job) | Auto-scales with traffic |

**Example**: Training a 7B model needs `ml.g5.12xlarge` (4 GPUs, 96 GB) because it stores model weights + gradients + optimizer states (3× the model size). But serving that same model for inference only needs `ml.g5.2xlarge` (1 GPU, 24 GB) because it only holds the model weights.

### 8.3 Inference Instance Types Explained

| Instance | What's Inside | When to Use | Real-World Analogy |
|----------|--------------|-------------|--------------------|
| `ml.t3.medium` | CPU only, 2 vCPU, 4 GB | Testing, tiny models | A bicycle — slow but free |
| `ml.c5.xlarge` | CPU only, 4 vCPU, 8 GB | Tabular model predictions (XGBoost) | A car — good for daily commute |
| `ml.g4dn.xlarge` | 1× NVIDIA T4 (16 GB) | Small neural networks, image classification | A sports car — fast enough for most |
| `ml.g5.xlarge` | 1× NVIDIA A10G (24 GB) | Medium models, small LLMs (1-3B) | A race car — serious performance |
| `ml.g5.2xlarge` | 1× NVIDIA A10G (24 GB) + more CPU/RAM | LLM inference (7B models) | A race car with bigger fuel tank |
| `ml.inf2.xlarge` | AWS Inferentia2 chip | Cost-optimized LLM inference | An electric car — cheaper to run |
| `ml.p4d.24xlarge` | 8× NVIDIA A100 (320 GB) | Huge LLM inference (70B+) | A jet — for when nothing else works |

---

## 9. Cost Optimization — "50% vs GPU" Explained

### 9.1 What Does "AWS Trainium is 50% Cheaper Than GPUs" Mean?

AWS builds its own custom AI chips:

| Chip | Purpose | vs NVIDIA GPUs |
|------|---------|----------------|
| **AWS Trainium** | Training models | ~50% cheaper than equivalent NVIDIA GPU instances |
| **AWS Trainium2** | Training (next-gen) | Even more cost-effective |
| **AWS Inferentia** | Serving predictions (inference) | ~70% cheaper than equivalent NVIDIA GPU instances |

**Concrete Example:**

| Task | NVIDIA Option | Cost/hr | AWS Chip Option | Cost/hr | Savings |
|------|--------------|---------|-----------------|---------|----------|
| Train 70B model | `ml.p4d.24xlarge` (8× A100) | ~$32.77 | `ml.trn1.32xlarge` (16× Trainium) | ~$21.50 | **34% cheaper** |
| Serve 7B model | `ml.g5.2xlarge` (1× A10G) | ~$1.21 | `ml.inf2.xlarge` (Inferentia2) | ~$0.76 | **37% cheaper** |
| Serve at scale | `ml.g5.12xlarge` (4× A10G) | ~$5.67 | `ml.inf2.24xlarge` (6× Inferentia2) | ~$2.50 | **56% cheaper** |

**Why the savings?**
- NVIDIA GPUs are general-purpose (gaming, graphics, AI, scientific computing)
- AWS Trainium/Inferentia are purpose-built ONLY for AI workloads
- Purpose-built = more efficient = cheaper per computation
- Trade-off: Less framework flexibility (mainly PyTorch via Neuron SDK)

### 9.2 When to Use AWS Chips vs NVIDIA GPUs

| Use AWS Trainium/Inferentia When... | Use NVIDIA GPUs When... |
|-------------------------------------|-------------------------|
| Cost is the #1 priority | You need maximum framework flexibility |
| Running PyTorch models | Using TensorFlow, JAX, or custom CUDA kernels |
| Standard architectures (Transformers) | Experimental/custom architectures |
| High-volume inference at scale | Small-scale or prototyping |
| You can use the Neuron SDK | You need CUDA-specific libraries |

### 9.3 Full Cost Optimization Strategies (Expanded)

| Strategy | What It Means | Savings | Trade-off |
|----------|--------------|---------|------------|
| **Managed Spot Training** | Use spare AWS capacity (can be interrupted) | Up to 90% | Job may be interrupted; use checkpointing |
| **Warm Pools** | Keep instances ready between training jobs | Faster starts (no cold boot) | Pay for idle warm time |
| **Heterogeneous Clusters** | Mix cheap CPU instances with expensive GPU instances | 30-40% | More complex configuration |
| **AWS Trainium** | Use AWS's custom training chip instead of NVIDIA | ~50% vs GPU | Limited to PyTorch + Neuron SDK |
| **AWS Inferentia** | Use AWS's custom inference chip instead of NVIDIA | ~70% vs GPU | Limited model architecture support |
| **Serverless Inference** | No always-on instance; pay only when requests come | 100% during idle | Cold start latency (seconds) |
| **Auto-scaling** | Scale instances up/down based on traffic | Match demand exactly | Brief latency during scale-up |
| **SageMaker Neo** | Compile model for specific hardware (optimizes execution) | 2× throughput | Compilation step required |
| **Training Plans** | Reserve GPU capacity in advance | Predictable pricing | Commitment required |

---

## 10. SLM Fine-Tuning Examples: Hugging Face & Gemma

### 10.1 What is SLM Fine-Tuning?

**SLM** = Small Language Model (typically 1B–7B parameters)

Fine-tuning an SLM means taking a pre-trained small model (like Gemma 2B, Phi-3, or DistilBERT) and training it further on YOUR specific data so it becomes an expert in YOUR domain.

**Why SLMs over LLMs?**
- 10-100× cheaper to fine-tune and serve
- Can run on a single GPU
- Faster inference (lower latency)
- Can be deployed on edge devices
- Often sufficient for focused tasks (classification, extraction, summarization)

### 10.2 Example: Fine-Tune Google Gemma 2B on SageMaker

> **Story**: "I want to fine-tune Google's Gemma 2B model on my customer support conversations so it can auto-respond to common questions."

#### Step 1: Prepare Training Data

```python
# training_data.jsonl format (instruction fine-tuning)
# Each line is a JSON object:
# {"prompt": "Customer question", "completion": "Ideal response"}

import json

data = [
    {"prompt": "How do I reset my password?", "completion": "Go to Settings > Security > Reset Password. You'll receive an email with a reset link."},
    {"prompt": "What's your refund policy?", "completion": "We offer full refunds within 30 days of purchase. Contact support@example.com to initiate."},
    {"prompt": "How do I cancel my subscription?", "completion": "Navigate to Account > Subscription > Cancel. Your access continues until the billing period ends."}
]

with open("train.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# Upload to S3
import boto3
s3 = boto3.client("s3")
s3.upload_file("train.jsonl", "my-ml-bucket", "gemma-finetune/train.jsonl")
```

#### Step 2: Fine-Tune Gemma 2B with Hugging Face on SageMaker

```python
import sagemaker
from sagemaker.huggingface import HuggingFace

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Hugging Face Estimator for Gemma 2B fine-tuning
huggingface_estimator = HuggingFace(
    entry_point="train.py",              # Your training script
    source_dir="./scripts",              # Directory with train.py
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",       # 1× A10G GPU (24 GB) — enough for 2B model
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters={
        "model_id": "google/gemma-2b",
        "epochs": 3,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-5,
        "max_seq_length": 512,
        "lora_r": 16,                    # LoRA rank (parameter-efficient fine-tuning)
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    environment={
        "HUGGING_FACE_HUB_TOKEN": "<your-hf-token>"  # Gemma requires acceptance of license
    }
)

# Start training
huggingface_estimator.fit({
    "training": "s3://my-ml-bucket/gemma-finetune/"
})
```

#### Step 3: The Training Script (train.py)

```python
# scripts/train.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Apply LoRA (Parameter-Efficient Fine-Tuning)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows only ~0.5% params are trainable

    # Load dataset
    train_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    dataset = load_dataset("json", data_files=f"{train_dir}/train.jsonl", split="train")

    # Tokenize
    def tokenize(example):
        text = f"### Question: {example['prompt']}\n### Answer: {example['completion']}"
        return tokenizer(text, truncation=True, max_length=args.max_seq_length, padding="max_length")

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch"
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()

    # Save
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    main()
```

#### Step 4: Deploy the Fine-Tuned Model

```python
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_data=huggingface_estimator.model_data,  # S3 path to trained model
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310"
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",    # 1× A10G for inference
    initial_instance_count=1,
    endpoint_name="gemma-2b-support-bot"
)

# Test
result = predictor.predict({
    "inputs": "### Question: How do I upgrade my plan?\n### Answer:"
})
print(result[0]["generated_text"])
```

**Cost Breakdown:**
- Training: `ml.g5.2xlarge` × ~1 hour = ~$1.21
- Inference: `ml.g5.xlarge` = ~$1.01/hr (or use Serverless for pay-per-request)

---

### 10.3 Example: Fine-Tune Hugging Face DistilBERT for Text Classification

> **Story**: "I want to classify customer emails into categories (billing, technical, general) using a small, fast model."

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point="train_classifier.py",
    source_dir="./scripts",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",      # 1× T4 GPU — plenty for 66M param model
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters={
        "model_id": "distilbert-base-uncased",
        "num_labels": 3,                  # billing, technical, general
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 5e-5
    }
)

huggingface_estimator.fit({"training": "s3://my-bucket/email-classification/"})

# Deploy — CPU is enough for this small model!
predictor = huggingface_estimator.deploy(
    instance_type="ml.c5.xlarge",         # CPU inference — $0.17/hr (no GPU needed!)
    initial_instance_count=1
)
```

**Why CPU for inference?** DistilBERT is only 66M parameters — it runs fast on CPU. No need to pay for a GPU.

---

### 10.4 Example: Fine-Tune Gemma 7B with QLoRA (4-bit Quantization)

> **Story**: "I want to fine-tune the larger Gemma 7B model but I only have budget for a single GPU."

**Solution**: Use QLoRA — quantize the model to 4-bit and only train small adapter layers.

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point="train_qlora.py",
    source_dir="./scripts",
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",       # 1× A10G (24 GB) — QLoRA makes 7B fit!
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters={
        "model_id": "google/gemma-7b",
        "epochs": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,  # Effective batch size = 8
        "learning_rate": 1e-4,
        "max_seq_length": 1024,
        "use_4bit": "true",               # 4-bit quantization!
        "lora_r": 32,
        "lora_alpha": 64
    },
    environment={"HUGGING_FACE_HUB_TOKEN": "<your-hf-token>"}
)

huggingface_estimator.fit({"training": "s3://my-bucket/gemma-7b-data/"})
```

**How QLoRA makes this possible:**

| Without QLoRA | With QLoRA |
|---------------|------------|
| 7B model in FP16 = ~14 GB | 7B model in 4-bit = ~3.5 GB |
| + Gradients = ~14 GB | + LoRA adapters = ~0.5 GB |
| + Optimizer = ~28 GB | + Optimizer (adapters only) = ~1 GB |
| **Total: ~56 GB** (needs 4 GPUs) | **Total: ~5 GB** (fits on 1 GPU!) |

---

### 10.5 SLM Fine-Tuning Compute Comparison

| Model | Parameters | Technique | Instance | GPUs | GPU Memory Used | Training Time | Cost |
|-------|-----------|-----------|----------|------|-----------------|---------------|------|
| DistilBERT | 66M | Full fine-tune | `ml.g4dn.xlarge` | 1× T4 | ~4 GB | ~30 min | ~$0.27 |
| Gemma 2B | 2B | LoRA | `ml.g5.xlarge` | 1× A10G | ~8 GB | ~1 hr | ~$1.01 |
| Gemma 2B | 2B | Full fine-tune | `ml.g5.2xlarge` | 1× A10G | ~16 GB | ~2 hr | ~$2.42 |
| Phi-3 Mini (3.8B) | 3.8B | LoRA | `ml.g5.2xlarge` | 1× A10G | ~12 GB | ~1.5 hr | ~$1.82 |
| Gemma 7B | 7B | QLoRA (4-bit) | `ml.g5.2xlarge` | 1× A10G | ~8 GB | ~3 hr | ~$3.63 |
| Gemma 7B | 7B | LoRA (FP16) | `ml.g5.12xlarge` | 4× A10G | ~20 GB | ~1.5 hr | ~$8.51 |
| Llama 3 8B | 8B | QLoRA (4-bit) | `ml.g5.2xlarge` | 1× A10G | ~10 GB | ~4 hr | ~$4.84 |
| Mistral 7B | 7B | LoRA | `ml.g5.4xlarge` | 1× A10G | ~18 GB | ~2 hr | ~$3.24 |

---

## 11. Conclusion

Amazon SageMaker AI is the most comprehensive ML platform on AWS, covering every stage of the ML lifecycle. The key decision points are:

1. **No code?** → Canvas
2. **Standard ML?** → Built-in algorithms + `ml.m5` instances
3. **Deep learning?** → Script mode + `ml.g5` GPU instances
4. **Foundation models?** → JumpStart + `ml.g5`/`ml.p4d` instances
5. **Massive scale?** → HyperPod + `ml.p5`/`ml.trn1` clusters
6. **Cost-sensitive?** → Spot instances + Trainium/Inferentia
7. **Production?** → Real-time endpoints + Model Monitor + Pipelines

The platform grows with you — from a single CSV in Canvas to training 100B+ parameter models across thousands of GPUs.

---

## References

1. What is SageMaker AI: https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html
2. SageMaker Training: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html
3. SageMaker Features: https://docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html
4. Deploy Models: https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html
5. Distributed Training: https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training-get-started.html
6. SageMaker Pricing: https://aws.amazon.com/sagemaker/pricing/
7. Instance Types: https://docs.aws.amazon.com/sagemaker/latest/dg/cmn-info-instance-types.html
8. Hugging Face on SageMaker: https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html
9. Real-time Inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
10. AWS Trainium: https://aws.amazon.com/machine-learning/trainium/
11. AWS Inferentia: https://aws.amazon.com/machine-learning/inferentia/

---

*Paper generated based on AWS official documentation. Pricing is approximate and varies by Region. Visit https://aws.amazon.com/sagemaker/pricing/ for current rates.*
