# Chapter 9: Empowering AI Models - Fine-Tuning RAG Data and Human Feedback

## 1. Core Problem: RAG Data Threshold

**Challenge**: Organizations accumulating RAG data reach threshold of non-parametric data (not pretrained on LLM), creating management issues:
- **Storage costs**: Exponential growth of vector databases
- **Retrieval resources**: Increasing computational overhead
- **Model capacity**: Limits on generative AI model handling
- **Data overlap**: Stacking data creates redundancy

**Pretrained Model Limitation**: Models trained up to cutoff date ignore new knowledge starting next day. Impossible to interact with content published after cutoff without retrieval.

**Solution**: Transform non-parametric RAG data into parametric fine-tuned data for static information that remains stable over long periods.

## 2. Static vs Dynamic Data Classification

**Dynamic Data**:
- Requires constant updates
- Cannot be fine-tuned easily
- Examples: Daily news, real-time market data, current events
- Best handled through RAG retrieval

**Static Data**:
- Remains stable for long periods
- Ideal for fine-tuning
- Examples: Hard science facts (physics, chemistry, biology), legal precedents, historical information
- Can become parametric (stored in LLM weights)

**Key Insight**: Not all organizations need exponential data like Google, Microsoft, Amazon. Many domains (hard sciences, specialized knowledge) have stable datasets suitable for fine-tuning to reduce RAG volume.

## 3. Fine-Tuning Threshold Architecture

**Threshold Indicators**:

1. **Volume of RAG Data to Process (D2)**
   - Embedding requires human and machine resources
   - Piling up static data makes no sense

2. **Volume of RAG Data to Store and Retrieve (D3)**
   - Stacking data creates overlap
   - Storage costs increase exponentially

3. **Retrieval Resource Requirements**
   - Even open-source systems require increasing management resources
   - Processing overhead grows with data volume

**Threshold Principle** (from Chapter 1): When processing (D2) and storage (D3) thresholds reached for static data, fine-tuning becomes optimal solution.

**Project-Specific Factors**: Threshold depends on volume, overlap, resources, domain requirements, update frequency.

## 4. RAG Ecosystem for Fine-Tuning

**Active Components**:

1. **Collecting (D1) and Preparing (D2) Dataset**
   - Download SciQ hard science dataset from Hugging Face
   - Process human-crafted crowdsourced questions

2. **Human Feedback (E2)**
   - SciQ dataset controlled and updated by humans
   - Simulates reliable human feedback fine-tuned to alleviate RAG volume
   - Explanations may come from human evaluations of models (Chapter 5 approach)

3. **Fine-Tuning (T2)**
   - Fine-tune cost-effective OpenAI GPT-4o-mini model
   - Transform non-parametric to parametric data

4. **Prompt Engineering (G3) and Generation/Output (G4)**
   - Engineer prompts per OpenAI recommendations
   - Display and validate outputs

5. **Metrics (E1)**
   - Monitor OpenAI Metrics interface
   - Track training loss, accuracy, usage metrics

**Inactive Components** (Not Used): D4 (validation), G1-G2 (retrieval), T1 (other transformations)

## 5. Environment Setup

**OpenAI Installation**:
```python
# Retrieve API key from file or manual entry
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline()
f.close()

# Install and configure OpenAI
!pip install openai==1.42.0
import openai
import os
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```

**Additional Packages**:
```python
!pip install jsonlines==4.0.0  # JSONL data generation
!pip install datasets==2.20.0  # Hugging Face datasets
```

**Dependency Conflicts**: Fast-moving AI packages create cross-platform conflicts (Chapter 2, Chapter 8). Accept issues as platforms continually update. May need special environment for this program.

## 6. SciQ Dataset Preparation

**Dataset Overview**:
- **Source**: Hugging Face (https://huggingface.co/datasets/sciq)
- **Type**: Human-crafted crowdsourced hard science questions
- **Domains**: Physics, chemistry, biology
- **Total Records**: 13,679 multiple-choice questions
- **Filtered Records**: 10,481 with support text

**Download and Filter**:
```python
from datasets import load_dataset
import pandas as pd

dataset_view = load_dataset("sciq", split="train")
filtered_dataset = dataset_view.filter(lambda x: x["support"] != "" and x["correct_answer"] != "")
print("Number of questions with support:", len(filtered_dataset))  # Output: 10481
```

**Data Cleaning**:
```python
df_view = pd.DataFrame(filtered_dataset)
columns_to_drop = ['distractor3', 'distractor1', 'distractor2']  # Wrong answers
df_view = df.drop(columns=columns_to_drop)
```

**Final Structure** (4 columns):
- **question**: Query text (becomes prompt)
- **correct_answer**: Validated answer
- **support**: Explanatory content
- **completion**: Merged correct_answer + support (for training)

## 7. JSONL Format Preparation

**OpenAI Requirements**: Very precise JSONL format required for fine-tuning completion models.

**Message Structure for GPT-4o-mini**:
```python
import jsonlines

items = []
for idx, row in df.iterrows():
    detailed_answer = row['correct_answer'] + " Explanation: " + row['support']
    items.append({
        "messages": [
            {"role": "system", "content": "Given a science question, reply with detailed answer"},
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": detailed_answer}
        ]
    })

# Write to JSONL file
with jsonlines.open('/content/QA_prompts_and_completions.json', mode='w') as writer:
    writer.write_all(items)
```

**Message Roles**:
- **system**: Initial instruction for LLM (provide detailed answers to science questions)
- **user**: Question from dataset (prompt)
- **assistant**: Detailed answer with explanation (completion)

**Verification**:
```python
dfile = "/content/QA_prompts_and_completions.json"
df = pd.read_json(dfile, lines=True)
df  # Verify structure before fine-tuning
```

## 8. Fine-Tuning Process

**OpenAI Client Setup**:
```python
from openai import OpenAI
import jsonlines

client = OpenAI()
```

**Upload Training File**:
```python
result_file = client.files.create(
    file=open("QA_prompts_and_completions.json", "rb"),
    purpose="fine-tune"
)
print(result_file)
param_training_file_name = result_file.id
```

**Create Fine-Tuning Job**:
```python
ft_job = client.fine_tuning.jobs.create(
    training_file=param_training_file_name,
    model="gpt-4o-mini-2024-07-18"
)
print(ft_job)
```

**Job Output Details**:
- **Job ID**: Unique identifier (e.g., ftjob-O1OEE7eEyFNJsO2Eu5otzWA8)
- **Status**: validating_files → running → succeeded/failed
- **Model**: gpt-4o-mini-2024-07-18 (cost-effective GPT-4 version)
- **Training File**: file-EUPGmm1yAd3axrQ0pyoeAKuE
- **Created At**: Timestamp (e.g., 2024-06-30 08:20:50)

**Hyperparameters** (Auto-configured by OpenAI):
- **n_epochs**: 'auto' - Automatic training cycles
- **batch_size**: 'auto' - Optimal batch size
- **learning_rate_multiplier**: 'auto' - Automatic learning rate adjustment

## 9. Monitoring Fine-Tuning Jobs

**Query Latest Jobs**:
```python
import pandas as pd
from openai import OpenAI

client = OpenAI()
response = client.fine_tuning.jobs.list(limit=3)

# Initialize lists
job_ids = []
created_ats = []
statuses = []
models = []
training_files = []
error_messages = []
fine_tuned_models = []

# Extract information
for job in response.data:
    job_ids.append(job.id)
    created_ats.append(job.created_at)
    statuses.append(job.status)
    models.append(job.model)
    training_files.append(job.training_file)
    error_message = job.error.message if job.error else None
    error_messages.append(error_message)
    fine_tuned_model = job.fine_tuned_model if hasattr(job, 'fine_tuned_model') else None
    fine_tuned_models.append(fine_tuned_model)
```

**Create Monitoring Dashboard**:
```python
df = pd.DataFrame({
    'Job ID': job_ids,
    'Created At': created_ats,
    'Status': statuses,
    'Model': models,
    'Training File': training_files,
    'Error Message': error_messages,
    'Fine-Tuned Model': fine_tuned_models
})

# Convert timestamps to readable format
df['Created At'] = pd.to_datetime(df['Created At'], unit='s')
df = df.sort_values(by='Created At', ascending=False)
df
```

**Status Values**: validating_files, running, failed, succeeded

**Retrieve Latest Model**:
```python
generation = False  # Until current model is fine-tuned

non_empty_models = df[df['Fine-Tuned Model'].notna() & (df['Fine-Tuned Model'] != '')]
if not non_empty_models.empty:
    first_non_empty_model = non_empty_models['Fine-Tuned Model'].iloc[0]
    print("The latest fine-tuned model is:", first_non_empty_model)
    generation = True
else:
    first_non_empty_model = 'None'
    print("No fine-tuned models found.")
```

**Output Example**: `ft:gpt-4o-mini-2024-07-18:personal::AHbfZfXX`

**Monitoring Options**:
- Refresh pandas DataFrame periodically
- Write code to check status and wait for completion
- Wait for OpenAI email notification

## 10. Using Fine-Tuned Model

**Define Prompt** (from original dataset):
```python
prompt = "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?"
```

**Run Fine-Tuned Model**:
```python
if generation == True:
    response = client.chat.completions.create(
        model=first_non_empty_model,
        temperature=0.0,  # Low value for hard science (no creativity needed)
        messages=[
            {"role": "system", "content": "Given a question, reply with detailed answer"},
            {"role": "user", "content": prompt}
        ]
    )
else:
    print("Error: Model is None, cannot proceed with API request")
```

**Parameters**:
- **model**: first_non_empty_model (pretrained model)
- **temperature=0.0**: No creativity for hard science completion
- **prompt**: Predefined question

**Display Raw Response**:
```python
if generation == True:
    print(response)  # ChatCompletion object with metadata
```

**Extract Response Text**:
```python
if generation == True:
    response_text = response.choices[0].message.content
    print(response_text)
```

**Format Output**:
```python
import textwrap
if generation == True:
    wrapped_text = textwrap.fill(response_text.strip(), 60)
    print(wrapped_text)
```

**Example Output**:
```
Coriolis effect Explanation: The Coriolis effect is a
phenomenon that causes moving objects, such as air and
water, to turn and twist in response to the rotation of the
Earth. It is responsible for the rotation of large weather
systems, such as hurricanes, and the direction of trade
winds and ocean currents. In the Northern Hemisphere, the
effect causes moving objects to turn to the right, while in
the Southern Hemisphere, objects turn to the left...
```

**Validation**: Output matches initial completion structure, confirming successful fine-tuning.

## 11. OpenAI Metrics Interface

**Access**: https://platform.openai.com/finetune/

**Job List View**:
- View all fine-tuning jobs
- Filter by successful/failed status
- Select job for detailed analysis

**Job Details**:
- **Status**: Completed successfully / Failed
- **Job ID**: Unique identifier for reference
- **Base model**: Pretrained model (e.g., gpt-4o-mini)
- **Output model**: Fine-tuned model identifier
- **Created at**: Job initiation timestamp
- **Trained tokens**: Total tokens processed during training
- **Epochs**: Complete passes through training data (more epochs = better learning, but risk overfitting)
- **Batch size**: Training examples per iteration (smaller = more updates, longer training)
- **LR multiplier**: Learning rate multiplier (smaller = conservative weight updates)
- **Seed**: Random number generator seed for reproducibility

**Training Loss Metric**:
- **Definition**: Model's average error on training dataset
- **Lower values**: Better fitting to training data
- **Example**: Training loss = 1.1570 (good fit)
- **Tracking**: View loss over Time and Step

**Usage Metrics**: https://platform.openai.com/usage
- Monitor cost per period and model
- Track API usage and spending

## 12. Fine-Tuning Best Practices

**Data Quality Requirements**:
- Verify training data for inconsistencies
- Check for missing values or incorrect labels
- Ensure JSON format adheres to OpenAI schema
- Validate field names, data types, structure

**Failure Troubleshooting**:
- Review error messages in monitoring dashboard
- Check JSONL format compliance
- Validate data completeness
- Adjust hyperparameters if needed

**Model Reuse**:
1. Save model name to text file or secure location
2. Run environment setup in new notebook
3. Define prompt related to training dataset
4. Enter fine-tuned model name in completion request
5. Run request and analyze response

**Incremental Improvement**:
- Start with small dataset
- Validate results
- Incrementally add better data or larger volumes
- Iterate until satisfactory goal reached

## 13. When to Fine-Tune vs RAG

**Fine-Tune When**:
- Data is static (stable for long periods)
- High-quality, structured datasets available
- Human feedback integrated
- Storage/retrieval costs exceed fine-tuning costs
- Domain-specific knowledge needs embedding
- Consistent response format required

**Use RAG When**:
- Data is dynamic (frequent updates)
- Real-time information needed
- Content published after model cutoff date
- Flexible retrieval requirements
- Lower upfront investment preferred
- Experimental or exploratory phase

**Hybrid Approach**: Fine-tune static core knowledge + RAG for dynamic updates = optimal solution for many domains.

## 14. Human Feedback Integration

**SciQ Dataset Human Feedback**:
- Crowdsourced questions controlled by humans
- Validated answers with explanations
- Updated based on human evaluation
- Quality assurance through human review

**Feedback Sources** (Chapter 5 approach):
- Direct human evaluation of model outputs
- Expert domain knowledge validation
- Crowdsourced quality control
- Iterative refinement based on user interactions

**Feedback Benefits**:
- Improves training data quality
- Reduces hallucinations
- Ensures domain accuracy
- Aligns outputs with human expectations

**Simulation**: SciQ dataset simulates how reliable human feedback can be fine-tuned to alleviate RAG volume requirements.

## 15. Cost-Effectiveness Analysis

**Fine-Tuning Costs**:
- One-time training cost (based on tokens)
- Lower inference costs for fine-tuned models
- No ongoing storage costs for embedded data
- Reduced retrieval computational overhead

**RAG Costs**:
- Ongoing storage costs (vector databases)
- Retrieval computational resources
- Embedding costs for new data
- Higher inference costs (retrieval + generation)

**GPT-4o-mini Advantages**:
- Cost-effective version of GPT-4
- Sufficient for completion tasks
- Lower training and inference costs
- Suitable for hard science domains

**Optimization Strategy**:
- Fine-tune static data to reduce RAG volume
- Use RAG only for dynamic, time-sensitive information
- Monitor usage metrics to track cost-effectiveness
- Adjust strategy based on domain requirements

**ROI Calculation**: Compare fine-tuning one-time cost vs ongoing RAG storage/retrieval costs over project lifetime.

## Interview-Ready Q&A

**Q1: What is the RAG data threshold problem?**
A: Organizations accumulating RAG data reach threshold where non-parametric data creates storage costs, retrieval resource issues, and model capacity limits.

**Q2: What is the difference between static and dynamic data?**
A: Static data remains stable for long periods (hard science facts) and is ideal for fine-tuning; dynamic data requires constant updates (news, market data) and needs RAG.

**Q3: When should you fine-tune instead of using RAG?**
A: When data is static, high-quality, structured, and storage/retrieval costs exceed fine-tuning costs; also when consistent response format needed.

**Q4: What is the JSONL format for OpenAI fine-tuning?**
A: JSON Lines format with messages array containing system (instruction), user (prompt), and assistant (completion) roles for each training example.

**Q5: What model is used for cost-effective fine-tuning?**
A: GPT-4o-mini-2024-07-18, a smaller, cost-effective version of GPT-4 sufficient for completion tasks.

**Q6: What are the key hyperparameters in fine-tuning?**
A: n_epochs (training cycles), batch_size (examples per iteration), learning_rate_multiplier (weight update size) - all set to 'auto' by OpenAI.

**Q7: How do you monitor fine-tuning jobs?**
A: Query OpenAI API for job list, create pandas DataFrame with Job ID, Status, Model, Training File, Error Message, Fine-Tuned Model; refresh periodically.

**Q8: What is training loss and why is it important?**
A: Model's average error on training dataset; lower values indicate better fitting; example 1.1570 shows good fit to training data.

**Q9: Why set temperature=0.0 for hard science completions?**
A: Eliminates creativity/randomness for factual accuracy; hard science requires precise, deterministic answers without variation.

**Q10: What is the SciQ dataset?**
A: 13,679 crowdsourced hard science questions (physics, chemistry, biology) with validated answers and support explanations; 10,481 records with support text.

**Q11: How does human feedback integrate with fine-tuning?**
A: SciQ dataset controlled and updated by humans; explanations may come from human evaluations of model outputs; ensures quality and accuracy.

**Q12: What are the three RAG ecosystem components used?**
A: (1) Collecting/Preparing dataset (D1, D2), (2) Human feedback (E2), (3) Fine-tuning (T2), (4) Prompt engineering/Generation (G3, G4), (5) Metrics (E1).

**Q13: How do you verify fine-tuning success?**
A: Run prompts from original dataset, compare outputs to initial completions, check for matching structure and content accuracy.

**Q14: What information does the OpenAI Metrics interface provide?**
A: Status, Job ID, Base/Output models, Created timestamp, Trained tokens, Epochs, Batch size, LR multiplier, Seed, Training loss over time.

**Q15: Can you combine fine-tuning and RAG?**
A: Yes - hybrid approach: fine-tune static core knowledge + RAG for dynamic updates = optimal solution for many domains.

## Tools & Technologies

**LLM**: OpenAI GPT-4o-mini-2024-07-18 (cost-effective fine-tuning)

**Dataset**: SciQ (Hugging Face, 13,679 science questions)

**Data Format**: JSONL (JSON Lines for OpenAI fine-tuning)

**Python Libraries**: openai==1.42.0, jsonlines==4.0.0, datasets==2.20.0, pandas, textwrap

**Platform**: Google Colab (with Google Drive for API key storage)

**Monitoring**: OpenAI Dashboard (https://platform.openai.com/finetune/)

**Metrics**: OpenAI Metrics Interface (training loss, usage tracking)

**Usage Tracking**: https://platform.openai.com/usage

**API**: OpenAI API (file upload, fine-tuning jobs, chat completions)

**Data Processing**: pandas DataFrame (data cleaning, monitoring)

**Human Feedback**: Crowdsourced validation (SciQ dataset)

**Documentation**: OpenAI fine-tuning guide (https://platform.openai.com/docs/guides/fine-tuning/)

**Alternative**: Hybrid approach (fine-tuning + RAG for dynamic data)

**Cost Optimization**: One-time training vs ongoing RAG storage/retrieval costs


---

## Yes/No Questions with Answers

**Q1: Do all organizations need to manage large volumes of RAG data?**
No, many corporations only need small data volumes.

**Q2: Is the GPT-4o-mini model described as insufficient for fine-tuning tasks?**
No, GPT-4o-mini is described as cost-effective for fine-tuning tasks.

**Q3: Can pretrained models update their knowledge base after the cutoff date without retrieval systems?**
No, models are static and rely on retrieval for new information.

**Q4: Is it the case that static data never changes and thus never requires updates?**
No, only that it remains stable for a long time, not forever.

**Q5: Is downloading data from Hugging Face the only source for preparing datasets?**
Yes, Hugging Face is specifically mentioned as the data source.

**Q6: Are all RAG data eventually embedded into the trained model's parameters?**
No, non-parametric data remains external.

**Q7: Does the chapter recommend using only new data for fine-tuning AI models?**
No, it suggests fine-tuning with relevant, often stable data.

**Q8: Is the OpenAI metric interface used to adjust the learning rate during model training?**
No, it monitors performance and costs after training.

**Q9: Can the fine-tuning process be effectively monitored using the OpenAI dashboard?**
Yes, the dashboard provides real-time updates on fine-tuning jobs.

**Q10: Is human feedback deemed unnecessary in the preparation of hard science datasets such as SciQ?**
No, human feedback is crucial for data accuracy and relevance.
