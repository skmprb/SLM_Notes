# Chapter 5: Boosting RAG Performance with Expert Human Feedback

## 1. Why Human Feedback (HF) is Essential

**Core Problem**: Even best metrics cannot convince dissatisfied users

**Key Insight**: 
- Parametric data (model weights): Fixed, requires retraining
- Non-parametric data (RAG documents): Visible, tweakable, controllable
- HF directly influences quality of future responses

**Adaptive RAG**: System adapts based on human feedback loops

---

## 2. Adaptive RAG Ecosystem

### Components (from Figure 1.3):

**D - Retriever**:
- D1: Collect & process (Wikipedia articles on LLMs)
- D4: Retrieval query

**G - Generator**:
- G1: User input
- G2: Augmented input with HF
- G3: Prompt engineering
- G4: Generation & output (GPT-4o)

**E - Evaluator**:
- E1: Metrics (cosine similarity, response time)
- E2: Human feedback (end-user + expert ratings)

---

## 3. Hybrid Adaptive RAG Definition

**Hybrid**:
- Integrates HF within retrieval process
- Real-time human-machine collaboration
- Documents labeled (manually or automatically)
- Expert feedback documents + raw documents

**Adaptive**:
- User rankings trigger system changes
- Expert feedback loop
- Automated document re-ranking
- Dynamic response adjustment

**Key Principle**: In real-life AI, what works, works!

---

## 4. Use Case: Company C

**Scenario**: Deploy conversational agent explaining AI/LLMs

**Problem**: 
- C-phone series technical issues
- Too many customer support requests
- Teams unconvinced about AI solutions

**Solution**: Proof of concept with adaptive RAG
- Prove AI works before scaling
- Customize for specific project
- Build ground-up skills
- Establish data governance
- Solve problems during POC

---

## 5. Automated RAG Triggers (Score-Based)

### Three Operational Modes:

**Ranking 1-2: No RAG**
- System lacks compensatory capability
- RAG temporarily deactivated
- Maintenance or fine-tuning needed
- User input processed without retrieval

**Ranking 3-4: Human-Expert Feedback Only**
- Document-based RAG deactivated
- Expert flashcards/snippets augment input
- No new user feedback required

**Ranking 5: Keyword-Search RAG + HF**
- Full RAG with previous HF
- Flashcards/snippets when necessary
- Optimal performance mode

**Note**: Scoring system varies per project—organize user workshops to define triggers

---

## 6. Implementation Architecture

### 1. Retriever

**1.1 Environment**:
```python
!pip install requests==2.32.3
!pip install beautifulsoup4==4.12.3
```

**1.2 Dataset Preparation**:
```python
urls = {
    "prompt engineering": "https://en.wikipedia.org/wiki/Prompt_engineering",
    "artificial intelligence": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "llm": "https://en.wikipedia.org/wiki/Large_language_model",
    "llms": "https://en.wikipedia.org/wiki/Large_language_model"
}
```

**1.3 Retrieval Process**:
```python
def process_query(user_input, num_words):
    user_input = user_input.lower()
    matched_keyword = next((keyword for keyword in urls if keyword in user_input), None)
    
    if matched_keyword:
        cleaned_text = fetch_and_clean(urls[matched_keyword])
        words = cleaned_text.split()
        first_n_words = ' '.join(words[:num_words])
        
        prompt = f"Summarize the following information about {matched_keyword}: {first_n_words}"
        return first_n_words
    else:
        return None
```

---

## 7. Generator Implementation

### 2.1 Adaptive RAG Selection:
```python
ranking = 1  # User panel mean score (1-5)

# Scenario 1: No RAG (ranking 1-2)
if ranking >= 1 and ranking < 3:
    text_input = user_input

# Scenario 2: HF only (ranking 3-4)
hf = False
if ranking > 3 and ranking < 5:
    hf = True
    # Load expert feedback document
    with open('human_feedback.txt', 'r') as file:
        content = file.read().replace('\n', ' ')
    text_input = content

# Scenario 3: Full RAG (ranking 5)
if ranking >= 5:
    max_words = 100
    rdata = process_query(user_input, max_words)
    text_input = rdata
```

### 2.2 User Input:
```python
user_input = input("Enter your query: ").lower()
# Example: "What is an LLM?"
```

### 2.6 Content Generation:
```python
from openai import OpenAI
import time

client = OpenAI()
gptmodel = "gpt-4o"

def call_gpt4_with_full_text(itext):
    text_input = '\n'.join(itext)
    prompt = f"Please summarize or elaborate: {text_input}"
    
    response = client.chat.completions.create(
        model=gptmodel,
        messages=[
            {"role": "system", "content": "You are an expert NLP assistant."},
            {"role": "assistant", "content": "You can explain LLMs clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

start_time = time.time()
gpt4_response = call_gpt4_with_full_text(text_input)
response_time = time.time() - start_time
```

---

## 8. Evaluator Implementation

### 3.1 Response Time:
```python
print(f"Response Time: {response_time:.2f} seconds")
# Example output: 7.88 seconds
```

### 3.2 Cosine Similarity:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

similarity_score = calculate_cosine_similarity(text_input, gpt4_response)
print(f"Cosine Similarity Score: {similarity_score:.3f}")
# Example output: 0.697
```

### 3.3 Human User Rating:
```python
import numpy as np

# Parameters
counter = 20          # Number of ratings
score_history = 60    # Total score
threshold = 4         # Minimum mean to avoid expert trigger

def evaluate_response(response):
    print("\nGenerated Response:")
    print(response)
    print("\n1 - Poor, 2 - Fair, 3 - Good, 4 - Very Good, 5 - Excellent")
    score = input("Enter score (1-5): ")
    
    try:
        score = int(score)
        if 1 <= score <= 5:
            return score
        else:
            print("Invalid score. Enter 1-5.")
            return evaluate_response(response)
    except ValueError:
        print("Invalid input. Enter numerical value.")
        return evaluate_response(response)

score = evaluate_response(gpt4_response)
counter += 1
score_history += score
mean_score = round(np.mean(score_history / counter), 2)

print("Rankings:", counter)
print("Score history:", mean_score)
```

### 3.4 Human-Expert Evaluation:
```python
counter_threshold = 10
score_threshold = 4

if counter > counter_threshold and mean_score <= score_threshold:
    print("Human expert evaluation required for feedback loop.")
    
    # Display thumbs up/down interface
    def save_feedback(feedback):
        with open('/content/expert_feedback.txt', 'w') as f:
            f.write(feedback)
        print("Feedback saved successfully.")
    
    # Expert provides feedback via interface
    # Feedback saved to expert_feedback.txt
```

---

## 9. Output Examples by Ranking

### Ranking 1-2 (No RAG):
```
GPT-4 Response:
It seems like you're asking about "LLM" which stands for "Language Model."
An LLM is a type of artificial intelligence model designed to understand and generate text.
Examples include OpenAI's GPT and Google's BERT.
```
**Issue**: Generic, doesn't relate to Company C's needs

### Ranking 3-4 (HF Only):
```
GPT-4 Response:
A Large Language Model (LLM) is a sophisticated AI system trained on vast text data
to generate human-like responses. For Company C's C-phone series, it can:
- Handle common technical questions
- Troubleshoot problems
- Guide users through setup
- Gather customer feedback
- Escalate to humans when necessary
- Provide personalized recommendations
```
**Result**: Satisfactory—defines LLMs AND shows customer service application

### Ranking 5 (Full RAG):
```
GPT-4 Response:
A large language model (LLM) is a type of language model known for its capability
to perform general-purpose language generation through generative AI, by taking
input text and repeatedly predicting the next token.
```
**Result**: Reasonable for technical users, but may lack business context

---

## 10. Key Technical Insights

### TF-IDF Cosine Similarity:
- Measures thematic and lexical overlap
- Transforms documents into vectors
- Weighs terms by importance
- Range: 0 (no similarity) to 1 (identical)
- **Limitation**: Measures similarity, NOT accuracy

### Human Feedback Loop:
1. Users rate responses (1-5)
2. System calculates mean score
3. Low scores trigger expert review
4. Expert provides feedback/corrections
5. Feedback stored as RAG document
6. System adapts for future queries

### Adaptive Triggers:
- Automated based on mean scores
- Configurable thresholds
- Project-specific definitions
- Workshop-driven design

---

## 11. Best Practices

1. **Separate Pipelines**: Retriever, generator, evaluator as independent modules
2. **User Workshops**: Define scoring system, triggers, interface with users
3. **Expert Panel**: Select domain experts within organization
4. **Feedback Storage**: Structured format (flashcards, snippets, documents)
5. **Threshold Configuration**: Adjust based on project needs
6. **Iterative Improvement**: Continuous feedback loop
7. **Transparency**: Show users what data influences responses
8. **Version Control**: Track feedback documents and changes
9. **Metrics Combination**: Use both automated and human evaluation
10. **Proof of Concept**: Start small, prove value, then scale

---

## 12. Workflow Summary

**Step 1**: User enters query
**Step 2**: System checks mean ranking score
**Step 3**: Select RAG mode (none, HF only, full RAG)
**Step 4**: Retrieve documents (if applicable)
**Step 5**: Augment input with retrieved data
**Step 6**: Generate response with GPT-4o
**Step 7**: Measure response time
**Step 8**: Calculate cosine similarity
**Step 9**: User rates response (1-5)
**Step 10**: Update mean score
**Step 11**: Trigger expert feedback if threshold exceeded
**Step 12**: Expert provides corrections/improvements
**Step 13**: Store feedback as RAG document
**Step 14**: System adapts for future queries

---

## 13. Key Takeaways

1. **HF is essential**, not optional—metrics alone insufficient
2. **Adaptive RAG** adjusts based on user satisfaction
3. **Three operational modes** based on ranking scores
4. **Hybrid approach** combines human expertise with automation
5. **Expert feedback loop** triggered by low user ratings
6. **Cosine similarity measures overlap**, not accuracy
7. **User workshops critical** for defining system behavior
8. **Proof of concept** validates approach before scaling
9. **Separate pipelines** enable parallel team development
10. **Real-life AI**: Pragmatic solutions over theoretical purity

---

## 14. Formulas & Metrics

### Cosine Similarity (TF-IDF):
```
similarity = (A · B) / (||A|| × ||B||)
Range: [0, 1]
0 = no similarity
1 = identical texts
```

### Mean Score Calculation:
```
mean_score = score_history / counter
```

### Trigger Condition:
```
if counter > counter_threshold AND mean_score <= score_threshold:
    trigger_expert_feedback()
```

### TF-IDF Weight:
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)
TF = term frequency in document
IDF = log(total_docs / docs_containing_term)
```

---

## 15. Interview-Ready Q&A

**Q: What is Adaptive RAG?**
RAG system that adjusts retrieval and generation based on human feedback loops, adapting to user satisfaction levels.

**Q: Why is human feedback essential?**
Metrics measure similarity, not accuracy or user satisfaction. HF provides real-world validation and improvement signals.

**Q: Three RAG operational modes?**
1) No RAG (ranking 1-2): Maintenance needed
2) HF only (ranking 3-4): Expert documents augment input
3) Full RAG (ranking 5): Keyword search + previous HF

**Q: How does expert feedback loop work?**
Low user ratings → Trigger expert review → Expert provides corrections → Stored as RAG document → System adapts.

**Q: Hybrid vs Adaptive RAG?**
Hybrid: Human-machine collaboration in retrieval. Adaptive: System changes based on feedback.

**Q: When to trigger expert feedback?**
When counter > threshold AND mean_score <= threshold (e.g., >10 ratings with mean ≤4).

**Q: Cosine similarity limitation?**
Measures lexical/thematic overlap, NOT accuracy or correctness. High similarity ≠ correct response.

**Q: Why separate pipelines?**
Enable parallel team development, independent deployment, specialized focus, easier maintenance.

---

## 16. Proof of Concept Benefits

1. **Validate AI feasibility** before major investment
2. **Identify challenges early** in controlled environment
3. **Customize for specific needs** vs generic solutions
4. **Build team skills** from ground up
5. **Establish data governance** and control
6. **Demonstrate value** to stakeholders
7. **Iterate quickly** with user feedback
8. **Scale confidently** after proving concept

---

## Tools & Technologies

**Frameworks**: Custom Python implementation (ground-up)
**Models**: GPT-4o (OpenAI)
**Data Sources**: Wikipedia articles (labeled URLs)
**Evaluation**: scikit-learn (TF-IDF, cosine similarity)
**Interface**: HTML/JavaScript (Google Colab)
**Storage**: Text files (feedback documents)
**Metrics**: Response time, cosine similarity, user ratings


---

## Yes/No Questions with Answers

**Q1: Is human feedback essential in improving RAG-driven generative AI systems?**
Yes, human feedback directly enhances the quality of AI responses.

**Q2: Can the core data in a generative AI model be changed without retraining the model?**
No, the model's core data is fixed unless it is retrained.

**Q3: Does Adaptive RAG involve real-time human feedback loops to improve retrieval?**
Yes, Adaptive RAG uses human feedback to refine retrieval results.

**Q4: Is the primary focus of Adaptive RAG to replace all human input with automated responses?**
No, it aims to blend automation with human feedback.

**Q5: Can human feedback in Adaptive RAG trigger changes in the retrieved documents?**
Yes, feedback can prompt updates to retrieved documents for better responses.

**Q6: Does Company C use Adaptive RAG solely for customer support issues?**
No, it's also used for explaining AI concepts to employees.

**Q7: Is human feedback used only when the AI responses have high user ratings?**
No, feedback is often used when responses are rated poorly.

**Q8: Does the program in this chapter provide only text-based retrieval outputs?**
No, it uses both text and expert feedback for responses.

**Q9: Is the Hybrid Adaptive RAG system static, meaning it cannot adjust based on feedback?**
No, it dynamically adjusts to feedback and rankings.

**Q10: Are user rankings completely ignored in determining the relevance of AI responses?**
No, user rankings directly influence the adjustments made to a system.
