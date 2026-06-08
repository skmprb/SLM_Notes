# Research Paper: AWS AI Services — Complete Catalog & Guide

## Abstract

Beyond Amazon Bedrock, SageMaker, Connect, and Lex, AWS offers 20+ purpose-built AI services that provide ready-made intelligence for specific tasks — from vision and speech to forecasting and fraud detection. This paper catalogs every AWS AI service, explains what each does, when to use it, provides code examples, and maps them to real-world use cases with a story-driven approach.

---

## 1. The AWS AI Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: AI APPLICATIONS (Ready-to-use)                             │
│  Amazon Q Business | Amazon Q Developer | PartyRock                   │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2: AI SERVICES (Pre-trained, API-based)                       │
│  Vision: Rekognition | Textract | Lookout for Vision                 │
│  Language: Comprehend | Translate | Kendra                           │
│  Speech: Polly | Transcribe                                          │
│  Conversation: Lex | Connect                                         │
│  Recommendations: Personalize | Forecast | Fraud Detector            │
│  Healthcare: Comprehend Medical | HealthScribe | HealthLake           │
│  Industrial: Lookout for Equipment | Monitron | Panorama             │
│  DevOps: DevOps Guru | CodeGuru                                      │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1: ML PLATFORMS (Build your own)                              │
│  Amazon Bedrock | Amazon SageMaker AI                                │
├─────────────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE: Trainium | Inferentia | GPUs | EFA                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Vision & Document AI

### 2.1 Amazon Rekognition — Image & Video Analysis

| Aspect | Details |
|--------|---------|
| **What** | Pre-trained computer vision for image/video analysis |
| **Input** | Images (JPEG/PNG) or video streams |
| **Capabilities** | Object detection, face analysis, celebrity recognition, text in images, content moderation, custom labels, PPE detection |
| **Use Cases** | Identity verification, content moderation, media search, security surveillance, retail analytics |
| **Pricing** | ~$0.001 per image (first 1M), video: ~$0.10/min |

```python
import boto3

rekognition = boto3.client('rekognition')

# Detect objects in an image
response = rekognition.detect_labels(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'photo.jpg'}},
    MaxLabels=10,
    MinConfidence=80
)

for label in response['Labels']:
    print(f"{label['Name']}: {label['Confidence']:.1f}%")
# Output: Person: 99.2%, Car: 95.8%, Road: 92.1%
```

**Story**: *"We need to automatically blur faces in user-uploaded photos for privacy compliance."* → Use Rekognition DetectFaces + blur coordinates.

---

### 2.2 Amazon Textract — Document Text Extraction

| Aspect | Details |
|--------|---------|
| **What** | Extract text, tables, forms, and key-value pairs from documents |
| **Input** | PDFs, images (scanned documents, receipts, invoices) |
| **Capabilities** | OCR, table extraction, form extraction, query-based extraction, expense analysis, ID document analysis |
| **Use Cases** | Invoice processing, loan applications, tax forms, ID verification, medical records |
| **Pricing** | ~$0.0015/page (detect text), ~$0.05/page (forms/tables) |

```python
import boto3

textract = boto3.client('textract')

# Extract text from a document
response = textract.analyze_document(
    Document={'S3Object': {'Bucket': 'my-bucket', 'Name': 'invoice.pdf'}},
    FeatureTypes=['TABLES', 'FORMS']
)

# Extract key-value pairs (form fields)
for block in response['Blocks']:
    if block['BlockType'] == 'KEY_VALUE_SET':
        print(f"Key: {block.get('Text', '')}")
```

**Story**: *"We process 10,000 invoices/month manually. It takes 3 people full-time."* → Textract automates extraction in seconds per document.

---

### 2.3 Amazon Lookout for Vision — Visual Defect Detection

| Aspect | Details |
|--------|---------|
| **What** | Detect defects and anomalies in images using computer vision |
| **Input** | Product images from manufacturing lines |
| **Use Cases** | Quality control, defect detection, damage assessment, compliance |
| **ML Required** | No — just provide 30+ normal images and a few defect images |

**Story**: *"Our factory produces 50,000 widgets/day. Manual inspection catches only 85% of defects."* → Lookout for Vision catches 99%+ with camera integration.

---

## 3. Language & Text AI

### 3.1 Amazon Comprehend — NLP & Text Analytics

| Aspect | Details |
|--------|---------|
| **What** | Natural Language Processing for text analysis |
| **Capabilities** | Sentiment analysis, entity extraction, key phrases, language detection, topic modeling, PII detection, custom classification |
| **Use Cases** | Social media monitoring, customer feedback analysis, document classification, compliance scanning |
| **Pricing** | ~$0.0001 per unit (100 characters) |

```python
import boto3

comprehend = boto3.client('comprehend')

# Sentiment analysis
response = comprehend.detect_sentiment(
    Text="I absolutely love this product! Best purchase ever.",
    LanguageCode='en'
)
print(f"Sentiment: {response['Sentiment']}")  # POSITIVE
print(f"Confidence: {response['SentimentScore']['Positive']:.2f}")  # 0.99

# Entity extraction
entities = comprehend.detect_entities(
    Text="Jeff Bezos founded Amazon in Seattle in 1994.",
    LanguageCode='en'
)
for entity in entities['Entities']:
    print(f"{entity['Text']} → {entity['Type']} ({entity['Score']:.2f})")
# Jeff Bezos → PERSON (0.99)
# Amazon → ORGANIZATION (0.98)
# Seattle → LOCATION (0.99)
# 1994 → DATE (0.99)
```

**Story**: *"We get 5,000 customer reviews/day. We need to know which products have negative sentiment trends."* → Comprehend analyzes all reviews in real-time.

---

### 3.2 Amazon Comprehend Medical — Healthcare NLP

| Aspect | Details |
|--------|---------|
| **What** | HIPAA-eligible NLP for medical text |
| **Capabilities** | Extract medical conditions, medications, dosages, procedures, anatomy; map to ICD-10-CM, RxNorm, SNOMED CT |
| **Use Cases** | Clinical note processing, insurance claims, pharmacovigilance, population health |

```python
comprehend_medical = boto3.client('comprehendmedical')

response = comprehend_medical.detect_entities_v2(
    Text="Patient presents with Type 2 diabetes. Prescribed Metformin 500mg twice daily."
)
# Extracts: Condition=Type 2 diabetes, Medication=Metformin, Dosage=500mg, Frequency=twice daily
```

---

### 3.3 Amazon Translate — Neural Machine Translation

| Aspect | Details |
|--------|---------|
| **What** | Real-time language translation |
| **Languages** | 75+ languages |
| **Capabilities** | Text translation, document translation (PDF, Word, PowerPoint), custom terminology, formality control, profanity masking |
| **Pricing** | ~$15 per million characters |

```python
translate = boto3.client('translate')

result = translate.translate_text(
    Text="Hello, how can I help you today?",
    SourceLanguageCode='en',
    TargetLanguageCode='es'
)
print(result['TranslatedText'])  # "Hola, ¿cómo puedo ayudarte hoy?"
```

**Story**: *"Our e-commerce site serves 20 countries. We need product descriptions in 12 languages."* → Translate handles it via API, with custom terminology for brand names.

---

### 3.4 Amazon Kendra — Intelligent Enterprise Search

| Aspect | Details |
|--------|---------|
| **What** | ML-powered enterprise search that understands natural language questions |
| **Data Sources** | S3, SharePoint, Salesforce, ServiceNow, databases, websites, Confluence, Google Drive |
| **Capabilities** | Natural language queries, FAQ extraction, document ranking, incremental learning, access control |
| **Use Cases** | Internal knowledge base, customer self-service, research, compliance |
| **Pricing** | Enterprise Edition: ~$1,008/month base |

**Story**: *"Our employees waste 2 hours/day searching across 15 different systems for information."* → Kendra indexes all sources and answers questions directly.

---

## 4. Speech AI

### 4.1 Amazon Polly — Text-to-Speech

| Aspect | Details |
|--------|---------|
| **What** | Convert text to lifelike speech |
| **Voices** | 60+ voices across 30+ languages |
| **Types** | Standard TTS, Neural TTS (NTTS), Long-Form, Newscaster style, Brand Voice (custom) |
| **Output** | MP3, OGG, PCM audio streams |
| **Use Cases** | IVR systems, audiobooks, accessibility, e-learning, news reading |
| **Pricing** | Standard: $4/1M chars, Neural: $16/1M chars |

```python
polly = boto3.client('polly')

response = polly.synthesize_speech(
    Text="Welcome to TechFlow Solutions. How can I help you today?",
    OutputFormat='mp3',
    VoiceId='Joanna',          # Neural voice
    Engine='neural'
)

# Save audio file
with open('welcome.mp3', 'wb') as f:
    f.write(response['AudioStream'].read())
```

---

### 4.2 Amazon Transcribe — Speech-to-Text

| Aspect | Details |
|--------|---------|
| **What** | Convert speech/audio to text |
| **Capabilities** | Real-time streaming, batch transcription, speaker diarization, custom vocabulary, automatic language detection, PII redaction, toxic speech detection, subtitles (SRT/VTT) |
| **Languages** | 100+ languages |
| **Use Cases** | Meeting transcription, call center analytics, subtitles, medical dictation, compliance |
| **Pricing** | ~$0.024/min (standard), ~$0.048/min (medical) |

```python
transcribe = boto3.client('transcribe')

# Start transcription job
transcribe.start_transcription_job(
    TranscriptionJobName='meeting-2025-07',
    Media={'MediaFileUri': 's3://my-bucket/meeting.mp4'},
    MediaFormat='mp4',
    LanguageCode='en-US',
    Settings={
        'ShowSpeakerLabels': True,
        'MaxSpeakerLabels': 5
    }
)
```

**Story**: *"We record all customer calls for compliance. We need searchable transcripts with speaker labels."* → Transcribe + Contact Lens handles this automatically.

---

## 5. Recommendations & Forecasting

### 5.1 Amazon Personalize — ML Recommendations

| Aspect | Details |
|--------|---------|
| **What** | Real-time personalized recommendations (same tech as Amazon.com) |
| **Input** | User interactions (clicks, purchases, views) + item catalog + user demographics |
| **Capabilities** | Product recommendations, personalized search, user segmentation, similar items, trending items |
| **Use Cases** | E-commerce, streaming (movies/music), news feeds, marketing campaigns |
| **Pricing** | Pay for data ingestion + training + inference |

```python
personalize_runtime = boto3.client('personalize-runtime')

# Get recommendations for a user
response = personalize_runtime.get_recommendations(
    campaignArn='arn:aws:personalize:us-east-1:<account>:campaign/my-campaign',
    userId='user-123',
    numResults=5
)

for item in response['itemList']:
    print(f"Recommended: {item['itemId']} (score: {item['score']:.3f})")
```

**Story**: *"Our streaming platform has 1M users and 50K movies. We need Netflix-style 'Because you watched...' recommendations."* → Personalize builds this in days, not months.

---

### 5.2 Amazon Forecast — Time Series Forecasting

| Aspect | Details |
|--------|---------|
| **What** | ML-powered time series forecasting (up to 50% more accurate than traditional methods) |
| **Input** | Historical time series data + related variables (weather, promotions, events) |
| **Use Cases** | Demand planning, inventory management, workforce scheduling, financial forecasting, capacity planning |
| **Pricing** | Pay for data storage + training + forecasts generated |

**Story**: *"We overstock 30% of products and understock 15%. We lose $2M/year on inventory mismanagement."* → Forecast predicts demand per SKU per location per week.

---

### 5.3 Amazon Fraud Detector — ML Fraud Detection

| Aspect | Details |
|--------|---------|
| **What** | Identify potentially fraudulent activity using ML + 20 years of Amazon fraud expertise |
| **Capabilities** | Online payment fraud, new account fraud, account takeover, loyalty program abuse |
| **Input** | Transaction data (amount, IP, device, email, etc.) |
| **Output** | Fraud score + rules-based decisions |
| **Pricing** | Pay per prediction |

**Story**: *"We lose $500K/year to fraudulent transactions. Our rules-based system catches only 60%."* → Fraud Detector catches 95%+ with custom ML models trained on YOUR data.

---

## 6. Healthcare AI

### 6.1 AWS HealthScribe — Clinical Documentation

| Aspect | Details |
|--------|---------|
| **What** | HIPAA-eligible service that generates clinical notes from patient-clinician conversations |
| **Capabilities** | Transcription, speaker identification, medical term extraction, clinical note generation with citations |
| **Use Cases** | Reduce clinician documentation burden, EHR integration |

### 6.2 AWS HealthLake — Healthcare Data Lake

| Aspect | Details |
|--------|---------|
| **What** | HIPAA-eligible FHIR-based data store for healthcare data |
| **Capabilities** | Store, transform, query, analyze health data at scale using FHIR R4 format |
| **Use Cases** | Population health analytics, clinical research, care quality reporting |

---

## 7. Industrial AI

### 7.1 Amazon Lookout for Equipment — Predictive Maintenance

| Aspect | Details |
|--------|---------|
| **What** | Detect equipment anomalies from sensor data |
| **Input** | Sensor data (pressure, temperature, vibration, flow rate) |
| **ML Required** | No — auto-trains on your equipment's normal behavior |
| **Use Cases** | Prevent machine failures, reduce downtime, optimize maintenance schedules |

### 7.2 Amazon Monitron — End-to-End Equipment Monitoring

| Aspect | Details |
|--------|---------|
| **What** | Complete system: sensors + gateway + ML + mobile app |
| **Includes** | Physical sensors, gateway device, cloud ML service, mobile app |
| **Use Cases** | Factory floor monitoring, predictive maintenance without IT infrastructure |

### 7.3 AWS Panorama — Computer Vision at the Edge

| Aspect | Details |
|--------|---------|
| **What** | Add computer vision to existing IP cameras |
| **Capabilities** | Run CV models on-premises on existing camera feeds |
| **Use Cases** | Workplace safety, retail analytics, quality inspection, traffic monitoring |

---

## 8. Developer & Operations AI

### 8.1 Amazon Q Developer — AI Coding Assistant

| Aspect | Details |
|--------|---------|
| **What** | AI-powered assistant for software development (successor to CodeWhisperer) |
| **Capabilities** | Code generation, code completion, code transformation, debugging, security scanning, /dev /test /review agents |
| **IDEs** | VS Code, JetBrains, Visual Studio, AWS Console, CLI |
| **Languages** | Python, JavaScript, TypeScript, Java, C#, Go, Rust, and more |
| **Pricing** | Free tier available; Pro: $19/user/month |

### 8.2 Amazon Q Business — Enterprise AI Assistant

| Aspect | Details |
|--------|---------|
| **What** | GenAI assistant that answers questions from your enterprise data |
| **Data Sources** | S3, SharePoint, Confluence, Slack, Salesforce, ServiceNow, Jira, Gmail, and 40+ connectors |
| **Capabilities** | Q&A, summarization, content generation, task completion, plugins |
| **Security** | Respects existing access controls (ACLs) |

### 8.3 Amazon DevOps Guru — Operational Anomaly Detection

| Aspect | Details |
|--------|---------|
| **What** | ML-powered operational insights for AWS applications |
| **Capabilities** | Detect anomalies (latency spikes, error rates), identify root cause, recommend remediation |
| **Input** | CloudWatch metrics, CloudFormation stacks, tagged resources |
| **Use Cases** | Prevent outages, reduce MTTR, proactive monitoring |

### 8.4 Amazon CodeGuru — Code Quality & Performance

| Aspect | Details |
|--------|---------|
| **What** | ML-powered code reviews and application profiling |
| **Reviewer** | Identifies security vulnerabilities, bugs, code quality issues |
| **Profiler** | Finds most expensive lines of code, optimizes performance |

---

## 9. Other AI Services

### 9.1 Amazon Augmented AI (A2I) — Human-in-the-Loop

| Aspect | Details |
|--------|---------|
| **What** | Add human review to ML predictions when confidence is low |
| **Integrates With** | Textract, Rekognition, SageMaker, custom models |
| **Use Cases** | Review low-confidence predictions, compliance verification, training data creation |

### 9.2 Amazon PartyRock — No-Code GenAI App Builder

| Aspect | Details |
|--------|---------|
| **What** | Code-free generative AI app builder for learning and experimentation |
| **Powered By** | Amazon Bedrock foundation models |
| **Use Cases** | Education, prototyping, prompt engineering practice |

### 9.3 AWS DeepRacer — Reinforcement Learning

| Aspect | Details |
|--------|---------|
| **What** | 1/18th scale autonomous race car for learning reinforcement learning |
| **Use Cases** | Education, ML competitions, RL experimentation |

---

## 10. Decision Guide: "I Need To..." → Use This Service

| I Need To... | Use This AWS Service |
|-------------|---------------------|
| Detect objects/faces in images | **Rekognition** |
| Extract text from scanned documents | **Textract** |
| Detect manufacturing defects | **Lookout for Vision** |
| Analyze sentiment in text | **Comprehend** |
| Extract medical info from clinical notes | **Comprehend Medical** |
| Translate text between languages | **Translate** |
| Search across enterprise documents | **Kendra** |
| Build a chatbot | **Lex** |
| Run a contact center | **Connect** |
| Convert text to speech | **Polly** |
| Convert speech to text | **Transcribe** |
| Generate clinical notes from conversations | **HealthScribe** |
| Recommend products to users | **Personalize** |
| Forecast demand/sales | **Forecast** |
| Detect fraudulent transactions | **Fraud Detector** |
| Monitor equipment health | **Lookout for Equipment** / **Monitron** |
| Add CV to existing cameras | **Panorama** |
| Get AI coding assistance | **Q Developer** |
| Answer questions from company data | **Q Business** |
| Detect operational anomalies | **DevOps Guru** |
| Review ML predictions with humans | **Augmented AI (A2I)** |
| Build/train custom ML models | **SageMaker AI** |
| Use foundation models (LLMs) | **Bedrock** |
| Build AI agents | **Bedrock AgentCore** |

---

## 11. User Story: "SmartRetail" — Using Multiple AI Services Together

### The Company
**SmartRetail** — an online retailer with 500K customers, 20K products, and a physical warehouse.

### The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMARTRETAIL AI ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CUSTOMER-FACING:                                                │
│  ├── Amazon Lex → Chatbot (order status, returns, FAQ)          │
│  ├── Amazon Connect → Phone support with Lex IVR                │
│  ├── Amazon Personalize → "Recommended for you" on homepage     │
│  ├── Amazon Translate → Product pages in 12 languages           │
│  └── Amazon Polly → Voice notifications for delivery updates    │
│                                                                  │
│  OPERATIONS:                                                     │
│  ├── Amazon Forecast → Demand prediction per SKU per week       │
│  ├── Amazon Fraud Detector → Payment fraud scoring              │
│  ├── Amazon Textract → Invoice/receipt processing               │
│  ├── Amazon Rekognition → Product image moderation              │
│  ├── Amazon Lookout for Vision → Warehouse QC (defect detect)   │
│  └── Amazon Monitron → Conveyor belt predictive maintenance     │
│                                                                  │
│  ANALYTICS:                                                      │
│  ├── Amazon Comprehend → Customer review sentiment analysis     │
│  ├── Amazon Transcribe → Call center transcription              │
│  ├── Amazon Kendra → Internal knowledge search for agents       │
│  └── Amazon DevOps Guru → Application health monitoring         │
│                                                                  │
│  DEVELOPMENT:                                                    │
│  ├── Amazon Q Developer → AI-assisted coding                    │
│  ├── Amazon Bedrock → GenAI product descriptions                │
│  └── Amazon SageMaker → Custom demand forecasting model         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Results

| Area | Before AI | After AI | Impact |
|------|-----------|----------|--------|
| Customer support | 100% human | 65% automated | -$200K/year |
| Product recommendations | Generic "bestsellers" | Personalized per user | +18% revenue |
| Inventory accuracy | 70% | 95% | -$500K waste |
| Fraud losses | $300K/year | $45K/year | -85% fraud |
| Document processing | 3 FTEs | Automated | -$180K/year |
| Equipment downtime | 40 hrs/month | 5 hrs/month | -87% downtime |
| Multilingual content | Manual translation | Automated | 10× faster |

**Total Annual Impact**: ~$1.5M savings + 18% revenue increase

---

## 12. Pricing Summary

| Service | Pricing Model | Free Tier |
|---------|--------------|-----------|
| Rekognition | Per image/video minute | 5,000 images/month (12 months) |
| Textract | Per page | 1,000 pages/month (3 months) |
| Comprehend | Per 100 characters | 50K units/month (12 months) |
| Translate | Per million characters | 2M characters/month (12 months) |
| Polly | Per million characters | 5M chars/month (12 months) |
| Transcribe | Per minute | 60 minutes/month (12 months) |
| Lex | Per request | 10K text + 5K speech/month (12 months) |
| Personalize | Per data + training + inference | 2 months free |
| Forecast | Per forecasts generated | — |
| Fraud Detector | Per prediction | — |
| Kendra | Per index + queries | — |
| Q Developer | Per user/month | Free tier available |
| Q Business | Per user/month | — |

---

## 13. Conclusion

AWS provides 25+ purpose-built AI services that cover every domain — vision, language, speech, recommendations, healthcare, industrial, and developer tools. The key insight is that these services are **pre-trained and API-based** — you don't need ML expertise to use them. They're designed to be combined: Lex + Connect for voice bots, Textract + Comprehend for document intelligence, Personalize + Translate for global e-commerce, Transcribe + Comprehend for call analytics.

For custom needs beyond what pre-built services offer, SageMaker AI lets you build your own models, and Bedrock provides access to foundation models for generative AI.

---

## References

1. AWS AI/ML Overview: https://docs.aws.amazon.com/whitepapers/latest/aws-overview/machine-learning.html
2. Choosing an AWS ML Service: https://docs.aws.amazon.com/decision-guides/latest/machine-learning-on-aws-how-to-choose/guide.html
3. Amazon Rekognition: https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html
4. Amazon Textract: https://docs.aws.amazon.com/textract/latest/dg/what-is.html
5. Amazon Comprehend: https://docs.aws.amazon.com/comprehend/latest/dg/what-is.html
6. Amazon Translate: https://docs.aws.amazon.com/translate/latest/dg/what-is.html
7. Amazon Polly: https://docs.aws.amazon.com/polly/latest/dg/what-is.html
8. Amazon Transcribe: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
9. Amazon Personalize: https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html
10. Amazon Forecast: https://docs.aws.amazon.com/forecast/latest/dg/what-is-forecast.html
11. Amazon Fraud Detector: https://docs.aws.amazon.com/frauddetector/latest/ug/what-is-frauddetector.html
12. Amazon Kendra: https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html
13. AWS HealthScribe: https://docs.aws.amazon.com/healthscribe/latest/devguide/what-is.html

---

*Paper generated based on AWS official documentation. Services and pricing are subject to change.*
