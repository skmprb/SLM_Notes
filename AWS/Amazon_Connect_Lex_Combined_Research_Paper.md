# Research Paper: Amazon Connect + Amazon Lex — Combined Use Case & User Story

> This paper is structured in three parts:
> - Part 1: Amazon Connect (Contact Center)
> - Part 2: Amazon Lex (Conversational AI)
> - Part 3: Combined Use Case with User Story

---

# PART 1: Amazon Connect

## 1.1 What is Amazon Connect?

Amazon Connect (now called Amazon Connect Customer) is a cloud-based, AI-native contact center platform. It enables businesses to build and manage customer communication experiences across voice, chat, email, SMS, and tasks — with AI-powered routing, real-time analytics, and agent assistance.

**Key Principle**: Pay only for what you use. No upfront costs, no long-term contracts, no per-seat licenses.

## 1.2 Core Capabilities

| Capability | Description |
|-----------|-------------|
| **Omnichannel** | Voice, chat, email, SMS, tasks — all in one platform |
| **Contact Flows** | Visual drag-and-drop IVR/routing designer |
| **AI-Powered Routing** | Intelligent routing based on skills, priority, availability |
| **Agent Workspace** | Unified desktop with customer profiles, cases, AI assistance |
| **Contact Lens** | Real-time conversational analytics (sentiment, compliance, trends) |
| **Voice ID** | Real-time caller authentication and fraud detection |
| **Outbound Campaigns** | Predictive dialer, ML-powered answering machine detection |
| **Forecasting & Scheduling** | ML-powered workforce management |
| **Customer Profiles** | Unified customer view from multiple data sources |
| **Cases** | Built-in case management for issue tracking |
| **Amazon Q in Connect** | AI-powered agent assist (real-time recommendations) |
| **Autonomous AI Agents** | AI agents that handle customer interactions end-to-end |

## 1.3 Architecture

```
Customer (Phone/Chat/Email/SMS)
        ↓
Amazon Connect Instance
        ↓
Contact Flow (IVR Logic)
    ├── Amazon Lex Bot (NLU/Conversation)
    ├── AWS Lambda (Business Logic)
    ├── Amazon Q in Connect (Agent Assist)
    └── Queue → Agent (CCP/Agent Workspace)
        ↓
Contact Lens (Analytics/Recording)
```

## 1.4 Key Components

| Component | Purpose |
|-----------|---------|
| **Instance** | Your contact center environment (phone numbers, users, settings) |
| **Contact Flows** | Define the customer experience (IVR menus, routing, Lex integration) |
| **Queues** | Hold contacts waiting for agents |
| **Routing Profiles** | Define which queues an agent handles and priority |
| **Phone Numbers** | Toll-free or DID numbers claimed for your instance |
| **Hours of Operation** | Define when queues are active |
| **Quick Connects** | Speed-dial entries for transfers |
| **Agent Hierarchy** | Organizational structure for reporting |

## 1.5 Pricing Model

| Channel | Cost |
|---------|------|
| Voice (inbound) | ~$0.018/min |
| Voice (outbound) | ~$0.018/min + telephony |
| Chat | ~$0.004/message |
| Tasks | ~$0.04/task |
| Contact Lens | Additional per-minute charge |
| Voice ID | Per-transaction |

## 1.6 Personas Served

- **Customers**: Reach support via any channel
- **Agents**: Handle interactions with AI assistance
- **Supervisors**: Monitor metrics, coach agents
- **Administrators**: Configure flows, routing, integrations

---

# PART 2: Amazon Lex

## 2.1 What is Amazon Lex V2?

Amazon Lex V2 is a fully managed service for building conversational interfaces (chatbots) using voice and text. It provides Natural Language Understanding (NLU) and Automatic Speech Recognition (ASR) powered by the same technology as Alexa.

**Key Principle**: No deep learning expertise needed. Define conversation flow, Lex handles the AI.

## 2.2 Core Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Bot** | The conversational application | "BankingBot" |
| **Intent** | What the user wants to do | "CheckBalance", "TransferFunds" |
| **Utterance** | What the user says to trigger an intent | "What's my balance?", "Check my account" |
| **Slot** | Information the bot needs to collect | AccountNumber, TransferAmount |
| **Slot Type** | Data type for a slot | AMAZON.Number, AMAZON.Date, Custom |
| **Fulfillment** | Action taken after slots are filled | Lambda function call |
| **Prompt** | Question the bot asks to fill a slot | "What's your account number?" |
| **Confirmation** | Verify before fulfilling | "Transfer $500 to savings. Correct?" |
| **Fallback Intent** | When the bot doesn't understand | "Sorry, I didn't get that. Can you rephrase?" |

## 2.3 How It Works

```
User Says: "I want to check my balance"
        ↓
ASR (Speech → Text) [if voice]
        ↓
NLU Engine:
    → Intent Classification: "CheckBalance" (confidence: 95%)
    → Slot Detection: None filled yet
        ↓
Bot Prompts: "What's your account number?"
        ↓
User Says: "12345"
        ↓
Slot Filled: AccountNumber = 12345
        ↓
All Required Slots Filled? → YES
        ↓
Fulfillment: Lambda function → Query database → Return balance
        ↓
Bot Responds: "Your account balance is $1,234.56"
```

## 2.4 Key Features

| Feature | Description |
|---------|-------------|
| **Assisted NLU** | LLM-powered intent classification (less training data needed) |
| **Conditional Branching** | Complex conversation flows without Lambda code |
| **Multi-Region Replication** | Deploy bots across regions for DR |
| **Custom Vocabularies** | Improve ASR for domain-specific terms |
| **Session Attributes** | Persist data across turns in a conversation |
| **Context Management** | Control conversation flow based on prior intents |
| **Lambda Integration** | Execute business logic during fulfillment |
| **Automated Chatbot Designer** | Analyze transcripts to auto-suggest intents/slots |
| **Multi-language** | 17+ languages supported |
| **DTMF Support** | Accept keypad input (press 1 for...) |

## 2.5 Integration Points

| Service | Integration |
|---------|-------------|
| **Amazon Connect** | Voice IVR + Chat bot in contact flows |
| **AWS Lambda** | Business logic, database queries, API calls |
| **Amazon Kendra** | Search-based FAQ answers |
| **Amazon Comprehend** | Sentiment analysis |
| **Slack** | Deploy bot to Slack workspace |
| **Facebook Messenger** | Deploy bot to Messenger |
| **Microsoft Teams** | Deploy bot to Teams |
| **WhatsApp** | Deploy bot via messaging |
| **Web/Mobile Apps** | SDK integration |

## 2.6 Use Cases

- Customer support chatbots
- E-commerce shopping assistants
- Appointment booking
- IT helpdesk automation
- Banking (balance, transfers)
- HR self-service (PTO, benefits)
- Insurance claims intake

## 2.7 Pricing

- **Text requests**: ~$0.00075 per request
- **Speech requests**: ~$0.004 per request
- **Free tier**: 10,000 text + 5,000 speech requests/month (first year)

---

# PART 3: Combined Use Case — Amazon Connect + Amazon Lex

## 3.1 The User Story

### 📖 "TechFlow Solutions — Automating Customer Support"

**Company**: TechFlow Solutions — a SaaS company with 50,000 customers
**Problem**: Their support team of 30 agents handles 2,000 calls/day. 60% of calls are simple questions (account balance, password reset, order status) that don't need a human. Agents are overwhelmed, wait times are 15+ minutes, and customer satisfaction is dropping.

**Goal**: Automate 60% of calls with an AI bot, reduce wait times to under 2 minutes, and let agents focus on complex issues.

---

### 📖 Chapter 1: "We need a phone system" → Amazon Connect

> *Sarah (CTO): "We're paying $50,000/month for our legacy Avaya system. We need something cloud-native that scales."*

**Solution**: Set up Amazon Connect as the contact center.

```
Step 1: Create Amazon Connect Instance
Step 2: Claim phone number (toll-free: 1-800-TECHFLOW)
Step 3: Create queues: "General Support", "Billing", "Technical"
Step 4: Create routing profiles for agents
Step 5: Design contact flow (IVR)
```

---

### 📖 Chapter 2: "Customers hate pressing buttons" → Amazon Lex

> *Mike (Support Manager): "Our old IVR says 'Press 1 for billing, press 2 for technical...' — customers hate it. They just want to SAY what they need."*

**Solution**: Build an Amazon Lex bot that understands natural language.

**Bot Name**: TechFlowSupportBot

**Intents Designed**:

| Intent | Sample Utterances | Slots | Fulfillment |
|--------|-------------------|-------|-------------|
| CheckAccountStatus | "What's my account status?", "Is my subscription active?" | AccountEmail | Lambda → Query DB |
| ResetPassword | "I forgot my password", "Reset my login" | AccountEmail | Lambda → Send reset email |
| CheckOrderStatus | "Where's my order?", "Track order 12345" | OrderNumber | Lambda → Query orders API |
| BillingInquiry | "What's my bill?", "When is payment due?" | AccountEmail | Lambda → Query billing |
| SpeakToAgent | "Talk to a human", "I need help", "Agent" | None | Transfer to Connect queue |
| CancelSubscription | "Cancel my account", "I want to cancel" | AccountEmail, Reason | Lambda → Create ticket + Transfer |

---

### 📖 Chapter 3: "Connect the bot to our phone system" → Integration

> *Dev Team: "How do we make the Lex bot answer phone calls?"*

**Solution**: Add the Lex bot to the Amazon Connect contact flow.

**Contact Flow Design**:

```
[Customer Calls 1-800-TECHFLOW]
        ↓
[Play Welcome Message]
"Welcome to TechFlow Solutions. How can I help you today?"
        ↓
[Get Customer Input — Amazon Lex Bot]
Bot: TechFlowSupportBot
        ↓
[Check Intent Result]
    ├── Intent: CheckAccountStatus → [Lambda: Query DB] → [Play Result] → [End]
    ├── Intent: ResetPassword → [Lambda: Send Email] → [Play Confirmation] → [End]
    ├── Intent: CheckOrderStatus → [Lambda: Query API] → [Play Status] → [End]
    ├── Intent: BillingInquiry → [Lambda: Get Bill] → [Play Amount] → [End]
    ├── Intent: SpeakToAgent → [Transfer to Queue: General Support]
    ├── Intent: CancelSubscription → [Transfer to Queue: Retention Team]
    └── FallbackIntent → [Play: "Sorry, I didn't understand"] → [Retry or Transfer]
```

---

### 📖 Chapter 4: "We need it to work on chat too" → Omnichannel

> *Sarah: "Customers also want to chat on our website. Can the same bot work there?"*

**Solution**: Same Lex bot works on both voice (Connect) and chat (Connect Chat Widget).

```javascript
// Website chat widget integration
amazon_connect('chatWidget', {
    instanceId: '<connect-instance-id>',
    contactFlowId: '<contact-flow-id>',
    region: 'us-east-1',
    name: 'Customer',
    // Same Lex bot handles chat conversations!
});
```

---

### 📖 Chapter 5: "How do we handle the business logic?" → Lambda

> *Dev Team: "When someone asks for their order status, how does the bot actually look it up?"*

**Solution**: Lambda function connected to Lex fulfillment.

```python
# lambda_function.py — Lex Fulfillment Handler
import json
import boto3

dynamodb = boto3.resource('dynamodb')
orders_table = dynamodb.Table('Orders')

def lambda_handler(event, context):
    intent_name = event['sessionState']['intent']['name']
    slots = event['sessionState']['intent']['slots']
    
    if intent_name == 'CheckOrderStatus':
        order_number = slots['OrderNumber']['value']['interpretedValue']
        
        # Query DynamoDB
        response = orders_table.get_item(Key={'order_id': order_number})
        
        if 'Item' in response:
            order = response['Item']
            message = f"Your order {order_number} is currently {order['status']}. "
            if order['status'] == 'shipped':
                message += f"Tracking number: {order['tracking_id']}. Expected delivery: {order['delivery_date']}."
        else:
            message = f"I couldn't find order {order_number}. Please check the number and try again."
        
        return {
            'sessionState': {
                'dialogAction': {'type': 'Close'},
                'intent': {
                    'name': intent_name,
                    'state': 'Fulfilled'
                }
            },
            'messages': [{'contentType': 'PlainText', 'content': message}]
        }
    
    elif intent_name == 'ResetPassword':
        email = slots['AccountEmail']['value']['interpretedValue']
        
        # Send password reset email via SES
        ses = boto3.client('ses')
        ses.send_email(
            Source='support@techflow.com',
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {'Data': 'Password Reset Request'},
                'Body': {'Text': {'Data': f'Click here to reset: https://techflow.com/reset?token=...'}}
            }
        )
        
        return {
            'sessionState': {
                'dialogAction': {'type': 'Close'},
                'intent': {'name': intent_name, 'state': 'Fulfilled'}
            },
            'messages': [{'contentType': 'PlainText', 'content': f"I've sent a password reset link to {email}. Please check your inbox."}]
        }
```

---

### 📖 Chapter 6: "We need analytics" → Contact Lens

> *Mike: "How do I know if the bot is working? What about the calls that go to agents?"*

**Solution**: Enable Contact Lens for real-time analytics.

- **Bot Performance**: Track intent match rate, fallback rate, containment rate
- **Agent Calls**: Sentiment analysis, compliance detection, auto-summarization
- **Dashboards**: Real-time queue metrics, agent performance, customer satisfaction

---

### 📖 Chapter 7: Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Calls handled by bot | 0% | 62% | +62% automation |
| Average wait time | 15 min | 1.5 min | -90% |
| Agent handle time | 8 min | 5 min | -37% (complex only) |
| Customer satisfaction | 3.2/5 | 4.5/5 | +40% |
| Monthly cost | $50,000 | $12,000 | -76% |
| Agents needed for simple queries | 18 | 0 | Freed for complex work |

---

## 3.2 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CUSTOMER CHANNELS                                │
│   📞 Phone    💬 Chat Widget    📧 Email    📱 SMS    🌐 Web App        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                        AMAZON CONNECT                                     │
│                                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    │
│  │ Phone Number │───→│  Contact Flow     │───→│  Queue + Routing    │    │
│  │ (Toll-free)  │    │  (IVR Logic)      │    │  Profile            │    │
│  └─────────────┘    └────────┬─────────┘    └──────────┬──────────┘    │
│                              │                          │               │
│                              ↓                          ↓               │
│                    ┌──────────────────┐      ┌─────────────────┐       │
│                    │  AMAZON LEX BOT   │      │  AGENT WORKSPACE │       │
│                    │  (NLU + Dialog)   │      │  (CCP + Profiles)│       │
│                    └────────┬─────────┘      └─────────────────┘       │
│                              │                                          │
│                              ↓                                          │
│                    ┌──────────────────┐                                  │
│                    │  AWS LAMBDA       │                                  │
│                    │  (Business Logic) │                                  │
│                    └────────┬─────────┘                                  │
│                              │                                          │
│              ┌───────────────┼───────────────┐                          │
│              ↓               ↓               ↓                          │
│     ┌──────────────┐ ┌────────────┐ ┌──────────────┐                   │
│     │  DynamoDB     │ │  SES       │ │  External    │                   │
│     │  (Orders/     │ │  (Emails)  │ │  APIs        │                   │
│     │   Accounts)   │ │            │ │  (CRM, ERP)  │                   │
│     └──────────────┘ └────────────┘ └──────────────┘                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CONTACT LENS: Sentiment | Compliance | Transcription | Trends   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3.3 Step-by-Step Implementation Guide

### Step 1: Create Amazon Connect Instance

```bash
# Via AWS Console: Amazon Connect → Create Instance
# Configure:
# - Identity: Store users in Connect
# - Admin: Create admin user
# - Telephony: Enable inbound + outbound
# - Data storage: S3 bucket for recordings
# - Review and create
```

### Step 2: Create Amazon Lex Bot

```python
import boto3

lex = boto3.client('lexv2-models', region_name='us-east-1')

# Create bot
bot = lex.create_bot(
    botName='TechFlowSupportBot',
    description='Customer support bot for TechFlow Solutions',
    roleArn='arn:aws:iam::<account>:role/LexBotRole',
    dataPrivacy={'childDirected': False},
    idleSessionTTLInSeconds=300
)

bot_id = bot['botId']

# Create bot locale (English)
lex.create_bot_locale(
    botId=bot_id,
    botVersion='DRAFT',
    localeId='en_US',
    nluIntentConfidenceThreshold=0.7,
    voiceSettings={'voiceId': 'Joanna'}
)

# Create CheckOrderStatus intent
lex.create_intent(
    botId=bot_id,
    botVersion='DRAFT',
    localeId='en_US',
    intentName='CheckOrderStatus',
    sampleUtterances=[
        {'utterance': 'Where is my order'},
        {'utterance': 'Track order {OrderNumber}'},
        {'utterance': 'What is the status of order {OrderNumber}'},
        {'utterance': 'I want to check my order'},
        {'utterance': 'Order status'}
    ]
)
```

### Step 3: Add Lex Bot to Connect

1. In Amazon Connect console → Contact Flows → "Get customer input" block
2. Select "Amazon Lex" as input type
3. Choose bot: TechFlowSupportBot
4. Map intents to flow branches

### Step 4: Deploy and Test

```bash
# Build the Lex bot
aws lexv2-models build-bot-locale \
    --bot-id <bot-id> \
    --bot-version DRAFT \
    --locale-id en_US

# Create bot version + alias
aws lexv2-models create-bot-version --bot-id <bot-id>
aws lexv2-models create-bot-alias \
    --bot-id <bot-id> \
    --bot-alias-name Production

# Test by calling the phone number!
```

---

## 3.4 Feature Comparison: Connect vs Lex

| Aspect | Amazon Connect | Amazon Lex |
|--------|---------------|------------|
| **Primary Role** | Contact center platform (infrastructure) | Conversational AI (intelligence) |
| **Handles** | Routing, queuing, agent management, telephony | Understanding language, managing dialog |
| **Channels** | Voice, chat, email, SMS, tasks | Voice + text (embedded in other services) |
| **Without the other** | Works (but with old-style DTMF IVR) | Works (in websites, apps, Slack, etc.) |
| **Together** | Natural language IVR + intelligent automation | Voice-enabled bot in a full contact center |
| **Pricing** | Per-minute/message | Per-request |
| **AI Features** | Contact Lens, Voice ID, Q in Connect | NLU, ASR, Assisted NLU (LLM) |

---

## 3.5 When to Use What

| Scenario | Use |
|----------|-----|
| Full contact center with agents | Amazon Connect |
| Chatbot on website/app only | Amazon Lex (standalone) |
| Voice IVR with natural language | Connect + Lex |
| Automate simple phone inquiries | Connect + Lex + Lambda |
| Agent assistance during calls | Connect + Amazon Q in Connect |
| Fraud detection on calls | Connect + Voice ID |
| Chat on website + phone support | Connect (omnichannel) + Lex |
| Slack/Teams bot for internal IT | Amazon Lex (standalone) |

---

## 4. Conclusion

Amazon Connect and Amazon Lex are complementary services that together create a powerful AI-driven customer experience platform:

- **Connect** = The contact center infrastructure (phones, queues, agents, analytics)
- **Lex** = The conversational brain (understands language, manages dialog, collects information)
- **Lambda** = The business logic glue (queries databases, calls APIs, sends emails)
- **Contact Lens** = The analytics layer (sentiment, compliance, trends)

The TechFlow story demonstrates how a company can reduce costs by 76%, automate 62% of interactions, and improve customer satisfaction by 40% — all by combining these services.

---

## References

1. What is Amazon Connect: https://docs.aws.amazon.com/connect/latest/adminguide/what-is-amazon-connect.html
2. Connect Feature Overview: https://docs.aws.amazon.com/connect/latest/adminguide/connect-feature-overview.html
3. Add Lex Bot to Connect: https://docs.aws.amazon.com/connect/latest/adminguide/amazon-lex.html
4. What is Amazon Lex V2: https://docs.aws.amazon.com/lexv2/latest/dg/what-is.html
5. Lex Intents: https://docs.aws.amazon.com/lexv2/latest/dg/add-intents.html
6. Lex Lambda Integration: https://docs.aws.amazon.com/lexv2/latest/dg/lambda.html
7. Connect Pricing: https://aws.amazon.com/connect/pricing/
8. Lex Pricing: https://aws.amazon.com/lex/pricing/

---

*Paper generated based on AWS official documentation. Features and pricing are subject to change.*
