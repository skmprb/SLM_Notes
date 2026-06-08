# Research Paper: Amazon Bedrock AgentCore — Examples & Configuration Guide

## Abstract

This paper provides hands-on examples and detailed configuration guides for Amazon Bedrock AgentCore's 12 core services. It covers deployment patterns, API usage, SDK integration, IAM setup, and real-world implementation scenarios using Python (Boto3), CLI, and the Strands Agents framework.

---

## 1. Getting Started

### 1.1 Prerequisites

- AWS Account with AgentCore access enabled
- AWS CLI v2 configured
- Python 3.9+ with Boto3
- IAM permissions for `bedrock-agentcore:*` actions

### 1.2 Install Dependencies

```bash
pip install boto3 strands-agents strands-agents-tools
pip install amazon-bedrock-agentcore-sdk
```

### 1.3 AgentCore Endpoints

| Service | Endpoint Format |
|---------|----------------|
| Control Plane | `bedrock-agentcore-control.{region}.amazonaws.com` |
| Runtime | `bedrock-agentcore-runtime.{region}.amazonaws.com` |
| Gateway | `https://{gateway-Id}.gateway.bedrock-agentcore.{Region}.amazonaws.com` |

---

## 2. AgentCore Runtime — Deploy an Agent

### 2.1 Deploy with Strands Agents Framework

```python
# agent.py - Your agent code
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, web_search

model = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful research assistant.",
    tools=[calculator, web_search]
)

# For local testing
if __name__ == "__main__":
    result = agent("What is 25 * 47?")
    print(result)
```

### 2.2 Deploy to AgentCore Runtime via CLI

```bash
# Initialize AgentCore project
agentcore init my-agent

# Deploy the agent
agentcore deploy --name my-research-agent \
    --entry-point agent.py \
    --region us-east-1

# Create an endpoint (alias)
aws bedrock-agentcore create-agent-runtime-endpoint \
    --agent-runtime-id <agent-id> \
    --name production \
    --description "Production endpoint"
```

### 2.3 Deploy with Container Image

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "agent.py"]
```

```bash
# Build and push to ECR
docker build -t my-agent .
docker tag my-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest

# Deploy container to AgentCore Runtime
aws bedrock-agentcore create-agent-runtime \
    --name my-container-agent \
    --runtime-configuration '{
        "containerConfiguration": {
            "imageUri": "<account>.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest"
        }
    }'
```

### 2.4 Invoke an Agent

```python
import boto3
import json

client = boto3.client('bedrock-agentcore-runtime', region_name='us-east-1')

response = client.invoke_agent_runtime(
    agentRuntimeId='<agent-id>',
    endpointId='<endpoint-id>',
    sessionId='user-session-123',
    payload=json.dumps({
        "message": "Analyze the quarterly sales data and create a summary report"
    }).encode()
)

# Read response
result = response['body'].read().decode()
print(result)
```

### 2.5 Long-Running Async Agent

```python
# Start async job (up to 8 hours)
response = client.invoke_agent_runtime(
    agentRuntimeId='<agent-id>',
    endpointId='<endpoint-id>',
    sessionId='long-task-001',
    invocationType='ASYNC',
    payload=json.dumps({
        "message": "Perform deep research on market trends for Q4 2025"
    }).encode()
)

job_id = response['jobId']

# Check status
status = client.get_agent_runtime_job(
    agentRuntimeId='<agent-id>',
    jobId=job_id
)
print(f"Status: {status['status']}")
```

### 2.6 WebSocket Bidirectional Streaming

```python
import websocket
import json

ws_url = "wss://<endpoint-url>/ws?sessionId=stream-session-001"

ws = websocket.WebSocket()
ws.connect(ws_url)

# Send message
ws.send(json.dumps({"message": "Stream a detailed analysis"}))

# Receive streaming response
while True:
    result = ws.recv()
    if not result:
        break
    print(json.loads(result)['chunk'], end='', flush=True)

ws.close()
```

---

## 3. AgentCore Harness — Single API Agent

### 3.1 Invoke with Inline Tools

```python
import boto3
import json

client = boto3.client('bedrock-agentcore', region_name='us-east-1')

response = client.invoke_harness(
    modelId="anthropic.claude-sonnet-4-20250514-v1:0",
    systemPrompt="You are a data analyst. Use tools to answer questions accurately.",
    tools=[
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    ],
    messages=[
        {"role": "user", "content": "What is the compound interest on $10000 at 5% for 3 years?"}
    ]
)

print(response['output']['content'])
```

### 3.2 Harness with MCP Server

```python
response = client.invoke_harness(
    modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
    systemPrompt="You are a project manager assistant.",
    mcpServers=[
        {
            "uri": "https://<gateway-id>.gateway.bedrock-agentcore.us-east-1.amazonaws.com/jira-tools"
        }
    ],
    messages=[
        {"role": "user", "content": "Create a new JIRA ticket for the login bug"}
    ]
)
```

---

## 4. AgentCore Memory — Configuration

### 4.1 Create a Memory Resource

```python
import boto3

client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Create memory with short-term and long-term strategies
response = client.create_memory(
    name='customer-support-memory',
    description='Memory for customer support agents',
    memoryStrategies=[
        {
            'strategyName': 'conversation-history',
            'type': 'SHORT_TERM',
            'configuration': {
                'shortTermConfiguration': {
                    'maxMessages': 50
                }
            }
        },
        {
            'strategyName': 'user-preferences',
            'type': 'LONG_TERM',
            'configuration': {
                'longTermConfiguration': {
                    'extractionPrompt': 'Extract user preferences, past issues, and account details.',
                    'consolidationPrompt': 'Merge duplicate information and keep the most recent.'
                }
            }
        }
    ]
)

memory_id = response['memoryId']
print(f"Memory created: {memory_id}")
```

### 4.2 Store Events (Conversations)

```python
# Store a conversation event
client.create_event(
    memoryId=memory_id,
    actorId='user-456',
    sessionId='session-789',
    messages=[
        {'role': 'user', 'content': 'I prefer email notifications over SMS'},
        {'role': 'assistant', 'content': 'Noted! I have updated your preference to email notifications.'}
    ],
    eventExpirationDuration=365  # days
)
```

### 4.3 Retrieve Memory Records

```python
# Retrieve relevant memories for context
records = client.retrieve_memory_records(
    memoryId=memory_id,
    actorId='user-456',
    query='What are this user preferences?',
    maxResults=5
)

for record in records['memoryRecords']:
    print(f"Memory: {record['content']}")
    print(f"Score: {record['score']}")
```

### 4.4 Integrate Memory with Agent

```python
from strands import Agent
from strands.models import BedrockModel
from strands.memory import AgentCoreMemory

model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

memory = AgentCoreMemory(
    memory_id='<memory-id>',
    region_name='us-east-1'
)

agent = Agent(
    model=model,
    memory=memory,
    system_prompt="You are a personalized assistant. Use memory to provide context-aware responses."
)

response = agent("What were my previous preferences?")
```

---

## 5. AgentCore Gateway — Convert APIs to MCP Tools

### 5.1 Create a Gateway

```python
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Create gateway
response = client.create_gateway(
    name='enterprise-tools-gateway',
    description='Gateway for enterprise tool access',
    protocolType='MCP'
)

gateway_id = response['gatewayId']
```

### 5.2 Add a Lambda Function as MCP Tool

```python
# Add Lambda target to gateway
client.create_gateway_target(
    gatewayId=gateway_id,
    name='order-lookup',
    description='Look up customer orders',
    targetConfiguration={
        'lambdaConfiguration': {
            'lambdaArn': 'arn:aws:lambda:us-east-1:<account>:function:order-lookup'
        }
    }
)
```

### 5.3 Add an Existing MCP Server

```python
# Connect to existing MCP server
client.create_gateway_target(
    gatewayId=gateway_id,
    name='github-tools',
    description='GitHub MCP server for code operations',
    targetConfiguration={
        'mcpConfiguration': {
            'uri': 'https://mcp-server.example.com/github',
            'authorizationType': 'OAUTH2'
        }
    }
)
```

### 5.4 Add AgentCore Runtime as Target

```python
# Route to an AgentCore Runtime agent
client.create_gateway_target(
    gatewayId=gateway_id,
    name='data-analyst-agent',
    description='Specialized data analysis agent',
    targetConfiguration={
        'agentCoreRuntimeConfiguration': {
            'agentRuntimeId': '<agent-runtime-id>',
            'endpointId': '<endpoint-id>'
        }
    }
)
```

### 5.5 Use Gateway in Agent

```python
from strands import Agent
from strands.models import BedrockModel
from strands.tools import MCPClient

model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

# Connect to gateway as MCP client
mcp_tools = MCPClient(
    uri="https://<gateway-id>.gateway.bedrock-agentcore.us-east-1.amazonaws.com"
)

agent = Agent(
    model=model,
    tools=[mcp_tools],
    system_prompt="You can look up orders and manage GitHub repositories."
)

response = agent("Look up order #12345 and create a GitHub issue for the delay")
```

---

## 6. AgentCore Identity — Authentication Setup

### 6.1 Create a Workload Identity

```python
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Create identity for an agent
response = client.create_workload_identity(
    name='support-agent-identity',
    description='Identity for customer support agent',
    allowedAudiences=['https://api.example.com'],
    issuerUrl='https://cognito-idp.us-east-1.amazonaws.com/<user-pool-id>'
)

identity_id = response['workloadIdentityId']
```

### 6.2 Configure OAuth2 Credential Provider (Outbound)

```python
# Enable agent to access Slack on behalf of users
client.create_oauth2_credential_provider(
    name='slack-oauth',
    credentialProviderConfiguration={
        'oAuth2Configuration': {
            'authorizationEndpoint': 'https://slack.com/oauth/v2/authorize',
            'tokenEndpoint': 'https://slack.com/api/oauth.v2.access',
            'clientId': '<slack-client-id>',
            'clientSecret': '<slack-client-secret>',
            'scopes': ['chat:write', 'channels:read']
        }
    }
)
```

### 6.3 Configure Inbound Authentication

```python
# Configure JWT authorizer for end-user authentication
client.create_jwt_authorizer(
    name='user-auth',
    issuerUrl='https://login.microsoftonline.com/<tenant-id>/v2.0',
    audiences=['api://<app-id>'],
    claimsMapping={
        'userId': 'sub',
        'email': 'email'
    }
)
```

---

## 7. AgentCore Code Interpreter — Execute Code

### 7.1 Create Code Interpreter

```bash
aws bedrock-agentcore create-code-interpreter \
    --name my-code-interpreter \
    --runtime-environment PYTHON_3_11
```

### 7.2 Execute Code Directly (Boto3)

```python
import boto3

client = boto3.client('bedrock-agentcore-runtime', region_name='us-east-1')

# Start a session
session = client.create_code_interpreter_session(
    codeInterpreterId='<interpreter-id>'
)

session_id = session['sessionId']

# Execute Python code
result = client.execute_code(
    codeInterpreterId='<interpreter-id>',
    sessionId=session_id,
    code="""
import pandas as pd
import numpy as np

# Generate sample data
data = pd.DataFrame({
    'month': pd.date_range('2025-01', periods=12, freq='M'),
    'revenue': np.random.randint(50000, 150000, 12),
    'costs': np.random.randint(30000, 80000, 12)
})

data['profit'] = data['revenue'] - data['costs']
print(data.to_string())
print(f"\\nTotal Profit: ${data['profit'].sum():,.2f}")
""",
    runtime='python'
)

print(result['output'])
```

### 7.3 Use with Strands Agent

```python
from strands import Agent
from strands.models import BedrockModel
from strands_tools import code_interpreter

model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

agent = Agent(
    model=model,
    tools=[code_interpreter],
    system_prompt="You are a data scientist. Write and execute code to answer questions."
)

response = agent("Calculate the Fibonacci sequence up to 100 and plot it")
```

---

## 8. AgentCore Browser — Web Automation

### 8.1 Create Browser Session

```python
client = boto3.client('bedrock-agentcore-runtime', region_name='us-east-1')

# Create browser session
session = client.create_browser_session(
    configuration={
        'viewport': {'width': 1920, 'height': 1080},
        'proxy': None,
        'extensions': []
    }
)

browser_session_id = session['sessionId']
```

### 8.2 Use Browser with Agent (Playwright)

```python
from strands import Agent
from strands.models import BedrockModel
from strands_tools import browser_tool

model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

agent = Agent(
    model=model,
    tools=[browser_tool],
    system_prompt="""You are a web research agent. 
    Use the browser to navigate websites, extract information, and fill forms."""
)

response = agent("Go to news.ycombinator.com and summarize the top 5 stories")
```

---

## 9. AgentCore Policy — Governance Configuration

### 9.1 Create Policy (Natural Language)

```python
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

response = client.create_policy(
    name='agent-boundaries',
    description='Boundaries for customer support agents',
    policyDocument="""
    Rules:
    - Agents can only access customer data for the authenticated user
    - Agents cannot delete any records
    - Agents cannot access financial data without manager approval
    - Agents can only send emails to verified addresses
    - Maximum 10 API calls per minute per tool
    """
)
```

### 9.2 Create Policy (Cedar Language)

```cedar
// Cedar policy for agent tool access
permit(
    principal == AgentCore::Agent::"support-agent",
    action == AgentCore::Action::"invoke-tool",
    resource == AgentCore::Tool::"order-lookup"
) when {
    context.user.authenticated == true &&
    context.user.department == "support"
};

forbid(
    principal == AgentCore::Agent::"support-agent",
    action == AgentCore::Action::"invoke-tool",
    resource == AgentCore::Tool::"delete-record"
);
```

### 9.3 Attach Policy to Gateway

```python
client.attach_policy(
    policyId='<policy-id>',
    gatewayId='<gateway-id>'
)
```

---

## 10. AgentCore Observability — Monitoring Setup

### 10.1 Enable Observability

```python
# Enable observability for an agent runtime
client.update_agent_runtime(
    agentRuntimeId='<agent-id>',
    observabilityConfiguration={
        'enabled': True,
        'logDestination': {
            'cloudWatchLogs': {
                'logGroupArn': 'arn:aws:logs:us-east-1:<account>:log-group:/agentcore/agents'
            }
        },
        'traceConfiguration': {
            'enabled': True,
            'samplingRate': 1.0  # 100% sampling
        }
    }
)
```

### 10.2 Instrument Agent Code (ADOT SDK)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup OTEL tracing
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-agent")

# Instrument agent steps
with tracer.start_as_current_span("agent-reasoning") as span:
    span.set_attribute("agent.step", "orchestration")
    span.set_attribute("model.id", "claude-sonnet-4")
    # ... agent logic
```

### 10.3 View in CloudWatch

Navigate to CloudWatch → Transaction Search → Filter by:
- Service: `bedrock-agentcore`
- Agent: `<agent-name>`
- Time range: Last 1 hour

---

## 11. AgentCore Evaluations — Quality Assessment

### 11.1 Run Evaluation

```python
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

response = client.create_evaluation(
    name='support-agent-eval-v2',
    agentRuntimeId='<agent-id>',
    evaluationConfiguration={
        'dataSource': {
            'type': 'TRACES',
            'traceFilter': {
                'timeRange': {
                    'start': '2025-07-01T00:00:00Z',
                    'end': '2025-07-15T00:00:00Z'
                }
            }
        },
        'metrics': [
            'TASK_COMPLETION',
            'TOOL_USAGE_ACCURACY',
            'RESPONSE_QUALITY',
            'EDGE_CASE_HANDLING'
        ]
    }
)

eval_id = response['evaluationId']
```

### 11.2 Get Evaluation Results

```python
results = client.get_evaluation(evaluationId=eval_id)

print(f"Task Completion: {results['metrics']['taskCompletion']}%")
print(f"Tool Accuracy: {results['metrics']['toolUsageAccuracy']}%")
print(f"Response Quality: {results['metrics']['responseQuality']}/5")
```

---

## 12. AgentCore Registry — Publish & Discover

### 12.1 Publish an Agent to Registry

```python
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

response = client.publish_resource(
    resourceType='AGENT',
    name='customer-support-agent',
    description='Handles customer inquiries, order lookups, and issue resolution',
    metadata={
        'version': '2.1.0',
        'framework': 'strands',
        'model': 'claude-sonnet-4',
        'capabilities': ['order-lookup', 'ticket-creation', 'email-sending'],
        'department': 'support',
        'owner': 'platform-team'
    },
    agentRuntimeId='<agent-runtime-id>'
)
```

### 12.2 Discover Resources

```python
# Search for tools related to "customer orders"
results = client.search_registry(
    query='customer orders',
    resourceTypes=['AGENT', 'MCP_SERVER', 'TOOL'],
    maxResults=10
)

for resource in results['resources']:
    print(f"{resource['resourceType']}: {resource['name']}")
    print(f"  Description: {resource['description']}")
    print(f"  Score: {resource['relevanceScore']}")
```

---

## 13. IAM Permissions — Minimal Policy

### 13.1 Agent Developer Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:CreateAgentRuntime",
                "bedrock-agentcore:UpdateAgentRuntime",
                "bedrock-agentcore:DeleteAgentRuntime",
                "bedrock-agentcore:InvokeAgentRuntime",
                "bedrock-agentcore:CreateAgentRuntimeEndpoint",
                "bedrock-agentcore:GetAgentRuntime",
                "bedrock-agentcore:ListAgentRuntimes"
            ],
            "Resource": "arn:aws:bedrock-agentcore:*:*:agent-runtime/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:CreateMemory",
                "bedrock-agentcore:CreateEvent",
                "bedrock-agentcore:RetrieveMemoryRecords"
            ],
            "Resource": "arn:aws:bedrock-agentcore:*:*:memory/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:InvokeGateway"
            ],
            "Resource": "arn:aws:bedrock-agentcore:*:*:gateway/*"
        }
    ]
}
```

### 13.2 AgentCore Runtime Service Role Trust Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock-agentcore.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

---

## 14. End-to-End Example: Customer Support Agent

### 14.1 Architecture

```
Customer → AgentCore Identity (auth) → AgentCore Runtime (agent)
                                              ↓
                                    AgentCore Memory (context)
                                              ↓
                                    AgentCore Gateway → [Order API, Ticket System, Email]
                                              ↓
                                    AgentCore Policy (enforce rules)
                                              ↓
                                    AgentCore Observability (monitor)
```

### 14.2 Complete Agent Code

```python
from strands import Agent
from strands.models import BedrockModel
from strands.memory import AgentCoreMemory
from strands.tools import MCPClient

# Model
model = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

# Memory
memory = AgentCoreMemory(
    memory_id="customer-support-memory",
    region_name="us-east-1"
)

# Tools via Gateway
tools = MCPClient(
    uri="https://<gateway-id>.gateway.bedrock-agentcore.us-east-1.amazonaws.com"
)

# Agent
agent = Agent(
    model=model,
    memory=memory,
    tools=[tools],
    system_prompt="""You are a customer support agent for an e-commerce company.
    
    You can:
    - Look up customer orders
    - Create support tickets
    - Send email notifications
    - Check product availability
    
    Always:
    - Verify the customer's identity before sharing order details
    - Be empathetic and professional
    - Escalate to a human if the issue is complex
    - Use memory to recall previous interactions
    """
)

# Handle customer request
response = agent(
    "Hi, I ordered a laptop last week (order #98765) and it hasn't arrived yet. Can you help?"
)
print(response)
```

---

## 15. Conclusion

Amazon Bedrock AgentCore provides a comprehensive, modular platform for deploying production AI agents. The key configuration patterns are:

1. **Runtime**: Deploy any framework agent with container or direct code deployment
2. **Harness**: Single API call for simple agent invocations
3. **Memory**: Short-term + long-term context with framework integrations
4. **Gateway**: Convert any API/Lambda into MCP tools
5. **Identity**: Enterprise SSO + outbound OAuth for third-party access
6. **Code Interpreter**: Sandboxed code execution in Python/JS/TS
7. **Browser**: Web automation with Playwright/BrowserUse
8. **Policy**: Cedar-based governance attached to Gateway
9. **Observability**: OpenTelemetry tracing with CloudWatch integration
10. **Evaluations**: Automated quality assessment from traces
11. **Registry**: Publish/discover agents and tools organization-wide
12. **Payments**: x402 microtransactions for paid APIs

All services are consumption-based with no upfront costs, and can be adopted incrementally based on your needs.

---

## References

1. AgentCore Developer Guide: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/
2. AgentCore Runtime: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
3. AgentCore Code Interpreter: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html
4. AgentCore Getting Started: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html
5. Strands Agents Framework: https://github.com/strands-agents/strands-agents
6. AgentCore Quotas: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html

---

*Paper generated based on AWS official documentation. API signatures and configurations may evolve. Always refer to the latest AWS documentation for production implementations.*
