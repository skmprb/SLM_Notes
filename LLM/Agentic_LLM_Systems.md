# Agentic LLM Systems

## 🎯 Overview

Agentic LLM systems enable autonomous reasoning, planning, and action execution. These systems combine language models with structured approaches to problem-solving, self-reflection, and multi-agent coordination.

## 📋 Planning

### Basic Planning Framework

```python
class LLMPlanner:
    def __init__(self, model):
        self.model = model
        
    def create_plan(self, goal: str, available_actions: List[str]) -> List[Dict]:
        """Create a step-by-step plan to achieve a goal."""
        planning_prompt = f"""
Goal: {goal}

Available actions:
{chr(10).join(f"- {action}" for action in available_actions)}

Create a detailed step-by-step plan to achieve this goal. Format each step as:
Step X: Action - Description

Plan:"""
        
        response = self.model.generate(planning_prompt, max_tokens=400)
        return self.parse_plan(response)
    
    def parse_plan(self, plan_text: str) -> List[Dict]:
        """Parse plan text into structured steps."""
        steps = []
        lines = plan_text.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('Step'):
                # Extract step number, action, and description
                parts = line.split(':', 2)
                if len(parts) >= 2:
                    step_num = parts[0].strip()
                    content = parts[1].strip()
                    
                    # Split action and description
                    if ' - ' in content:
                        action, description = content.split(' - ', 1)
                    else:
                        action = content
                        description = ""
                    
                    steps.append({
                        'step': step_num,
                        'action': action.strip(),
                        'description': description.strip()
                    })
        
        return steps

class HierarchicalPlanner(LLMPlanner):
    def __init__(self, model):
        super().__init__(model)
        
    def create_hierarchical_plan(self, goal: str, max_depth: int = 3) -> Dict:
        """Create a hierarchical plan with sub-goals."""
        return self._plan_recursive(goal, depth=0, max_depth=max_depth)
    
    def _plan_recursive(self, goal: str, depth: int, max_depth: int) -> Dict:
        """Recursively create hierarchical plans."""
        if depth >= max_depth:
            return {'goal': goal, 'type': 'atomic', 'depth': depth}
        
        # Determine if goal should be decomposed
        decomposition_prompt = f"""
Goal: {goal}

Should this goal be broken down into smaller sub-goals? 
If yes, list 2-4 sub-goals. If no, say "ATOMIC".

Response:"""
        
        response = self.model.generate(decomposition_prompt, max_tokens=200)
        
        if 'ATOMIC' in response.upper():
            return {'goal': goal, 'type': 'atomic', 'depth': depth}
        
        # Extract sub-goals
        sub_goals = self.extract_subgoals(response)
        
        # Recursively plan for each sub-goal
        sub_plans = []
        for sub_goal in sub_goals:
            sub_plan = self._plan_recursive(sub_goal, depth + 1, max_depth)
            sub_plans.append(sub_plan)
        
        return {
            'goal': goal,
            'type': 'composite',
            'depth': depth,
            'sub_plans': sub_plans
        }
    
    def extract_subgoals(self, response: str) -> List[str]:
        """Extract sub-goals from response."""
        lines = response.strip().split('\n')
        sub_goals = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Goal:'):
                # Remove numbering and bullet points
                cleaned = line.lstrip('0123456789.- ')
                if cleaned:
                    sub_goals.append(cleaned)
        
        return sub_goals
```

### Dynamic Planning

```python
class DynamicPlanner(LLMPlanner):
    def __init__(self, model):
        super().__init__(model)
        self.execution_history = []
        
    def adaptive_planning(self, goal: str, current_state: Dict, 
                         available_actions: List[str]) -> List[Dict]:
        """Create plan that adapts to current state."""
        context = self.build_context(current_state)
        
        planning_prompt = f"""
Goal: {goal}

Current situation:
{context}

Available actions:
{chr(10).join(f"- {action}" for action in available_actions)}

Previous attempts:
{self.format_execution_history()}

Create an adaptive plan considering the current situation and previous attempts:

Plan:"""
        
        response = self.model.generate(planning_prompt, max_tokens=400)
        return self.parse_plan(response)
    
    def replan(self, original_goal: str, failed_step: Dict, 
              new_state: Dict, available_actions: List[str]) -> List[Dict]:
        """Replan after a failed step."""
        replan_prompt = f"""
Original goal: {original_goal}

Failed step: {failed_step['action']} - {failed_step['description']}
Failure reason: {failed_step.get('failure_reason', 'Unknown')}

Current state after failure:
{self.build_context(new_state)}

Available actions:
{chr(10).join(f"- {action}" for action in available_actions)}

Create a new plan to recover from this failure and achieve the original goal:

Recovery Plan:"""
        
        response = self.model.generate(replan_prompt, max_tokens=400)
        return self.parse_plan(response)
    
    def build_context(self, state: Dict) -> str:
        """Build context string from state dictionary."""
        context_parts = []
        for key, value in state.items():
            context_parts.append(f"{key}: {value}")
        return '\n'.join(context_parts)
    
    def format_execution_history(self) -> str:
        """Format execution history for context."""
        if not self.execution_history:
            return "No previous attempts"
        
        history_parts = []
        for i, attempt in enumerate(self.execution_history[-3:], 1):  # Last 3 attempts
            history_parts.append(f"Attempt {i}: {attempt['action']} - {attempt['result']}")
        
        return '\n'.join(history_parts)
```

## 🪞 Reflection

### Self-Reflection Framework

```python
class ReflectiveAgent:
    def __init__(self, model):
        self.model = model
        self.reflection_history = []
        
    def reflect_on_action(self, action: Dict, result: Dict, goal: str) -> Dict:
        """Reflect on the outcome of an action."""
        reflection_prompt = f"""
Goal: {goal}

Action taken: {action['action']}
Action description: {action.get('description', '')}

Result: {result['outcome']}
Success: {result['success']}

Reflect on this action:
1. Was this action appropriate for the goal?
2. What went well?
3. What could be improved?
4. What should be done differently next time?

Reflection:"""
        
        reflection_response = self.model.generate(reflection_prompt, max_tokens=300)
        
        reflection = {
            'action': action,
            'result': result,
            'goal': goal,
            'reflection_text': reflection_response,
            'timestamp': self.get_timestamp(),
            'lessons_learned': self.extract_lessons(reflection_response)
        }
        
        self.reflection_history.append(reflection)
        return reflection
    
    def extract_lessons(self, reflection_text: str) -> List[str]:
        """Extract key lessons from reflection."""
        lesson_prompt = f"""
From this reflection:
{reflection_text}

Extract 2-3 key lessons learned as bullet points:

Lessons:"""
        
        lessons_response = self.model.generate(lesson_prompt, max_tokens=150)
        
        # Parse lessons
        lessons = []
        for line in lessons_response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                lessons.append(line.lstrip('-• '))
        
        return lessons
    
    def reflect_on_overall_performance(self, goal: str, final_outcome: Dict) -> Dict:
        """Reflect on overall performance for a goal."""
        recent_reflections = self.reflection_history[-10:]  # Last 10 reflections
        
        performance_prompt = f"""
Goal: {goal}
Final outcome: {final_outcome}

Recent action reflections:
{self.format_recent_reflections(recent_reflections)}

Provide an overall performance reflection:
1. How well was the goal achieved?
2. What were the main strengths in the approach?
3. What were the main weaknesses?
4. What patterns do you notice in the actions taken?
5. How would you approach this type of goal differently in the future?

Overall Reflection:"""
        
        overall_reflection = self.model.generate(performance_prompt, max_tokens=400)
        
        return {
            'goal': goal,
            'final_outcome': final_outcome,
            'overall_reflection': overall_reflection,
            'action_count': len(recent_reflections),
            'timestamp': self.get_timestamp()
        }
    
    def format_recent_reflections(self, reflections: List[Dict]) -> str:
        """Format recent reflections for context."""
        if not reflections:
            return "No recent reflections"
        
        formatted = []
        for i, reflection in enumerate(reflections, 1):
            formatted.append(f"{i}. {reflection['action']['action']} -> {reflection['result']['outcome']}")
        
        return '\n'.join(formatted)
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

class MetaCognitiveAgent(ReflectiveAgent):
    def __init__(self, model):
        super().__init__(model)
        
    def meta_reflect(self, recent_goals: List[str], outcomes: List[Dict]) -> Dict:
        """Reflect on patterns across multiple goals."""
        meta_prompt = f"""
Recent goals and outcomes:
{self.format_goals_outcomes(recent_goals, outcomes)}

Meta-cognitive reflection:
1. What patterns do you see in your problem-solving approach?
2. What types of goals do you handle well vs. poorly?
3. What are your consistent strengths and weaknesses?
4. How has your performance changed over time?
5. What fundamental improvements should you make to your approach?

Meta-reflection:"""
        
        meta_reflection = self.model.generate(meta_prompt, max_tokens=500)
        
        return {
            'meta_reflection': meta_reflection,
            'goals_analyzed': len(recent_goals),
            'timestamp': self.get_timestamp()
        }
    
    def format_goals_outcomes(self, goals: List[str], outcomes: List[Dict]) -> str:
        """Format goals and outcomes for meta-reflection."""
        formatted = []
        for goal, outcome in zip(goals, outcomes):
            formatted.append(f"Goal: {goal}")
            formatted.append(f"Outcome: {outcome.get('success', 'Unknown')} - {outcome.get('summary', '')}")
            formatted.append("")
        
        return '\n'.join(formatted)
```

## 🔄 ReAct Pattern

### ReAct Implementation

```python
class ReActAgent:
    def __init__(self, model, tools: Dict[str, callable]):
        self.model = model
        self.tools = tools
        self.max_iterations = 10
        
    def solve(self, problem: str) -> Dict:
        """Solve problem using ReAct (Reasoning + Acting) pattern."""
        conversation = []
        observations = []
        
        # Initial thought
        current_input = f"Problem: {problem}\n\nThought 1:"
        
        for iteration in range(self.max_iterations):
            # Generate thought
            thought = self.model.generate(current_input, max_tokens=200, stop=["Action", "Observation"])
            conversation.append(f"Thought {iteration + 1}: {thought}")
            
            # Generate action
            action_input = current_input + thought + f"\nAction {iteration + 1}:"
            action = self.model.generate(action_input, max_tokens=100, stop=["Observation", "Thought"])
            conversation.append(f"Action {iteration + 1}: {action}")
            
            # Execute action and get observation
            observation = self.execute_action(action.strip())
            observations.append(observation)
            conversation.append(f"Observation {iteration + 1}: {observation}")
            
            # Check if we have a final answer
            if self.is_final_answer(action, observation):
                final_answer = self.extract_final_answer(observation)
                break
            
            # Prepare next iteration
            current_input = "\n".join(conversation) + f"\n\nThought {iteration + 2}:"
        
        else:
            final_answer = "Maximum iterations reached without finding answer"
        
        return {
            'problem': problem,
            'conversation': conversation,
            'observations': observations,
            'final_answer': final_answer,
            'iterations': iteration + 1
        }
    
    def execute_action(self, action: str) -> str:
        """Execute an action and return observation."""
        # Parse action
        action = action.strip()
        
        # Handle different action types
        if action.startswith("Search"):
            query = self.extract_search_query(action)
            return self.tools.get('search', lambda x: f"Search results for: {x}")(query)
        
        elif action.startswith("Calculate"):
            expression = self.extract_calculation(action)
            return self.tools.get('calculate', lambda x: f"Calculation result: {x}")(expression)
        
        elif action.startswith("Think"):
            return "Continuing to think about the problem..."
        
        elif action.startswith("Answer"):
            return self.extract_answer_content(action)
        
        else:
            return f"Unknown action: {action}"
    
    def extract_search_query(self, action: str) -> str:
        """Extract search query from action."""
        # Simple extraction - in practice, use more robust parsing
        if ":" in action:
            return action.split(":", 1)[1].strip()
        return action.replace("Search", "").strip()
    
    def extract_calculation(self, action: str) -> str:
        """Extract calculation from action."""
        if ":" in action:
            return action.split(":", 1)[1].strip()
        return action.replace("Calculate", "").strip()
    
    def extract_answer_content(self, action: str) -> str:
        """Extract answer content from action."""
        if ":" in action:
            return action.split(":", 1)[1].strip()
        return action.replace("Answer", "").strip()
    
    def is_final_answer(self, action: str, observation: str) -> bool:
        """Check if we have reached a final answer."""
        return action.strip().startswith("Answer") or "final answer" in observation.lower()
    
    def extract_final_answer(self, observation: str) -> str:
        """Extract the final answer from observation."""
        return observation.strip()

# Enhanced ReAct with structured actions
class StructuredReActAgent(ReActAgent):
    def __init__(self, model, tools: Dict[str, callable]):
        super().__init__(model, tools)
        self.action_schema = {
            'search': {'description': 'Search for information', 'format': 'Search: <query>'},
            'calculate': {'description': 'Perform calculation', 'format': 'Calculate: <expression>'},
            'analyze': {'description': 'Analyze information', 'format': 'Analyze: <content>'},
            'answer': {'description': 'Provide final answer', 'format': 'Answer: <response>'}
        }
    
    def get_action_prompt(self) -> str:
        """Get formatted action options."""
        actions = []
        for action, info in self.action_schema.items():
            actions.append(f"{info['format']} - {info['description']}")
        
        return "Available actions:\n" + "\n".join(actions)
    
    def solve_structured(self, problem: str) -> Dict:
        """Solve using structured ReAct with explicit action schema."""
        conversation = []
        
        # Initial setup with action schema
        initial_prompt = f"""Problem: {problem}

{self.get_action_prompt()}

Solve this step by step using the Thought -> Action -> Observation pattern.

Thought 1:"""
        
        current_input = initial_prompt
        
        for iteration in range(self.max_iterations):
            # Generate thought
            thought = self.model.generate(current_input, max_tokens=200, stop=["Action"])
            conversation.append(f"Thought {iteration + 1}: {thought}")
            
            # Generate action
            action_prompt = current_input + thought + f"\nAction {iteration + 1}:"
            action = self.model.generate(action_prompt, max_tokens=100, stop=["Observation"])
            conversation.append(f"Action {iteration + 1}: {action}")
            
            # Execute and observe
            observation = self.execute_structured_action(action.strip())
            conversation.append(f"Observation {iteration + 1}: {observation}")
            
            if self.is_final_answer(action, observation):
                break
            
            # Next iteration
            current_input = "\n".join(conversation) + f"\n\nThought {iteration + 2}:"
        
        return {
            'problem': problem,
            'conversation': conversation,
            'final_answer': self.extract_final_answer(observation),
            'iterations': iteration + 1
        }
    
    def execute_structured_action(self, action: str) -> str:
        """Execute structured action with validation."""
        action = action.strip()
        
        # Validate action format
        valid_action = False
        for action_type, schema in self.action_schema.items():
            if action.startswith(schema['format'].split(':')[0]):
                valid_action = True
                break
        
        if not valid_action:
            return f"Invalid action format. Use one of: {list(self.action_schema.keys())}"
        
        return self.execute_action(action)
```

## 👥 Multi-Agent Systems

### Basic Multi-Agent Framework

```python
class Agent:
    def __init__(self, name: str, model, role: str, capabilities: List[str]):
        self.name = name
        self.model = model
        self.role = role
        self.capabilities = capabilities
        self.message_history = []
        
    def process_message(self, message: Dict) -> Dict:
        """Process incoming message and generate response."""
        # Add to history
        self.message_history.append(message)
        
        # Create context from recent messages
        context = self.build_context()
        
        # Generate response based on role and capabilities
        response_prompt = f"""
Role: {self.role}
Capabilities: {', '.join(self.capabilities)}

Recent conversation:
{context}

Current message from {message['sender']}: {message['content']}

Respond as {self.name} based on your role and capabilities:"""
        
        response_content = self.model.generate(response_prompt, max_tokens=300)
        
        response = {
            'sender': self.name,
            'recipient': message['sender'],
            'content': response_content.strip(),
            'timestamp': self.get_timestamp()
        }
        
        self.message_history.append(response)
        return response
    
    def build_context(self, max_messages: int = 5) -> str:
        """Build conversation context from recent messages."""
        recent_messages = self.message_history[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"{msg['sender']}: {msg['content']}")
        
        return '\n'.join(context_parts)
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = []
        self.conversation_log = []
        
    def add_agent(self, agent: Agent):
        """Add agent to the system."""
        self.agents[agent.name] = agent
    
    def send_message(self, sender: str, recipient: str, content: str) -> Dict:
        """Send message between agents."""
        if recipient not in self.agents:
            raise ValueError(f"Agent {recipient} not found")
        
        message = {
            'sender': sender,
            'recipient': recipient,
            'content': content,
            'timestamp': self.get_timestamp()
        }
        
        # Log message
        self.conversation_log.append(message)
        
        # Process message if recipient is an agent
        if recipient in self.agents:
            response = self.agents[recipient].process_message(message)
            self.conversation_log.append(response)
            return response
        
        return message
    
    def broadcast_message(self, sender: str, content: str) -> List[Dict]:
        """Broadcast message to all agents except sender."""
        responses = []
        
        for agent_name in self.agents:
            if agent_name != sender:
                response = self.send_message(sender, agent_name, content)
                responses.append(response)
        
        return responses
    
    def facilitate_discussion(self, topic: str, max_rounds: int = 5) -> List[Dict]:
        """Facilitate multi-agent discussion on a topic."""
        discussion_log = []
        
        # Initial broadcast
        initial_message = f"Let's discuss: {topic}. Each agent should contribute based on their expertise."
        
        # Start with first agent
        agent_names = list(self.agents.keys())
        if not agent_names:
            return discussion_log
        
        current_speaker = agent_names[0]
        
        for round_num in range(max_rounds):
            # Current agent speaks
            if round_num == 0:
                message_content = f"Starting discussion on: {topic}"
            else:
                # Agent responds to previous discussion
                recent_context = self.get_recent_context(3)
                message_content = f"Continuing discussion on {topic}. Recent context: {recent_context}"
            
            # Send to all other agents
            responses = self.broadcast_message(current_speaker, message_content)
            discussion_log.extend(responses)
            
            # Rotate speaker
            current_idx = agent_names.index(current_speaker)
            current_speaker = agent_names[(current_idx + 1) % len(agent_names)]
        
        return discussion_log
    
    def get_recent_context(self, num_messages: int) -> str:
        """Get recent conversation context."""
        recent = self.conversation_log[-num_messages:]
        return '; '.join([f"{msg['sender']}: {msg['content'][:50]}..." for msg in recent])
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

# Specialized agents
class PlannerAgent(Agent):
    def __init__(self, name: str, model):
        super().__init__(
            name=name,
            model=model,
            role="Strategic Planner",
            capabilities=["planning", "goal_decomposition", "resource_allocation"]
        )
    
    def create_collaborative_plan(self, goal: str, available_agents: List[str]) -> Dict:
        """Create plan involving multiple agents."""
        planning_prompt = f"""
Goal: {goal}

Available agents and their capabilities:
{self.format_agent_capabilities(available_agents)}

Create a collaborative plan that assigns tasks to appropriate agents:

Plan:"""
        
        plan = self.model.generate(planning_prompt, max_tokens=400)
        
        return {
            'goal': goal,
            'plan': plan,
            'assigned_agents': available_agents,
            'planner': self.name
        }
    
    def format_agent_capabilities(self, agent_names: List[str]) -> str:
        """Format agent capabilities for planning."""
        # This would typically query the actual agents
        return '\n'.join([f"- {name}: [capabilities would be listed here]" for name in agent_names])

class ExecutorAgent(Agent):
    def __init__(self, name: str, model, tools: Dict[str, callable]):
        super().__init__(
            name=name,
            model=model,
            role="Task Executor",
            capabilities=["action_execution", "tool_usage", "result_reporting"]
        )
        self.tools = tools
    
    def execute_task(self, task: Dict) -> Dict:
        """Execute assigned task."""
        execution_prompt = f"""
Task: {task['description']}
Available tools: {list(self.tools.keys())}

Execute this task step by step and report results:

Execution:"""
        
        execution_plan = self.model.generate(execution_prompt, max_tokens=300)
        
        # Simulate execution (in practice, would use actual tools)
        result = {
            'task': task,
            'execution_plan': execution_plan,
            'status': 'completed',
            'executor': self.name
        }
        
        return result

class CoordinatorAgent(Agent):
    def __init__(self, name: str, model):
        super().__init__(
            name=name,
            model=model,
            role="Coordinator",
            capabilities=["coordination", "conflict_resolution", "progress_tracking"]
        )
    
    def coordinate_agents(self, agents: List[str], shared_goal: str) -> Dict:
        """Coordinate multiple agents toward shared goal."""
        coordination_prompt = f"""
Shared goal: {shared_goal}
Agents to coordinate: {', '.join(agents)}

As a coordinator, create a coordination strategy:
1. How should agents communicate?
2. What is the task distribution?
3. How will progress be tracked?
4. How will conflicts be resolved?

Coordination Strategy:"""
        
        strategy = self.model.generate(coordination_prompt, max_tokens=400)
        
        return {
            'goal': shared_goal,
            'agents': agents,
            'coordination_strategy': strategy,
            'coordinator': self.name
        }
```

## 📚 Summary

### Key Components

**Planning**
- Hierarchical goal decomposition
- Dynamic replanning based on feedback
- Context-aware adaptive planning

**Reflection**
- Action-level reflection for learning
- Meta-cognitive reflection across goals
- Pattern recognition and improvement

**ReAct Pattern**
- Structured reasoning and acting cycles
- Tool integration for enhanced capabilities
- Iterative problem-solving approach

**Multi-Agent Systems**
- Specialized agent roles and capabilities
- Coordinated task execution
- Collaborative problem-solving

### Design Principles
- **Modularity**: Separate reasoning, planning, and execution
- **Adaptability**: Dynamic adjustment based on feedback
- **Transparency**: Clear reasoning traces and decision logs
- **Collaboration**: Effective multi-agent coordination

### Applications
- **Complex problem solving**: Multi-step reasoning tasks
- **Autonomous systems**: Self-directed goal achievement
- **Collaborative AI**: Multiple AI agents working together
- **Adaptive workflows**: Dynamic task planning and execution

### Future Directions
- **Improved planning**: More sophisticated goal decomposition
- **Better reflection**: Enhanced learning from experience
- **Robust coordination**: More effective multi-agent protocols
- **Tool integration**: Seamless external tool usage

Agentic LLM systems represent a significant advancement toward more autonomous and capable AI systems that can reason, plan, act, and collaborate effectively in complex environments.