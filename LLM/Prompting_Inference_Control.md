# Prompting and Inference Control

## 🎯 Overview

Prompting techniques enable users to control LLM behavior and extract desired outputs without fine-tuning. These methods leverage the model's pre-trained knowledge and reasoning capabilities through carefully crafted inputs.

## 🛠️ Prompt Engineering

### Basic Prompt Design

```python
class PromptTemplate:
    def __init__(self, template: str, variables: List[str]):
        self.template = template
        self.variables = variables
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        return self.template.format(**kwargs)

# Example templates
CLASSIFICATION_TEMPLATE = PromptTemplate(
    template="""Classify the following text into one of these categories: {categories}

Text: {text}

Category:""",
    variables=["categories", "text"]
)

SUMMARIZATION_TEMPLATE = PromptTemplate(
    template="""Summarize the following text in {max_words} words or less:

{text}

Summary:""",
    variables=["text", "max_words"]
)

QA_TEMPLATE = PromptTemplate(
    template="""Answer the following question based on the given context. If the answer cannot be found in the context, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    variables=["context", "question"]
)
```

### Advanced Prompt Techniques

```python
class AdvancedPromptBuilder:
    def __init__(self):
        self.system_prompts = {
            'helpful': "You are a helpful, harmless, and honest assistant.",
            'expert': "You are an expert in {domain} with deep knowledge and experience.",
            'creative': "You are a creative assistant who thinks outside the box.",
            'analytical': "You are an analytical assistant who provides detailed reasoning."
        }
    
    def build_role_prompt(self, role: str, domain: str = None) -> str:
        """Build role-based system prompt."""
        if role in self.system_prompts:
            prompt = self.system_prompts[role]
            if domain and '{domain}' in prompt:
                prompt = prompt.format(domain=domain)
            return prompt
        return self.system_prompts['helpful']
    
    def build_structured_prompt(self, task: str, context: str = None, 
                              examples: List[Dict] = None, 
                              constraints: List[str] = None) -> str:
        """Build structured prompt with multiple components."""
        parts = []
        
        # Task description
        parts.append(f"Task: {task}")
        
        # Context if provided
        if context:
            parts.append(f"Context: {context}")
        
        # Examples if provided
        if examples:
            parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Input: {example['input']}")
                parts.append(f"Output: {example['output']}")
        
        # Constraints if provided
        if constraints:
            parts.append("Constraints:")
            for constraint in constraints:
                parts.append(f"- {constraint}")
        
        return "\n\n".join(parts)
    
    def build_chain_of_thought_prompt(self, problem: str, 
                                    encourage_reasoning: bool = True) -> str:
        """Build chain-of-thought reasoning prompt."""
        base_prompt = f"Problem: {problem}"
        
        if encourage_reasoning:
            base_prompt += "\n\nLet's think step by step:"
        
        return base_prompt
```

### Prompt Optimization

```python
class PromptOptimizer:
    def __init__(self, model, evaluator):
        self.model = model
        self.evaluator = evaluator
        
    def optimize_prompt(self, base_prompt: str, test_cases: List[Dict], 
                       variations: List[str]) -> str:
        """Optimize prompt by testing variations."""
        best_prompt = base_prompt
        best_score = 0
        
        # Test base prompt
        base_score = self.evaluate_prompt(base_prompt, test_cases)
        best_score = base_score
        
        # Test variations
        for variation in variations:
            score = self.evaluate_prompt(variation, test_cases)
            if score > best_score:
                best_score = score
                best_prompt = variation
        
        return best_prompt
    
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict]) -> float:
        """Evaluate prompt performance on test cases."""
        scores = []
        
        for test_case in test_cases:
            # Format prompt with test input
            formatted_prompt = prompt.format(**test_case['input'])
            
            # Generate response
            response = self.model.generate(formatted_prompt)
            
            # Evaluate response
            score = self.evaluator.evaluate(response, test_case['expected'])
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def generate_prompt_variations(self, base_prompt: str) -> List[str]:
        """Generate prompt variations for optimization."""
        variations = []
        
        # Add instruction variations
        instruction_variants = [
            "Please",
            "Carefully",
            "Step by step,",
            "Think carefully and"
        ]
        
        for variant in instruction_variants:
            variations.append(f"{variant} {base_prompt.lower()}")
        
        # Add format variations
        format_variants = [
            f"{base_prompt}\n\nFormat your response as:",
            f"{base_prompt}\n\nProvide a detailed explanation:",
            f"{base_prompt}\n\nBe concise and direct:"
        ]
        
        variations.extend(format_variants)
        
        return variations
```

## 🎓 In-Context Learning

### Basic In-Context Learning

```python
class InContextLearner:
    def __init__(self, model):
        self.model = model
        
    def create_context(self, examples: List[Dict], task_description: str = None) -> str:
        """Create in-context learning prompt from examples."""
        context_parts = []
        
        if task_description:
            context_parts.append(task_description)
        
        # Add examples
        for example in examples:
            if 'input' in example and 'output' in example:
                context_parts.append(f"Input: {example['input']}")
                context_parts.append(f"Output: {example['output']}")
            elif 'question' in example and 'answer' in example:
                context_parts.append(f"Q: {example['question']}")
                context_parts.append(f"A: {example['answer']}")
        
        return "\n\n".join(context_parts)
    
    def predict(self, context: str, new_input: str, 
               input_format: str = "Input", output_format: str = "Output") -> str:
        """Make prediction using in-context learning."""
        prompt = f"{context}\n\n{input_format}: {new_input}\n{output_format}:"
        
        response = self.model.generate(prompt, max_tokens=100, stop=["\n\n"])
        return response.strip()

# Example usage
examples = [
    {"input": "The movie was fantastic!", "output": "Positive"},
    {"input": "I hated this book.", "output": "Negative"},
    {"input": "The weather is okay.", "output": "Neutral"}
]

icl = InContextLearner(model)
context = icl.create_context(examples, "Classify sentiment as Positive, Negative, or Neutral:")
result = icl.predict(context, "This product is amazing!")
```

### Dynamic Example Selection

```python
class DynamicExampleSelector:
    def __init__(self, example_pool: List[Dict], similarity_metric='cosine'):
        self.example_pool = example_pool
        self.similarity_metric = similarity_metric
        self.embeddings = None
        
    def compute_embeddings(self, embedding_model):
        """Compute embeddings for all examples."""
        texts = [ex['input'] for ex in self.example_pool]
        self.embeddings = embedding_model.encode(texts)
    
    def select_examples(self, query: str, k: int = 3, 
                       embedding_model=None) -> List[Dict]:
        """Select most relevant examples for query."""
        if self.embeddings is None and embedding_model:
            self.compute_embeddings(embedding_model)
        
        if self.embeddings is None:
            # Fallback to random selection
            import random
            return random.sample(self.example_pool, min(k, len(self.example_pool)))
        
        # Compute query embedding
        query_embedding = embedding_model.encode([query])
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Select top-k most similar examples
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.example_pool[i] for i in top_indices]

class AdaptiveInContextLearner(InContextLearner):
    def __init__(self, model, example_selector: DynamicExampleSelector):
        super().__init__(model)
        self.example_selector = example_selector
    
    def predict_with_selection(self, query: str, k: int = 3, 
                             embedding_model=None) -> str:
        """Predict using dynamically selected examples."""
        # Select relevant examples
        examples = self.example_selector.select_examples(query, k, embedding_model)
        
        # Create context and predict
        context = self.create_context(examples)
        return self.predict(context, query)
```

## 🎯 Few-Shot Prompting

### Few-Shot Learning Framework

```python
class FewShotLearner:
    def __init__(self, model):
        self.model = model
        
    def few_shot_classify(self, examples: List[Dict], query: str, 
                         labels: List[str]) -> str:
        """Few-shot classification."""
        prompt_parts = ["Classify the following examples:"]
        
        # Add examples
        for example in examples:
            prompt_parts.append(f"Text: {example['text']}")
            prompt_parts.append(f"Label: {example['label']}")
            prompt_parts.append("")
        
        # Add query
        prompt_parts.append(f"Text: {query}")
        prompt_parts.append("Label:")
        
        prompt = "\n".join(prompt_parts)
        response = self.model.generate(prompt, max_tokens=10)
        
        # Extract predicted label
        predicted = response.strip()
        return predicted if predicted in labels else labels[0]
    
    def few_shot_generate(self, examples: List[Dict], input_text: str, 
                         task_description: str = None) -> str:
        """Few-shot text generation."""
        prompt_parts = []
        
        if task_description:
            prompt_parts.append(task_description)
            prompt_parts.append("")
        
        # Add examples
        for example in examples:
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append("")
        
        # Add new input
        prompt_parts.append(f"Input: {input_text}")
        prompt_parts.append("Output:")
        
        prompt = "\n".join(prompt_parts)
        return self.model.generate(prompt, max_tokens=200)

# Specialized few-shot tasks
class FewShotTaskSolver:
    def __init__(self, model):
        self.model = model
        
    def solve_math_word_problems(self, examples: List[Dict], problem: str) -> str:
        """Solve math word problems with few-shot learning."""
        prompt = "Solve these math word problems:\n\n"
        
        for example in examples:
            prompt += f"Problem: {example['problem']}\n"
            prompt += f"Solution: {example['solution']}\n\n"
        
        prompt += f"Problem: {problem}\nSolution:"
        
        return self.model.generate(prompt, max_tokens=150)
    
    def translate_with_examples(self, examples: List[Dict], 
                              text: str, target_lang: str) -> str:
        """Translate using few-shot examples."""
        prompt = f"Translate to {target_lang}:\n\n"
        
        for example in examples:
            prompt += f"{example['source']}\n{example['target']}\n\n"
        
        prompt += f"{text}\n"
        
        return self.model.generate(prompt, max_tokens=100)
```

## 🧠 Chain-of-Thought

### Basic Chain-of-Thought

```python
class ChainOfThoughtReasoner:
    def __init__(self, model):
        self.model = model
        
    def solve_with_cot(self, problem: str, examples: List[Dict] = None) -> Dict[str, str]:
        """Solve problem using chain-of-thought reasoning."""
        prompt_parts = []
        
        # Add examples if provided
        if examples:
            for example in examples:
                prompt_parts.append(f"Problem: {example['problem']}")
                prompt_parts.append(f"Let's think step by step.")
                prompt_parts.append(f"Reasoning: {example['reasoning']}")
                prompt_parts.append(f"Answer: {example['answer']}")
                prompt_parts.append("")
        
        # Add current problem
        prompt_parts.append(f"Problem: {problem}")
        prompt_parts.append("Let's think step by step.")
        prompt_parts.append("Reasoning:")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate reasoning
        reasoning_response = self.model.generate(prompt, max_tokens=300, stop=["Answer:"])
        
        # Generate final answer
        answer_prompt = prompt + reasoning_response + "\nAnswer:"
        answer_response = self.model.generate(answer_prompt, max_tokens=50)
        
        return {
            'reasoning': reasoning_response.strip(),
            'answer': answer_response.strip()
        }
    
    def multi_step_reasoning(self, problem: str, steps: List[str]) -> Dict[str, any]:
        """Guide reasoning through specific steps."""
        results = {'steps': {}, 'final_answer': ''}
        
        current_context = f"Problem: {problem}\n\n"
        
        for i, step in enumerate(steps, 1):
            step_prompt = current_context + f"Step {i}: {step}\n"
            step_response = self.model.generate(step_prompt, max_tokens=150)
            
            results['steps'][f'step_{i}'] = {
                'instruction': step,
                'response': step_response.strip()
            }
            
            current_context += f"Step {i}: {step}\n{step_response}\n\n"
        
        # Generate final answer
        final_prompt = current_context + "Final Answer:"
        final_answer = self.model.generate(final_prompt, max_tokens=100)
        results['final_answer'] = final_answer.strip()
        
        return results

# Specialized CoT for different domains
class MathCoTSolver(ChainOfThoughtReasoner):
    def solve_arithmetic(self, problem: str) -> Dict[str, str]:
        """Solve arithmetic problems with CoT."""
        examples = [
            {
                'problem': "What is 23 + 47?",
                'reasoning': "I need to add 23 and 47. 23 + 47 = 70.",
                'answer': "70"
            }
        ]
        
        return self.solve_with_cot(problem, examples)
    
    def solve_word_problem(self, problem: str) -> Dict[str, str]:
        """Solve math word problems with CoT."""
        steps = [
            "Identify what we need to find",
            "Extract the relevant numbers and operations",
            "Set up the calculation",
            "Perform the calculation",
            "Check if the answer makes sense"
        ]
        
        return self.multi_step_reasoning(problem, steps)
```

### Advanced Chain-of-Thought

```python
class AdvancedCoTReasoner:
    def __init__(self, model):
        self.model = model
        
    def tree_of_thoughts(self, problem: str, num_branches: int = 3) -> Dict[str, any]:
        """Generate multiple reasoning paths (Tree of Thoughts)."""
        branches = []
        
        for i in range(num_branches):
            # Generate different reasoning approach
            prompt = f"""Problem: {problem}

Let's approach this from angle {i+1}. Think step by step:"""
            
            reasoning = self.model.generate(prompt, max_tokens=200, temperature=0.7)
            
            # Generate answer for this branch
            answer_prompt = prompt + reasoning + "\n\nTherefore, the answer is:"
            answer = self.model.generate(answer_prompt, max_tokens=50, temperature=0.3)
            
            branches.append({
                'branch_id': i+1,
                'reasoning': reasoning.strip(),
                'answer': answer.strip()
            })
        
        # Select best branch (simplified)
        best_branch = self.select_best_branch(branches)
        
        return {
            'all_branches': branches,
            'best_branch': best_branch,
            'final_answer': best_branch['answer']
        }
    
    def select_best_branch(self, branches: List[Dict]) -> Dict:
        """Select the best reasoning branch."""
        # Simple heuristic: longest reasoning (more detailed)
        return max(branches, key=lambda b: len(b['reasoning']))
    
    def verify_reasoning(self, problem: str, reasoning: str, answer: str) -> Dict[str, any]:
        """Verify the correctness of reasoning."""
        verification_prompt = f"""Problem: {problem}

Proposed reasoning: {reasoning}

Proposed answer: {answer}

Is this reasoning correct? Explain any errors or confirm correctness:"""
        
        verification = self.model.generate(verification_prompt, max_tokens=200)
        
        return {
            'verification': verification.strip(),
            'is_likely_correct': 'correct' in verification.lower() and 'error' not in verification.lower()
        }
```

## 🔄 Self-Consistency

### Self-Consistency Implementation

```python
class SelfConsistencyReasoner:
    def __init__(self, model):
        self.model = model
        
    def generate_multiple_solutions(self, problem: str, num_samples: int = 5, 
                                  temperature: float = 0.7) -> List[Dict]:
        """Generate multiple solutions with different reasoning paths."""
        solutions = []
        
        prompt = f"""Problem: {problem}

Let's think step by step:"""
        
        for i in range(num_samples):
            # Generate reasoning with some randomness
            reasoning = self.model.generate(
                prompt, 
                max_tokens=300, 
                temperature=temperature,
                stop=["Answer:", "Therefore:"]
            )
            
            # Extract final answer
            answer_prompt = prompt + reasoning + "\n\nTherefore, the answer is:"
            answer = self.model.generate(answer_prompt, max_tokens=50, temperature=0.1)
            
            solutions.append({
                'solution_id': i+1,
                'reasoning': reasoning.strip(),
                'answer': self.extract_final_answer(answer.strip())
            })
        
        return solutions
    
    def extract_final_answer(self, response: str) -> str:
        """Extract the final numerical or categorical answer."""
        import re
        
        # Look for numbers
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]  # Return last number found
        
        # Look for common answer patterns
        answer_patterns = [
            r'answer is (.+?)[\.\n]',
            r'result is (.+?)[\.\n]',
            r'equals (.+?)[\.\n]'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return first few words if no pattern found
        return response.split()[:3] if response.split() else response
    
    def find_consensus_answer(self, solutions: List[Dict]) -> Dict[str, any]:
        """Find the most consistent answer across solutions."""
        from collections import Counter
        
        # Count answer frequencies
        answers = [sol['answer'] for sol in solutions]
        answer_counts = Counter(answers)
        
        # Find most common answer
        most_common_answer, frequency = answer_counts.most_common(1)[0]
        confidence = frequency / len(solutions)
        
        # Get all solutions with the consensus answer
        consensus_solutions = [sol for sol in solutions 
                             if sol['answer'] == most_common_answer]
        
        return {
            'consensus_answer': most_common_answer,
            'confidence': confidence,
            'frequency': frequency,
            'total_solutions': len(solutions),
            'consensus_solutions': consensus_solutions,
            'all_answers': dict(answer_counts)
        }
    
    def solve_with_self_consistency(self, problem: str, num_samples: int = 5) -> Dict[str, any]:
        """Solve problem using self-consistency approach."""
        # Generate multiple solutions
        solutions = self.generate_multiple_solutions(problem, num_samples)
        
        # Find consensus
        consensus = self.find_consensus_answer(solutions)
        
        return {
            'problem': problem,
            'all_solutions': solutions,
            'consensus': consensus,
            'final_answer': consensus['consensus_answer'],
            'confidence_score': consensus['confidence']
        }
```

### Weighted Self-Consistency

```python
class WeightedSelfConsistency(SelfConsistencyReasoner):
    def __init__(self, model, quality_scorer=None):
        super().__init__(model)
        self.quality_scorer = quality_scorer or self.default_quality_scorer
        
    def default_quality_scorer(self, reasoning: str) -> float:
        """Default quality scoring based on reasoning length and structure."""
        # Simple heuristics
        score = 0.5  # Base score
        
        # Longer reasoning might be more detailed
        if len(reasoning.split()) > 50:
            score += 0.2
        
        # Check for mathematical operations
        if any(op in reasoning for op in ['+', '-', '*', '/', '=']):
            score += 0.2
        
        # Check for step-by-step indicators
        step_indicators = ['first', 'then', 'next', 'finally', 'step']
        if any(indicator in reasoning.lower() for indicator in step_indicators):
            score += 0.1
        
        return min(score, 1.0)
    
    def weighted_consensus(self, solutions: List[Dict]) -> Dict[str, any]:
        """Find consensus using quality-weighted voting."""
        from collections import defaultdict
        
        # Calculate quality scores
        for solution in solutions:
            solution['quality_score'] = self.quality_scorer(solution['reasoning'])
        
        # Weighted voting
        answer_weights = defaultdict(float)
        
        for solution in solutions:
            answer = solution['answer']
            weight = solution['quality_score']
            answer_weights[answer] += weight
        
        # Find best weighted answer
        best_answer = max(answer_weights, key=answer_weights.get)
        total_weight = sum(answer_weights.values())
        confidence = answer_weights[best_answer] / total_weight if total_weight > 0 else 0
        
        return {
            'weighted_answer': best_answer,
            'weighted_confidence': confidence,
            'answer_weights': dict(answer_weights),
            'quality_scores': [sol['quality_score'] for sol in solutions]
        }
```

## 🔧 Tool Calling

### Basic Tool Integration

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        
    def register_tool(self, name: str, func: callable, description: str, 
                     parameters: Dict[str, str]):
        """Register a tool for LLM use."""
        self.tools[name] = {
            'function': func,
            'description': description,
            'parameters': parameters
        }
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt."""
        descriptions = []
        
        for name, tool in self.tools.items():
            desc = f"Tool: {name}\n"
            desc += f"Description: {tool['description']}\n"
            desc += f"Parameters: {', '.join(tool['parameters'].keys())}\n"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def execute_tool(self, name: str, **kwargs) -> any:
        """Execute a registered tool."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        tool = self.tools[name]
        return tool['function'](**kwargs)

# Example tools
def calculator(expression: str) -> float:
    """Safe calculator for basic arithmetic."""
    try:
        # Simple safety check
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query: str, num_results: int = 3) -> List[str]:
    """Simulate web search (placeholder)."""
    # In practice, integrate with actual search API
    return [
        f"Search result 1 for '{query}'",
        f"Search result 2 for '{query}'",
        f"Search result 3 for '{query}'"
    ][:num_results]

def get_current_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Register tools
registry = ToolRegistry()
registry.register_tool(
    "calculator", 
    calculator, 
    "Perform arithmetic calculations",
    {"expression": "Mathematical expression to evaluate"}
)
registry.register_tool(
    "web_search", 
    web_search, 
    "Search the web for information",
    {"query": "Search query", "num_results": "Number of results (optional)"}
)
registry.register_tool(
    "get_time", 
    get_current_time, 
    "Get current date and time",
    {}
)
```

### Tool-Augmented LLM

```python
import json
import re

class ToolAugmentedLLM:
    def __init__(self, model, tool_registry: ToolRegistry):
        self.model = model
        self.tool_registry = tool_registry
        
    def create_tool_prompt(self, user_query: str) -> str:
        """Create prompt that includes tool descriptions."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        
        prompt = f"""You are an AI assistant with access to the following tools:

{tool_descriptions}

To use a tool, format your response as:
TOOL_CALL: tool_name(parameter1="value1", parameter2="value2")

You can use multiple tools in sequence. Always provide a final answer after using tools.

User Query: {user_query}

Response:"""
        
        return prompt
    
    def parse_tool_calls(self, response: str) -> List[Dict]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        
        # Find tool call patterns
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(pattern, response)
        
        for tool_name, params_str in matches:
            # Parse parameters
            params = {}
            if params_str.strip():
                # Simple parameter parsing
                param_matches = re.findall(r'(\w+)="([^"]*)"', params_str)
                params = {key: value for key, value in param_matches}
            
            tool_calls.append({
                'tool': tool_name,
                'parameters': params
            })
        
        return tool_calls
    
    def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute parsed tool calls."""
        results = []
        
        for call in tool_calls:
            try:
                result = self.tool_registry.execute_tool(
                    call['tool'], 
                    **call['parameters']
                )
                results.append({
                    'tool': call['tool'],
                    'parameters': call['parameters'],
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'tool': call['tool'],
                    'parameters': call['parameters'],
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def generate_with_tools(self, user_query: str, max_iterations: int = 3) -> Dict[str, any]:
        """Generate response using tools iteratively."""
        conversation_history = []
        tool_results_history = []
        
        current_query = user_query
        
        for iteration in range(max_iterations):
            # Create prompt
            prompt = self.create_tool_prompt(current_query)
            
            # Add previous tool results to context
            if tool_results_history:
                prompt += "\n\nPrevious tool results:\n"
                for result in tool_results_history:
                    prompt += f"{result['tool']}: {result['result']}\n"
            
            # Generate response
            response = self.model.generate(prompt, max_tokens=500)
            conversation_history.append(response)
            
            # Parse and execute tool calls
            tool_calls = self.parse_tool_calls(response)
            
            if not tool_calls:
                # No more tool calls, return final response
                return {
                    'final_response': response,
                    'conversation_history': conversation_history,
                    'tool_results': tool_results_history,
                    'iterations': iteration + 1
                }
            
            # Execute tools
            tool_results = self.execute_tool_calls(tool_calls)
            tool_results_history.extend(tool_results)
            
            # Prepare next iteration
            current_query = f"Based on the tool results, continue with: {user_query}"
        
        return {
            'final_response': conversation_history[-1] if conversation_history else "",
            'conversation_history': conversation_history,
            'tool_results': tool_results_history,
            'iterations': max_iterations,
            'max_iterations_reached': True
        }

# Example usage
model = None  # Your LLM model
tool_llm = ToolAugmentedLLM(model, registry)

# Example query that would use tools
query = "What's 15 * 23 + 47, and what time is it now?"
result = tool_llm.generate_with_tools(query)
```

### Advanced Tool Orchestration

```python
class AdvancedToolOrchestrator:
    def __init__(self, model, tool_registry: ToolRegistry):
        self.model = model
        self.tool_registry = tool_registry
        self.execution_plan = []
        
    def plan_tool_usage(self, user_query: str) -> List[Dict]:
        """Plan which tools to use and in what order."""
        planning_prompt = f"""Given the user query and available tools, create an execution plan.

Available tools:
{self.tool_registry.get_tool_descriptions()}

User Query: {user_query}

Create a step-by-step plan using the format:
Step 1: Use tool_name to accomplish X
Step 2: Use tool_name to accomplish Y
...

Plan:"""
        
        plan_response = self.model.generate(planning_prompt, max_tokens=300)
        
        # Parse plan (simplified)
        steps = []
        for line in plan_response.split('\n'):
            if line.strip().startswith('Step'):
                steps.append({'description': line.strip()})
        
        return steps
    
    def execute_planned_tools(self, plan: List[Dict], user_query: str) -> Dict[str, any]:
        """Execute tools according to plan."""
        results = []
        context = f"Original query: {user_query}\n\n"
        
        for i, step in enumerate(plan, 1):
            step_prompt = f"""{context}
Current step: {step['description']}

Execute this step by calling the appropriate tool:"""
            
            response = self.model.generate(step_prompt, max_tokens=200)
            
            # Parse and execute tool calls
            tool_calls = self.parse_tool_calls(response)
            
            if tool_calls:
                tool_results = self.execute_tool_calls(tool_calls)
                results.extend(tool_results)
                
                # Update context with results
                for result in tool_results:
                    context += f"Step {i} result: {result['result']}\n"
        
        # Generate final synthesis
        synthesis_prompt = f"""{context}

Based on all the tool results above, provide a comprehensive answer to the original query: {user_query}

Final Answer:"""
        
        final_answer = self.model.generate(synthesis_prompt, max_tokens=300)
        
        return {
            'plan': plan,
            'tool_results': results,
            'final_answer': final_answer,
            'execution_context': context
        }
```

## 📚 Summary

### Key Techniques

**Prompt Engineering**
- Template-based prompts for consistency
- Role-based and structured prompting
- Prompt optimization through testing

**In-Context Learning**
- Learning from examples without fine-tuning
- Dynamic example selection for relevance
- Adaptive context construction

**Few-Shot Prompting**
- Task-specific examples for guidance
- Specialized solvers for different domains
- Example quality and selection strategies

**Chain-of-Thought**
- Step-by-step reasoning for complex problems
- Multi-path reasoning with Tree of Thoughts
- Verification and error checking

**Self-Consistency**
- Multiple solution generation for reliability
- Consensus finding through voting
- Quality-weighted decision making

**Tool Calling**
- External tool integration for enhanced capabilities
- Structured tool execution and orchestration
- Planning and multi-step tool usage

### Best Practices
- **Clear instructions**: Specific and unambiguous prompts
- **Example quality**: High-quality, relevant examples
- **Iterative refinement**: Test and optimize prompts
- **Error handling**: Robust parsing and execution
- **Context management**: Maintain relevant information across interactions

### Applications
- **Problem solving**: Mathematical and logical reasoning
- **Information retrieval**: Search and synthesis tasks
- **Creative tasks**: Writing and content generation
- **Analysis**: Data interpretation and insights
- **Automation**: Tool-augmented workflows

These techniques enable sophisticated control over LLM behavior and unlock capabilities beyond basic text generation, making LLMs more useful for complex, real-world applications.