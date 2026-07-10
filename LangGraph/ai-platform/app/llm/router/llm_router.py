"""
LLM-Based Router - Uses a small/fast model to classify the task, then routes.

This is the most sophisticated routing strategy:
    - A cheap model (GPT-4o-mini) classifies the request
    - Based on classification, routes to the best model for that task

Trade-offs:
    + Most accurate classification
    + Handles edge cases better than rules
    - Adds latency (one extra LLM call)
    - Adds cost (classification call)
    - Non-deterministic

When to use:
    - Tasks are ambiguous and hard to classify with rules
    - You need high accuracy in routing
    - The cost savings from correct routing outweigh the classification cost

LangChain equivalent:
    - Using a chain to classify, then routing based on output
    - LangGraph's conditional edges based on LLM output
"""

from typing import Optional
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from app.llm.router.base import BaseRouter, RoutingContext, RoutingDecision, TaskType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

CLASSIFICATION_PROMPT = """Classify the following user message into exactly ONE category.

Categories:
- SIMPLE_QA: Simple factual questions, greetings, basic info
- CODING: Code writing, debugging, code review, technical implementation
- REASONING: Complex analysis, comparisons, multi-step logic, math
- CREATIVE: Creative writing, brainstorming, storytelling
- VISION: Requests involving images or visual content
- LONG_CONTEXT: Requests that involve very long documents or context
- SUMMARIZATION: Summarizing text, documents, or conversations
- TRANSLATION: Language translation tasks

User message: {message}

Respond with ONLY the category name, nothing else."""

# Routing table: TaskType → (provider, model, reason)
ROUTING_TABLE = {
    TaskType.SIMPLE_QA: ("google", "gemini-1.5-flash", "Simple Q&A - fast and cheap"),
    TaskType.CODING: ("anthropic", "claude-3-5-sonnet-20241022", "Coding - Claude excels"),
    TaskType.REASONING: ("openai", "gpt-4o", "Complex reasoning - GPT-4o"),
    TaskType.CREATIVE: ("anthropic", "claude-3-5-sonnet-20241022", "Creative - Claude's strength"),
    TaskType.VISION: ("openai", "gpt-4o", "Vision - GPT-4o multimodal"),
    TaskType.LONG_CONTEXT: ("google", "gemini-1.5-pro", "Long context - 1M token window"),
    TaskType.SUMMARIZATION: ("google", "gemini-1.5-flash", "Summarization - fast model sufficient"),
    TaskType.TRANSLATION: ("openai", "gpt-4o-mini", "Translation - mini handles well"),
}


class LLMRouter(BaseRouter):
    """
    Uses a classifier LLM to determine the best model for the task.

    Architecture:
        User message → Classifier (cheap model) → TaskType → Routing Table → Target Model
    """

    def __init__(self, classifier_llm: Optional[BaseLLM] = None):
        """
        Args:
            classifier_llm: The cheap/fast model used for classification.
                           If None, will use the factory to create one.
        """
        if classifier_llm is None:
            from app.llm.factory import create_llm
            self.classifier = create_llm("openai")  # Use GPT-4o-mini for classification
        else:
            self.classifier = classifier_llm

    def route(self, context: RoutingContext) -> RoutingDecision:
        """Classify the message using LLM, then route based on classification."""
        task_type = self._classify(context.message)

        # Look up routing decision
        if task_type in ROUTING_TABLE:
            provider, model, reason = ROUTING_TABLE[task_type]
        else:
            # Fallback
            provider, model, reason = "openai", "gpt-4o-mini", "Fallback - unknown task type"

        decision = RoutingDecision(
            provider=provider,
            model=model,
            reason=f"LLM classified as {task_type.value} → {reason}",
        )

        logger.info(f"LLMRouter | classified={task_type.value} | routed_to={model}")
        return decision

    def _classify(self, message: str) -> TaskType:
        """Use the classifier LLM to determine task type."""
        prompt = CLASSIFICATION_PROMPT.format(message=message[:500])  # Truncate for cost

        try:
            response: LLMResponse = self.classifier.generate(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model="gpt-4o-mini", temperature=0, max_tokens=20),
            )

            # Parse the classification
            classification = response.content.strip().upper()
            return TaskType(classification.lower())

        except (ValueError, KeyError):
            logger.warning(f"LLMRouter classification failed, defaulting to SIMPLE_QA")
            return TaskType.SIMPLE_QA
        except Exception as e:
            logger.error(f"LLMRouter classifier error: {e}")
            return TaskType.SIMPLE_QA
