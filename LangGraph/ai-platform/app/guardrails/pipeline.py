"""
Guardrails (Phase 12) - Input/output validation, safety checks.

Input guardrails (before LLM call):
    - PII detection (emails, phones, SSNs)
    - Prompt injection detection
    - Content policy (toxicity, harmful content)
    - Length/token limits

Output guardrails (after LLM response):
    - PII leakage check
    - Hallucination indicators
    - Schema compliance
    - Content policy enforcement

Architecture:
    Request → [Input Guards] → LLM → [Output Guards] → Response
                    ↓                        ↓
              Block/Redact              Block/Redact/Flag

LangChain equivalent:
    - Guardrails AI library
    - NeMo Guardrails
    - Custom RunnablePassthrough with checks
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"  # Allow but flag for review


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    action: GuardrailAction = GuardrailAction.ALLOW
    reason: Optional[str] = None
    redacted_content: Optional[str] = None  # If action=REDACT, the cleaned version
    violations: list[str] = field(default_factory=list)


class BaseGuardrail(ABC):
    """Abstract guardrail check."""
    name: str = "base"

    @abstractmethod
    def check(self, content: str) -> GuardrailResult: ...


# ============================================================
# INPUT GUARDRAILS
# ============================================================

class PIIDetector(BaseGuardrail):
    """Detect and optionally redact PII (emails, phones, SSNs)."""
    name = "pii_detector"

    PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def __init__(self, action: GuardrailAction = GuardrailAction.REDACT):
        self.action = action

    def check(self, content: str) -> GuardrailResult:
        violations = []
        redacted = content

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{pii_type}: {len(matches)} found")
                redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)

        if violations:
            return GuardrailResult(
                passed=self.action != GuardrailAction.BLOCK,
                action=self.action,
                reason=f"PII detected: {', '.join(violations)}",
                redacted_content=redacted,
                violations=violations,
            )
        return GuardrailResult(passed=True)


class PromptInjectionDetector(BaseGuardrail):
    """Detect common prompt injection patterns."""
    name = "injection_detector"

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"system\s*:\s*",
        r"<\|im_start\|>",
        r"###\s*(?:system|instruction)",
        r"forget\s+(?:everything|all|your)",
    ]

    def check(self, content: str) -> GuardrailResult:
        content_lower = content.lower()
        violations = []

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, content_lower):
                violations.append(f"Pattern: {pattern}")

        if violations:
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.BLOCK,
                reason="Potential prompt injection detected",
                violations=violations,
            )
        return GuardrailResult(passed=True)


class ContentLengthGuard(BaseGuardrail):
    """Enforce maximum input length."""
    name = "content_length"

    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars

    def check(self, content: str) -> GuardrailResult:
        if len(content) > self.max_chars:
            return GuardrailResult(
                passed=False,
                action=GuardrailAction.BLOCK,
                reason=f"Content too long: {len(content)} chars (max {self.max_chars})",
                violations=[f"length={len(content)}"],
            )
        return GuardrailResult(passed=True)


# ============================================================
# GUARDRAIL PIPELINE
# ============================================================

class GuardrailPipeline:
    """Runs multiple guardrails in sequence."""

    def __init__(self):
        self.input_guards: list[BaseGuardrail] = [
            PIIDetector(),
            PromptInjectionDetector(),
            ContentLengthGuard(),
        ]
        self.output_guards: list[BaseGuardrail] = [
            PIIDetector(action=GuardrailAction.REDACT),
        ]

    def check_input(self, content: str) -> GuardrailResult:
        """Run all input guardrails. Returns first blocking result or ALLOW."""
        for guard in self.input_guards:
            result = guard.check(content)
            if not result.passed:
                logger.warning(f"Input blocked by {guard.name}: {result.reason}")
                return result
            if result.action == GuardrailAction.REDACT and result.redacted_content:
                content = result.redacted_content
        return GuardrailResult(passed=True, redacted_content=content)

    def check_output(self, content: str) -> GuardrailResult:
        """Run all output guardrails."""
        for guard in self.output_guards:
            result = guard.check(content)
            if not result.passed:
                logger.warning(f"Output blocked by {guard.name}: {result.reason}")
                return result
            if result.action == GuardrailAction.REDACT and result.redacted_content:
                content = result.redacted_content
        return GuardrailResult(passed=True, redacted_content=content)


_pipeline: Optional[GuardrailPipeline] = None


def get_guardrail_pipeline() -> GuardrailPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GuardrailPipeline()
    return _pipeline
