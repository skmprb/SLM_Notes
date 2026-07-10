"""
Prompt Management (Phase 9) - Template registry, versioning, variable substitution.

Production concerns:
    - Don't hardcode prompts in code
    - Version prompts (rollback if quality drops)
    - Variable substitution (dynamic context injection)
    - Token budget awareness (trim context to fit)
    - Hot-reload from files without restart

LangChain equivalent:
    - ChatPromptTemplate.from_messages()
    - PromptTemplate with input_variables
    - Hub (langchain hub for prompt versioning)
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class PromptTemplate:
    """A versioned prompt template with variable substitution."""
    name: str
    content: str
    version: str = "1.0"
    description: str = ""
    variables: list[str] = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render template with variables. Raises if required var missing."""
        result = self.content
        for var in self.variables:
            placeholder = "{" + var + "}"
            if placeholder in result:
                if var not in kwargs:
                    raise ValueError(f"Missing required variable: {var}")
                result = result.replace(placeholder, str(kwargs[var]))
        return result


class PromptRegistry:
    """Central registry for all prompt templates."""

    def __init__(self):
        self._prompts: dict[str, PromptTemplate] = {}
        self._load_from_files()

    def register(self, template: PromptTemplate) -> None:
        self._prompts[template.name] = template
        logger.info(f"Prompt registered: {template.name} v{template.version}")

    def get(self, name: str) -> Optional[PromptTemplate]:
        return self._prompts.get(name)

    def render(self, name: str, **kwargs) -> str:
        """Get and render a prompt by name."""
        template = self._prompts.get(name)
        if not template:
            raise ValueError(f"Prompt '{name}' not found")
        return template.render(**kwargs)

    def list_prompts(self) -> list[dict]:
        return [
            {"name": p.name, "version": p.version, "description": p.description, "variables": p.variables}
            for p in self._prompts.values()
        ]

    def _load_from_files(self) -> None:
        """Load .md prompt files from templates directory."""
        if not TEMPLATES_DIR.exists():
            return
        for file in TEMPLATES_DIR.glob("*.md"):
            name = file.stem
            content = file.read_text(encoding="utf-8")
            # Extract variables like {variable_name} from content
            import re
            variables = re.findall(r"\{(\w+)\}", content)
            self.register(PromptTemplate(
                name=name, content=content, variables=list(set(variables)),
                description=f"Loaded from {file.name}",
            ))


_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
