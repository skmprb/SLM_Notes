"""
Conversation Models - Data structures for session and message management.

Hierarchy:
    User → has many Sessions
    Session → has many Messages
    Message → has role, content, metadata

This is what every chat application needs:
    - ChatGPT has "conversations"
    - Claude has "chats"
    - We call them "sessions" (more generic, supports non-chat use cases)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    # Metadata
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_llm_format(self) -> dict:
        """Convert to the format LLMs expect: {"role": "user", "content": "..."}"""
        return {"role": self.role.value, "content": self.content}


@dataclass
class Session:
    """A conversation session containing messages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    total_tokens: int = 0
    message_count: int = 0

    def add_message(self, message: Message):
        """Add a message and update metadata."""
        self.messages.append(message)
        self.message_count = len(self.messages)
        self.updated_at = datetime.utcnow()
        if message.tokens_used:
            self.total_tokens += message.tokens_used

    def get_messages_for_llm(self, max_messages: int | None = None) -> list[dict]:
        """
        Get messages in LLM format, optionally limited.

        This is the simplest context window management:
        just take the last N messages.
        """
        messages = self.messages
        if max_messages and len(messages) > max_messages:
            # Always keep system message + last N messages
            system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
            non_system = [m for m in messages if m.role != MessageRole.SYSTEM]
            messages = system_msgs + non_system[-(max_messages - len(system_msgs)):]
        return [m.to_llm_format() for m in messages]

    def auto_title(self) -> str:
        """Generate a title from the first user message."""
        for msg in self.messages:
            if msg.role == MessageRole.USER:
                return msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        return "New conversation"
