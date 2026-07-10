"""
ConversationService - Manages conversation lifecycle.

Responsibilities:
    - Create/retrieve/delete sessions
    - Add messages to sessions
    - Manage context window (trim old messages)
    - Build message list for LLM (system prompt + history + current message)

Architecture:
    API → ChatService → ConversationService → Store (InMemory/SQLite)
                      → LLMService → Provider

LangChain equivalent:
    - LangGraph Checkpointer (thread-scoped state persistence)
    - ChatMessageHistory (stores messages)
    - trim_messages() (context window management)
    - The key insight: LangGraph ties memory to GRAPH STATE
      We tie memory to SESSIONS (more traditional, easier to understand first)
"""

from typing import Optional
from app.storage.models import Session, Message, MessageRole
from app.storage.conversation_store import BaseConversationStore, InMemoryStore, SQLiteStore
from app.config.settings import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant built into an Enterprise AI Platform. 
You provide clear, concise, and accurate responses. 
If you don't know something, say so honestly."""


class ConversationService:
    """
    Manages conversation sessions and message history.

    Usage:
        service = ConversationService()

        # Start a new conversation
        session = service.create_session()

        # Add user message and get context for LLM
        messages = service.prepare_messages(session.id, "What is Python?")

        # After LLM responds, save the response
        service.add_assistant_message(session.id, "Python is...", model="gpt-4o", tokens=42)
    """

    def __init__(self, store: Optional[BaseConversationStore] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        if store:
            self.store = store
        elif self.settings.conversation_store == "sqlite":
            self.store = SQLiteStore(self.settings.sqlite_db_path)
        else:
            self.store = InMemoryStore()

        self.max_history_messages = 20  # Keep last 20 messages in context
        logger.info(f"ConversationService initialized | store={type(self.store).__name__}")

    def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create a new conversation session."""
        session = self.store.create_session(session_id)
        # Add system prompt as first message
        system_msg = Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        session.add_message(system_msg)
        self.store.save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self.store.get_session(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.store.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id)

    def prepare_messages(self, session_id: str, user_message: str) -> list[dict]:
        """
        Add user message to session and return messages for LLM.

        This is the core method:
            1. Get session
            2. Add user message
            3. Apply context window management
            4. Return messages in LLM format

        Returns:
            List of message dicts ready for the LLM
        """
        session = self.get_or_create_session(session_id)

        # Add user message
        user_msg = Message(role=MessageRole.USER, content=user_message)
        session.add_message(user_msg)

        # Auto-title from first user message
        if not session.title:
            session.title = session.auto_title()

        self.store.save_session(session)

        # Return messages with context window management
        return session.get_messages_for_llm(max_messages=self.max_history_messages)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> Message:
        """Save the assistant's response to the session."""
        session = self.store.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        session.add_message(assistant_msg)
        self.store.save_session(session)
        return assistant_msg

    def list_sessions(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """List all sessions with summary info."""
        sessions = self.store.list_sessions(limit, offset)
        return [
            {
                "id": s.id,
                "title": s.title or "Untitled",
                "message_count": s.message_count,
                "total_tokens": s.total_tokens,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        return self.store.delete_session(session_id)

    def get_session_messages(self, session_id: str) -> list[dict]:
        """Get all messages in a session (for display, not for LLM)."""
        session = self.store.get_session(session_id)
        if not session:
            return []
        return [
            {
                "id": m.id,
                "role": m.role.value,
                "content": m.content,
                "model": m.model,
                "provider": m.provider,
                "tokens_used": m.tokens_used,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in session.messages
            if m.role != MessageRole.SYSTEM  # Don't expose system prompt
        ]
