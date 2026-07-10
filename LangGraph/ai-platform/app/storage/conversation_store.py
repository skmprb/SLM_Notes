"""
Conversation Store - Persistence layer for sessions and messages.

Two implementations:
    1. InMemoryStore: For development/testing (data lost on restart)
    2. SQLiteStore: For local persistence (survives restarts)

In production, you'd add:
    - PostgresStore
    - RedisStore (for fast access + TTL)
    - DynamoDBStore (for serverless)

LangChain equivalent:
    - LangGraph Checkpointer (persists graph state per thread)
    - InMemoryStore / PostgresStore / SQLiteStore
    - The key difference: LangGraph stores GRAPH STATE, we store MESSAGES
"""

from abc import ABC, abstractmethod
from typing import Optional
from app.storage.models import Session, Message
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseConversationStore(ABC):
    """Abstract interface for conversation persistence."""

    @abstractmethod
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create a new conversation session."""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        ...

    @abstractmethod
    def save_session(self, session: Session) -> None:
        """Persist a session (create or update)."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        ...

    @abstractmethod
    def list_sessions(self, limit: int = 20, offset: int = 0) -> list[Session]:
        """List sessions, most recent first."""
        ...


class InMemoryStore(BaseConversationStore):
    """
    In-memory conversation store. Fast but not persistent.

    Use for:
        - Development
        - Testing
        - Short-lived sessions (serverless functions)
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        logger.info("InMemoryStore initialized")

    def create_session(self, session_id: Optional[str] = None) -> Session:
        session = Session(id=session_id) if session_id else Session()
        self._sessions[session.id] = session
        logger.info(f"Session created | id={session.id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def save_session(self, session: Session) -> None:
        self._sessions[session.id] = session

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self, limit: int = 20, offset: int = 0) -> list[Session]:
        sessions = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        return sessions[offset:offset + limit]


class SQLiteStore(BaseConversationStore):
    """
    SQLite-based conversation store. Persistent across restarts.

    Use for:
        - Local development with persistence
        - Single-server deployments
        - Prototyping before moving to PostgreSQL
    """

    def __init__(self, db_path: str = "conversations.db"):
        import sqlite3
        import json

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        logger.info(f"SQLiteStore initialized | db={db_path}")

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                total_tokens INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                model TEXT,
                provider TEXT,
                tokens_used INTEGER,
                latency_ms REAL,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self.conn.commit()

    def create_session(self, session_id: Optional[str] = None) -> Session:
        session = Session(id=session_id) if session_id else Session()
        self.conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, total_tokens, message_count) VALUES (?, ?, ?, ?, ?, ?)",
            (session.id, session.title, session.created_at.isoformat(), session.updated_at.isoformat(), 0, 0),
        )
        self.conn.commit()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        from app.storage.models import MessageRole
        from datetime import datetime

        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None

        session = Session(
            id=row[0], title=row[1],
            created_at=datetime.fromisoformat(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
            total_tokens=row[4], message_count=row[5],
        )

        # Load messages
        msg_rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,)
        ).fetchall()

        for mr in msg_rows:
            session.messages.append(Message(
                id=mr[0], role=MessageRole(mr[2]), content=mr[3],
                model=mr[4], provider=mr[5], tokens_used=mr[6],
                latency_ms=mr[7], timestamp=datetime.fromisoformat(mr[8]),
            ))

        return session

    def save_session(self, session: Session) -> None:
        self.conn.execute(
            "REPLACE INTO sessions (id, title, created_at, updated_at, total_tokens, message_count) VALUES (?, ?, ?, ?, ?, ?)",
            (session.id, session.title, session.created_at.isoformat(),
             session.updated_at.isoformat(), session.total_tokens, session.message_count),
        )
        # Save new messages (upsert)
        for msg in session.messages:
            self.conn.execute(
                "REPLACE INTO messages (id, session_id, role, content, model, provider, tokens_used, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (msg.id, session.id, msg.role.value, msg.content, msg.model,
                 msg.provider, msg.tokens_used, msg.latency_ms, msg.timestamp.isoformat()),
            )
        self.conn.commit()

    def delete_session(self, session_id: str) -> bool:
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        result = self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()
        return result.rowcount > 0

    def list_sessions(self, limit: int = 20, offset: int = 0) -> list[Session]:
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at, total_tokens, message_count FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        from datetime import datetime
        return [
            Session(id=r[0], title=r[1], created_at=datetime.fromisoformat(r[2]),
                    updated_at=datetime.fromisoformat(r[3]), total_tokens=r[4], message_count=r[5])
            for r in rows
        ]
