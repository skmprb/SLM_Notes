"""
RAG - Retrieval-Augmented Generation (Phase 10)

Pipeline: Ingest → Chunk → Embed → Store → Retrieve → Rerank → Inject into prompt

Production concerns:
    - Chunking strategy (size, overlap)
    - Embedding model abstraction
    - Vector store abstraction (swap Pinecone/Chroma/FAISS)
    - Hybrid search (semantic + keyword)
    - Reranking for precision
    - Citation tracking
    - Token budget for context

LangChain equivalent:
    - RecursiveCharacterTextSplitter → our Chunker
    - OpenAIEmbeddings → our EmbeddingService
    - FAISS/Chroma/Pinecone → our VectorStore
    - create_retrieval_chain() → our RetrievalService
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import hashlib

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """A document chunk with metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievalResult:
    """A retrieved document with relevance score."""
    document: Document
    score: float
    rank: int


# ============================================================
# CHUNKER
# ============================================================

class Chunker:
    """Split documents into chunks with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(Document(
                content=chunk_text,
                metadata={**(metadata or {}), "chunk_index": len(chunks)},
            ))
            start += self.chunk_size - self.chunk_overlap
        logger.info(f"Chunked text into {len(chunks)} chunks | size={self.chunk_size} overlap={self.chunk_overlap}")
        return chunks


# ============================================================
# EMBEDDING SERVICE (Abstract)
# ============================================================

class BaseEmbedding(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        ...


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embeddings (text-embedding-3-small)."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from app.config.settings import get_settings
        self.model = model
        self.api_key = get_settings().openai_api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


# ============================================================
# VECTOR STORE (In-Memory for learning, swap for Pinecone/Chroma in prod)
# ============================================================

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]: ...


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        self._documents: list[Document] = []

    def add(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        logger.info(f"Added {len(documents)} docs to vector store | total={len(self._documents)}")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Cosine similarity search."""
        scored = []
        for doc in self._documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(scored[:top_k])
        ]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# RETRIEVAL SERVICE (Orchestrates the full pipeline)
# ============================================================

class RetrievalService:
    """Orchestrates: query → embed → search → format context."""

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.embedding = embedding or OpenAIEmbedding()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.chunker = Chunker()

    def ingest(self, text: str, metadata: Optional[dict] = None) -> int:
        """Ingest a document: chunk → embed → store. Returns chunk count."""
        chunks = self.chunker.chunk(text, metadata)
        texts = [c.content for c in chunks]
        embeddings = self.embedding.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self.vector_store.add(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        query_embedding = self.embedding.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k)
        logger.info(f"Retrieved {len(results)} docs for query | top_score={results[0].score:.3f}" if results else "No results")
        return results

    def format_context(self, results: list[RetrievalResult], max_tokens: int = 2000) -> str:
        """Format retrieved docs into a context string for the prompt."""
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # rough token-to-char estimate

        for r in results:
            if total_chars + len(r.document.content) > char_limit:
                break
            context_parts.append(f"[Source {r.rank}]: {r.document.content}")
            total_chars += len(r.document.content)

        return "\n\n".join(context_parts)
