"""
Vector Store: ChromaDB with strict user_id isolation.
- Single collection, metadata-filtered per user
- user_id filter is ALWAYS injected server-side (never trusted from client)
- Background ingestion pipeline: chunk → embed → upsert
"""
import uuid
from typing import AsyncIterator

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agents.state import DocPayload
from app.config.settings import get_settings

settings = get_settings()

_chroma_client: chromadb.AsyncHttpClient | None = None
_embeddings: OpenAIEmbeddings | None = None


def get_chroma() -> chromadb.AsyncHttpClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _chroma_client


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings


def _chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


async def ingest_document(
    user_id: str,
    doc_id: str,
    payload: DocPayload,
) -> None:
    """
    Background pipeline: extract chunks → embed → upsert to ChromaDB.
    All chunks tagged with user_id + doc_id for isolation.
    """
    client = get_chroma()
    collection = await client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    embedder = get_embeddings()

    chunks: list[str] = []
    metadatas: list[dict] = []

    # Text chunks
    for i, chunk in enumerate(_chunk_text(payload["text"])):
        chunks.append(chunk)
        metadatas.append({
            "user_id": user_id,
            "doc_id": doc_id,
            "doc_name": payload["doc_name"],
            "type": "text",
            "chunk_index": i,
        })

    # Table chunks (each table as one chunk — preserve structure)
    for i, table in enumerate(payload["tables"]):
        chunks.append(table)
        metadatas.append({
            "user_id": user_id,
            "doc_id": doc_id,
            "doc_name": payload["doc_name"],
            "type": "table",
            "chunk_index": i,
        })

    if not chunks:
        return

    vectors = await embedder.aembed_documents(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    await collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )


async def retrieve(
    user_id: str,
    query: str,
    doc_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks for a user.
    user_id filter is ALWAYS applied — guaranteed isolation.
    Optionally filter by doc_id for session-specific retrieval.
    """
    client = get_chroma()
    collection = await client.get_or_create_collection(name=settings.chroma_collection)
    embedder = get_embeddings()

    query_vector = await embedder.aembed_query(query)

    # Build where filter — user_id always enforced
    where: dict = {"user_id": {"$eq": user_id}}
    if doc_id:
        where = {"$and": [{"user_id": {"$eq": user_id}}, {"doc_id": {"$eq": doc_id}}]}

    results = await collection.query(
        query_embeddings=[query_vector],
        n_results=top_k or settings.retrieval_top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "content": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),    # cosine similarity
        })

    return chunks
