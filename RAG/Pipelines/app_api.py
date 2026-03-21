from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor
from TextChunkingSplitting import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import Vectorstore
from query_engine import QueryEngine
from similarity_search import SimilaritySearch
from prompt_builder import PromptBuilder
from llm_generator import LLMGenerator
from response_processor import ResponseProcessor
from evaluation_monitoring  import RAGEvaluator, RAGMonitor
from cache_optimizer import RAGCache, RAGOptimizer
import time

app = FastAPI(title="RAG Pipeline API")

# Initialize all components
collector = DataCollector()
preprocessor = DataPreprocessor()
chunker = TextChunker()
embedder = EmbeddingGenerator()
store = Vectorstore()
query_engine = QueryEngine(embedder, provider="tfidf")
searcher = SimilaritySearch()
builder = PromptBuilder()
llm = LLMGenerator()
processor = ResponseProcessor()
evaluator = RAGEvaluator()
monitor = RAGMonitor()
cache = RAGCache()
optimizer = RAGOptimizer()

class IngestRequest(BaseModel):
    source: str  # URL, file path
    source_type: str  # "url", "pdf", "text"

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    prompt_style: Optional[str] = "simple"
    llm_provider: Optional[str] = "openai"

@app.post("/ingest")
def ingest(req: IngestRequest):
    """Ingest a data source into the pipeline"""
    if req.source_type == "url":
        data = collector.collect_url(req.source)
    elif req.source_type == "pdf":
        data = collector.collect_pdf(req.source)
    else:
        data = collector.collect_text_file(req.source)
    
    cleaned = preprocessor.preprocess(data)
    chunks = chunker.chunk_with_metadata(cleaned)
    chunks = embedder.generate_embeddings(chunks, provider="tfidf")
    store.store_faiss(chunks)
    
    return {"status": "success", "chunks_stored": len(chunks)}

@app.post("/query")
def query(req: QueryRequest):
    """Query the RAG pipeline"""
    # Check cache
    cached = cache.get(req.question)
    if cached:
        return {"answer": cached, "cached": True}
    
    start = time.time()
    
    # Retrieve
    query_vector = query_engine.get_query_embedding(req.question)
    results = searcher.search(query_vector, method="faiss", top_k=req.top_k)
    
    # Optimize
    results = optimizer.deduplicate_chunks(results)
    results = optimizer.rerank_chunks(req.question, results)
    results = optimizer.compress_context(results)
    
    # Generate
    prompt = builder.build_prompt(req.question, results, style=req.prompt_style)
    answer = llm.generate(prompt, provider=req.llm_provider)
    
    # Post-process
    final = processor.process(answer, req.question, results, output_format="json")
    
    # Evaluate & Monitor
    latency = time.time() - start
    evaluation = evaluator.evaluate(req.question, answer, results)
    monitor.track_query(req.question, answer, results, latency, evaluation)
    
    # Cache
    cache.set(req.question, final)
    
    return {**final, "latency": latency, "evaluation": evaluation}

@app.get("/stats")
def stats():
    """Get pipeline stats"""
    return monitor.get_stats()

@app.post("/clear-cache")
def clear_cache():
    """Clear cache"""
    cache.clear()
    return {"status": "cache cleared"}
