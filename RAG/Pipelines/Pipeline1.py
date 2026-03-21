import time
from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor
from TextChunkingSplitting import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import Vectorstore
from query_engine import QueryEngine
from similarity_search import SimilaritySearch
from prompt_builder import PromptBuilder
from response_processor import ResponseProcessor
from evaluation_monitoring import RAGEvaluator, RAGMonitor
from cache_optimizer import RAGCache, RAGOptimizer

# ===== STEP 1: Initialize all components =====
print("=" * 60)
print("INITIALIZING RAG PIPELINE")
print("=" * 60)

collector = DataCollector()
preprocessor = DataPreprocessor()
chunker = TextChunker()
embedder = EmbeddingGenerator()
store = Vectorstore()
searcher = SimilaritySearch()
builder = PromptBuilder()
processor = ResponseProcessor()
evaluator = RAGEvaluator()
monitor = RAGMonitor()
cache = RAGCache()
optimizer = RAGOptimizer()

# ===== STEP 2: Collect Data =====
print("\n[Step 2] Collecting data...")
data = collector.collect_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
print(f"  Collected {len(data['content'])} characters from {data['source']}")

# ===== STEP 3: Clean & Preprocess =====
print("\n[Step 3] Cleaning & preprocessing...")
cleaned = preprocessor.preprocess(data)
print(f"  Cleaned text: {len(cleaned['content'])} characters")

# ===== STEP 5: Chunk Text =====
print("\n[Step 5] Chunking text...")
chunks = chunker.chunk_with_metadata(cleaned, method='words', chunk_size=200, overlap=50)
print(f"  Created {len(chunks)} chunks")

# ===== STEP 6: Generate Embeddings =====
print("\n[Step 6] Generating embeddings (TF-IDF)...")
chunks = embedder.generate_embeddings(chunks, provider="tfidf")
print(f"  Embedding dimension: {len(chunks[0]['embedding'])}")

# ===== STEP 7: Store in Vector DB =====
print("\n[Step 7] Storing in FAISS...")
store.store_faiss(chunks)
print("  Stored successfully!")

# ===== STEP 8: Query Embedding =====
query = "What is Artificial Intelligence?"
print(f"\n[Step 8] Query: '{query}'")

# Check cache first
cached = cache.get(query)
if cached:
    print("  Cache HIT!")
    print(f"  Answer: {cached}")
else:
    print("  Cache MISS - running pipeline...")
    start = time.time()

    query_engine = QueryEngine(embedder, provider="tfidf")
    query_vector = query_engine.get_query_embedding(query)
    print(f"  Query embedding dimension: {len(query_vector)}")

    # ===== STEP 9-10: Similarity Search + Top-K =====
    print("\n[Step 9-10] Similarity search (top 5)...")
    results = searcher.search(query_vector, method="faiss", top_k=5)
    for i, r in enumerate(results):
        print(f"  [{i+1}] Score: {r['score']:.4f} | {r['content'][:80]}...")

    # ===== OPTIMIZE: Deduplicate, Rerank, Compress =====
    print("\n[Optimize] Dedup → Rerank → Compress...")
    results = optimizer.deduplicate_chunks(results)
    results = optimizer.rerank_chunks(query, results)
    results = optimizer.compress_context(results, max_tokens=1000)
    print(f"  Optimized to {len(results)} chunks")

    # ===== STEP 11: Build Prompt =====
    print("\n[Step 11] Building prompt (sources style)...")
    prompt = builder.build_prompt(query, results, style="sources")
    print(f"  Prompt length: {len(prompt)} characters")

    # ===== STEP 12-13: Send to LLM & Generate =====
    # NOTE: Uncomment ONE provider below that you have access to
    print("\n[Step 12-13] Generating answer...")
    
    # --- Option A: OpenAI (needs OPENAI_API_KEY) ---
    # from llm_generator import LLMGenerator
    # llm = LLMGenerator()
    # answer = llm.generate(prompt, provider="openai")
    
    # --- Option B: Ollama (needs local ollama running) ---
    # from llm_generator import LLMGenerator
    # llm = LLMGenerator()
    # answer = llm.generate(prompt, provider="ollama")
    
    # --- Option C: No LLM - just show retrieved context ---
    answer = "LLM not configured. Top retrieved context:\n" + results[0]['content'][:300]
    
    print(f"  Answer: {answer[:200]}...")

    # ===== STEP 14: Post-process =====
    print("\n[Step 14] Post-processing...")
    final = processor.process(answer, query, results, output_format="json")
    print(f"  Format: JSON | Confidence: {final['confidence']}")

    # ===== STEP 15: Evaluate & Monitor =====
    latency = time.time() - start
    evaluation = evaluator.evaluate(query, answer, results)
    monitor.track_query(query, answer, results, latency, evaluation)
    print(f"\n[Step 15] Evaluation:")
    print(f"  Relevance:    {evaluation['relevance']}")
    print(f"  Groundedness: {evaluation['groundedness']}")
    print(f"  Coverage:     {evaluation['coverage']}")
    print(f"  Latency:      {latency:.2f}s")

    # ===== STEP 16: Cache =====
    cache.set(query, answer)
    print("\n[Step 16] Result cached!")

    # Save logs
    monitor.save_logs()

print("\n" + "=" * 60)
print("RAG PIPELINE COMPLETE!")
print("=" * 60)
