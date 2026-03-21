import hashlib
import json
import time
from typing import List, Dict, Any
from pathlib import Path
from functools import lru_cache

class RAGCache:
    
    def __init__(self, cache_dir: str = "data/cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.ttl = ttl  # Time to live in seconds
    
    def _hash_key(self, text: str) -> str:
        """Generate hash key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    # --- In-Memory Cache ---
    def get_memory(self, query: str) -> Any:
        """Get from memory cache"""
        key = self._hash_key(query)
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['value']
            del self.memory_cache[key]
        return None
    
    def set_memory(self, query: str, value: Any):
        """Set in memory cache"""
        key = self._hash_key(query)
        self.memory_cache[key] = {"value": value, "timestamp": time.time()}
    
    # --- Disk Cache ---
    def get_disk(self, query: str) -> Any:
        """Get from disk cache"""
        key = self._hash_key(query)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, 'r') as f:
                entry = json.load(f)
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['value']
            path.unlink()
        return None
    
    def set_disk(self, query: str, value: Any):
        """Set in disk cache"""
        key = self._hash_key(query)
        path = self.cache_dir / f"{key}.json"
        with open(path, 'w') as f:
            json.dump({"value": value, "timestamp": time.time()}, f)
    
    # --- Unified Cache ---
    def get(self, query: str) -> Any:
        """Check memory first, then disk"""
        result = self.get_memory(query)
        if result:
            return result
        result = self.get_disk(query)
        if result:
            self.set_memory(query, result)  # Promote to memory
        return result
    
    def set(self, query: str, value: Any):
        """Store in both memory and disk"""
        self.set_memory(query, value)
        self.set_disk(query, value)
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


class RAGOptimizer:
    
    def deduplicate_chunks(self, chunks: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """Remove near-duplicate chunks"""
        unique = []
        seen = set()
        for c in chunks:
            words = frozenset(c['content'].lower().split())
            is_dup = False
            for s in seen:
                overlap = len(words & s) / max(len(words | s), 1)
                if overlap > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
                seen.add(words)
        return unique
    
    def rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Simple keyword-based reranking"""
        query_words = set(query.lower().split())
        for c in chunks:
            chunk_words = set(c['content'].lower().split())
            c['rerank_score'] = len(query_words & chunk_words) / len(query_words)
        return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    def compress_context(self, chunks: List[Dict], max_tokens: int = 2000) -> List[Dict]:
        """Trim chunks to fit token budget"""
        result = []
        total = 0
        for c in chunks:
            token_est = len(c['content'].split())
            if total + token_est > max_tokens:
                remaining = max_tokens - total
                c['content'] = ' '.join(c['content'].split()[:remaining])
                result.append(c)
                break
            result.append(c)
            total += token_est
        return result







## usage example:
# from cache_optimizer import RAGCache, RAGOptimizer

# cache = RAGCache(ttl=3600)  # 1 hour TTL
# optimizer = RAGOptimizer()

# query = "What is Artificial Intelligence?"

# # Check cache first
# cached = cache.get(query)
# if cached:
#     print("Cache hit!")
#     answer = cached
# else:
#     # ... run RAG pipeline ...
    
#     # Optimize before sending to LLM
#     results = optimizer.deduplicate_chunks(results)
#     results = optimizer.rerank_chunks(query, results)
#     results = optimizer.compress_context(results, max_tokens=2000)
    
#     # Generate answer
#     prompt = builder.build_prompt(query, results, style="sources")
#     answer = llm.generate(prompt, provider="openai")
    
#     # Cache the result
#     cache.set(query, answer)
#     print("Cached for next time!")

# print(answer)
