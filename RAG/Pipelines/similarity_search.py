from typing import List, Dict
import numpy as np
import math
from collections import Counter

class SimilaritySearch:
    
    # --- Cosine Similarity (Manual) ---
    def cosine_similarity(self, query: List[float], embeddings: List[List[float]]) -> List[float]:
        """Compute cosine similarity between query and all embeddings"""
        query = np.array(query)
        embeddings = np.array(embeddings)
        dot_products = np.dot(embeddings, query)
        query_norm = np.linalg.norm(query)
        embed_norms = np.linalg.norm(embeddings, axis=1)
        return (dot_products / (query_norm * embed_norms)).tolist()
    
    # --- Euclidean Distance (Manual) ---
    def euclidean_distance(self, query: List[float], embeddings: List[List[float]]) -> List[float]:
        """Compute euclidean distance between query and all embeddings"""
        query = np.array(query)
        embeddings = np.array(embeddings)
        return np.linalg.norm(embeddings - query, axis=1).tolist()
    
    # --- Dot Product (Manual) ---
    def dot_product(self, query: List[float], embeddings: List[List[float]]) -> List[float]:
        """Compute dot product similarity"""
        query = np.array(query)
        embeddings = np.array(embeddings)
        return np.dot(embeddings, query).tolist()
    
    # --- FAISS Search ---
    def search_faiss(self, query: List[float], index_path: str = "data/faiss_index", top_k: int = 5) -> List[Dict]:
        """Search using FAISS"""
        import faiss, pickle
        
        index = faiss.read_index(f"{index_path}.index")
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            chunks = pickle.load(f)
        
        query_vec = np.array([query]).astype('float32')
        distances, indices = index.search(query_vec, top_k)
        
        return [
            {**chunks[idx], 'score': float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]
    
    # --- ChromaDB Search ---
    def search_chroma(self, query: List[float], collection_name: str = "rag_collection", top_k: int = 5) -> List[Dict]:
        """Search using ChromaDB"""
        import chromadb
        
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection(name=collection_name)
        results = collection.query(query_embeddings=[query], n_results=top_k)
        
        return [
            {'content': doc, 'metadata': meta, 'score': dist}
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
        ]
    
    # --- BM25 Search (Keyword-based, no embeddings needed) ---
    def search_bm25(self, query: str, chunks: List[Dict], top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Dict]:
        """
        BM25 (Best Matching 25) - sparse keyword-based ranking
        Unlike cosine/euclidean, BM25 works on raw text (no embeddings needed)
        
        Formula: score(D,Q) = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * |D|/avgdl))
        
        Parameters:
            k1: term frequency saturation (1.2-2.0 typical, higher = more weight to repeated terms)
            b:  length normalization (0=no normalization, 1=full normalization, 0.75 typical)
        """
        # Tokenize all chunk documents
        docs = [c['content'].lower().split() for c in chunks]
        query_terms = query.lower().split()
        
        # Compute average document length
        avgdl = sum(len(d) for d in docs) / len(docs) if docs else 1
        N = len(docs)
        
        # Compute IDF for each query term: log((N - df + 0.5) / (df + 0.5) + 1)
        df = Counter()  # document frequency: how many docs contain each term
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        
        idf = {}
        for term in query_terms:
            n = df.get(term, 0)
            idf[term] = math.log((N - n + 0.5) / (n + 0.5) + 1)
        
        # Score each document
        scores = []
        for doc in docs:
            score = 0.0
            doc_len = len(doc)
            tf_counter = Counter(doc)
            for term in query_terms:
                tf = tf_counter.get(term, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score += idf.get(term, 0) * (numerator / denominator)
            scores.append(score)
        
        scored_chunks = [{**c, 'score': s} for c, s in zip(chunks, scores)]
        return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
    
    # --- Main Search Method ---
    def search(self, query_embedding: List[float] = None, chunks: List[Dict] = None, method: str = "cosine", top_k: int = 5, query: str = None, **kwargs) -> List[Dict]:
        """Unified search across all methods"""
        
        # BM25 (keyword-based, no embeddings needed)
        if method == "bm25":
            if not query:
                raise ValueError("BM25 requires 'query' (raw text string), not embeddings")
            return self.search_bm25(query, chunks, top_k=top_k, **kwargs)
        
        # Vector DB searches
        if method == "faiss":
            return self.search_faiss(query_embedding, top_k=top_k, **kwargs)
        if method == "chroma":
            return self.search_chroma(query_embedding, top_k=top_k, **kwargs)
        
        # Manual similarity searches (need chunks with embeddings)
        embeddings = [c['embedding'] for c in chunks]
        
        if method == "cosine":
            scores = self.cosine_similarity(query_embedding, embeddings)
            scored_chunks = [
                {**c, 'score': s} for c, s in zip(chunks, scores)
            ]
            return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
        
        elif method == "euclidean":
            scores = self.euclidean_distance(query_embedding, embeddings)
            scored_chunks = [
                {**c, 'score': s} for c, s in zip(chunks, scores)
            ]
            return sorted(scored_chunks, key=lambda x: x['score'])[:top_k]  # Lower = better
        
        elif method == "dot_product":
            scores = self.dot_product(query_embedding, embeddings)
            scored_chunks = [
                {**c, 'score': s} for c, s in zip(chunks, scores)
            ]
            return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]



## example usage of similarity search in a pipeline
# from embedding_generator import EmbeddingGenerator
# from TextChunkingSplitting import TextChunker
# from dataCollection import DataCollector
# from dataClean_Processing import DataPreprocessor
# from query_engine import QueryEngine
# from similarity_search import SimilaritySearch

# collector = DataCollector()
# preprocessor = DataPreprocessor()
# chunker = TextChunker()
# embedder = EmbeddingGenerator()
# searcher = SimilaritySearch()

# # Collect → Clean → Chunk → Embed
# data = collector.collect_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
# cleaned = preprocessor.preprocess(data)
# chunks = chunker.chunk_with_metadata(cleaned)
# chunks = embedder.generate_embeddings(chunks, provider="tfidf")

# # Query
# query_engine = QueryEngine(embedder, provider="tfidf")
# query_vector = query_engine.get_query_embedding("What is Artificial Intelligence?")

# # Search with different methods
# print("=== Cosine Similarity ===")
# results = searcher.search(query_vector, chunks, method="cosine", top_k=3)
# for r in results:
#     print(f"Score: {r['score']:.4f} | {r['content'][:100]}...")

# print("\n=== Euclidean Distance ===")
# results = searcher.search(query_vector, chunks, method="euclidean", top_k=3)
# for r in results:
#     print(f"Score: {r['score']:.4f} | {r['content'][:100]}...")

# print("\n=== FAISS ===")
# from vector_store import Vectorstore
# store = Vectorstore()
# store.store_faiss(chunks)
# results = searcher.search(query_vector, method="faiss", top_k=3)
# for r in results:
#     print(f"Score: {r['score']:.4f} | {r['content'][:100]}...")
