from typing import List, Dict
import numpy as np

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
    
    # --- Main Search Method ---
    def search(self, query_embedding: List[float], chunks: List[Dict] = None, method: str = "cosine", top_k: int = 5, **kwargs) -> List[Dict]:
        """Unified search across all methods"""
        
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
