from typing import List, Dict
from embedding_generator import EmbeddingGenerator

class QueryEngine:
    
    def __init__(self, embedder: EmbeddingGenerator, provider: str = "tfidf"):
        self.embedder = embedder
        self.provider = provider
    
    def get_query_embedding_tfidf(self, query: str) -> List[float]:
        """Query embedding using stored TF-IDF vectorizer"""
        return self.embedder._tfidf_vectorizer.transform([query]).toarray()[0].tolist()
    
    def get_query_embedding_huggingface(self, query: str) -> List[float]:
        """Query embedding using HuggingFace"""
        return self.embedder.embed_huggingface([query])[0]
    
    def get_query_embedding_openai(self, query: str) -> List[float]:
        """Query embedding using OpenAI"""
        return self.embedder.embed_openai([query])[0]
    
    def get_query_embedding_bedrock(self, query: str) -> List[float]:
        """Query embedding using AWS Bedrock"""
        return self.embedder.embed_bedrock([query])[0]
    
    def get_query_embedding_cohere(self, query: str) -> List[float]:
        """Query embedding using Cohere"""
        return self.embedder.embed_cohere([query])[0]
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding using configured provider"""
        fn = {
            'tfidf': self.get_query_embedding_tfidf,
            'huggingface': self.get_query_embedding_huggingface,
            'openai': self.get_query_embedding_openai,
            'bedrock': self.get_query_embedding_bedrock,
            'cohere': self.get_query_embedding_cohere
        }[self.provider]
        return fn(query)
