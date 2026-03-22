from typing import List, Dict
import numpy as np

class EmbeddingGenerator:
    
    def embed_openai(self, text: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(inputs=text, model = model)
        return [item.embedding for item in response.data]
    
    def embed_huggingface(self, texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts).tolist()
    
    def embed_bedrock(self, texts: List[str], model_id:str = "amazon.titan-embed-text-v1") -> List[List[float]]:
        
        import boto3, json
        client = boto3.client('bedrock-runtime')
        embeddings = []
        for text in texts:
            response = client.invoke_model(
                modelId=model_id,
                body = json.dumps({"inputText": text})
                
            )
            
            result = json.loads(response['body'].read())
            embeddings.append(result['embedding'])
            return embeddings
    
    def embed_cohere(self, texts: List[str], model: str = "embed-english-v3.0") -> List[List[float]]:
        
        import cohere
        client = cohere.Client()
        response = client.embed(texts=texts, model=model, input_type="search_document")
        return response.embeddings
    
    def embed_tfidf(self, texts: List[str]) -> List[List[float]]:
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(texts)
        self._tfidf_vectorizer = vectorizer
        return embeddings.toarray().tolist()
    
    def embed_bm25(self, texts: List[str]) -> List[List[float]]:
        """BM25 doesn't produce embeddings - it's a scoring function.
        Store tokenized docs for later BM25 search in similarity_search.py.
        Returns dummy embeddings (empty) since BM25 scores at query time."""
        self._bm25_corpus = [t.lower().split() for t in texts]
        return [[] for _ in texts]  # no embeddings needed
    
    def generate_embeddings(self, chunks: List[Dict], provider: str = "tfidf", batch_size: int = 32, **kwargs) -> List[Dict]:
        
        # Extract the text content from each chunk dictionary to create a list of strings for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Create a mapping dictionary that associates provider names with their corresponding embedding methods
        # This allows dynamic selection of the embedding function based on the provider parameter
        embed_fn = {
            'openai': self.embed_openai,
            'huggingface': self.embed_huggingface,
            'cohere': self.embed_cohere,
            'tfidf': self.embed_tfidf,
            'bedrock': self.embed_bedrock,
            'bm25': self.embed_bm25
        }[provider]
        
        if provider in ('tfidf', 'bm25'):
            all_embeddings = embed_fn(texts, **kwargs)
        else:
            # Initialize an empty list to store all computed embeddings
            all_embeddings = []
            # Process texts in batches to manage memory usage and API rate limits
            # Loop through the texts list in chunks of size 'batch_size'
            for i in range(0, len(texts), batch_size):
                # Extract a batch of texts starting at index i with maximum size of batch_size
                batch = texts[i:i+batch_size]
                # Call the selected embedding function on the current batch and extend the results to all_embeddings
                # **kwargs allows passing additional parameters specific to each embedding provider
                all_embeddings.extend(embed_fn(batch, **kwargs))
        
        
            
        # Combine the original chunks with their corresponding embeddings
        # Iterate through both chunks and embeddings simultaneously using zip
        for chunk, embedding in zip(chunks, all_embeddings):
            # Add the computed embedding as a new key-value pair to each chunk dictionary
            chunk['embedding'] = embedding
            
        return chunks
    
        