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
        return embeddings.toarray().tolist()
    
    def generate_embeddings(self, chunks: List[Dict], provider: str = "tfidf", batch_size: int = 32, **kwargs) -> List[Dict]:
        
        texts = [chunk['content'] for chunk in chunks]
        
        embed_fn = {
            'openai': self.embed_openai,
            'huggingface': self.embed_huggingface,
            'cohere': self.embed_cohere,
            'tfidf': self.embed_tfidf,
            'bedrock': self.embed_bedrock
        }[provider]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            all_embeddings.extend(embed_fn(batch, **kwargs))
            
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding
            
        return chunks
    
        