from typing import List, Dict
import numpy as np
import json

class Vectorstore:
    
    def store_faiss(self, chunks: List[Dict], index_path:str = "data/faiss_index"):
        import faiss
        import pickle
        
        chunks = [c for c in chunks if c.get('embedding') and len(c['embedding']) > 0]
        
        embeddings = np.array([c['embedding'] for c in chunks]).astype('float32')
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        faiss.write_index(index,f"{index_path}.index")
        
        with open(f"{index_path}_metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)
        
        return index
    
    def search_faiss(self, query_embedding: List[float], index_path:str = "data/faiss_index", top_k: int = 5) -> List[Dict]:
        
        import faiss
        import pickle
        
        index =  faiss.read_index(f"{index_path}.index")
        with open(f"{index_path}_metadata.pkl", "rb") as f:
            chunks = pickle.load(f)
            
        query = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({**chunks[idx], 'score': float(dist)})
        return results
    
    def store_chroma(self, chunks: List[Dict], collection_name: str = "rag_collection"):
        
        import chromadb
        
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_or_create_collection(name=collection_name)
        
        collection.add(
            ids=[str(c['chunk_id']) for c in chunks],
            embeddings=[c['embedding'] for c in chunks],
            documents=[c['content'] for c in chunks],
            metadatas=[{'source':c['source'], 'type': c['type']} for c in chunks]
        )
        
        return collection
    
    def search_chroma(self, query_embedding: List[float], collection_name: str = "rag_collection", top_k: int = 5) -> List[Dict]:
        
        import chromadb
        
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [{
            'content': doc,
            'metadata': meta,
            'score': dist
        } for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]
        
        
    def store_pinecone(self, chunks: List[Dict], index_name:str = "rag_index"):
        
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone()
        dimension = len(chunks[0]['embedding'])
        
        if index_name not in pc.list_indexes():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                #serverless=ServerlessSpec(min_replicas=1, max_replicas=3)
                serverless = ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        index = pc.Index(index_name)
        
        vectors = [{
            "id": str(c['chunk_id']),
            "values": c['embedding'],
            'metadata' : {'source': c['source'], 'content': c['content']}
        } for c in chunks]
        
        for i in range(0, len(vectors), 100):
            index.upsert(vectors[i:i+100])
            
        return index
    
    def search_pinecone(self, query_embedding: List[float], index_name:str = "rag_index", top_k: int = 5) -> List[Dict]:
        
        from pinecone import Pinecone
        
        pc = Pinecone()
        index = pc.Index(index_name)
        
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [{
            'content': match['metadata']['content'],
            'source': match['metadata']['source'],
            'score': match['score']
        } for match in response['matches']]
        
    def store_weaviate(self, chunks: List[Dict], class_name:str = "Document"):
        import weaviate
        
        client = weaviate.Client("http://localhost:8080")
        
        with client.batch as batch:
            for c in chunks:
                batch.add_data_object(
                    data_onject={
                        'content': c['content'],
                        'source': c['source']
                    },
                    class_name=class_name,
                    vector=c['embedding']
                )
        return client
    
    def search_weaviate(self, query_embedding: List[float], class_name:str = "Document", top_k: int = 5) -> List[Dict]:
        
        import weaviate
        
        client = weaviate.Client("http://localhost:8080")
        
        
        result = client.query.get(class_name, ["content", "source"]).with_near_vector({
            "vector": query_embedding
        }).with_limit(top_k).with_additional("distance").do()
        
        return result['data']['Get'][class_name]
    
    