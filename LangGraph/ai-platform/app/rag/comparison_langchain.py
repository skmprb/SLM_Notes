"""
Phase 10: RAG — LangChain Comparison

| Our Code                  | LangChain Equivalent                          |
|---------------------------|-----------------------------------------------|
| Chunker                   | RecursiveCharacterTextSplitter                |
| BaseEmbedding             | Embeddings (base class)                       |
| OpenAIEmbedding           | OpenAIEmbeddings                              |
| InMemoryVectorStore       | FAISS / Chroma / Pinecone vectorstore         |
| RetrievalService.retrieve | vectorstore.similarity_search()               |
| RetrievalService.ingest   | VectorStore.from_documents()                  |
| format_context()          | create_stuff_documents_chain()                |
| Full pipeline             | create_retrieval_chain(retriever, chain)       |
"""
