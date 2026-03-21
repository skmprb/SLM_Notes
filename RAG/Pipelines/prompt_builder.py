from typing import List, Dict

class PromptBuilder:
    
    def build_simple(self, query: str, chunks: List[Dict]) -> str:
        """Basic prompt with context"""
        context = "\n\n".join([c['content'] for c in chunks])
        return f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""
    
    def build_with_sources(self, query: str, chunks: List[Dict]) -> str:
        """Prompt with numbered source references"""
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}]: {c['content']}")
        context = "\n\n".join(context_parts)
        
        return f"""Answer the question based on the provided sources. Cite sources using [Source N].

{context}

Question: {query}

Answer:"""
    
    def build_strict(self, query: str, chunks: List[Dict]) -> str:
        """Strict prompt - only answer from context"""
        context = "\n\n".join([c['content'] for c in chunks])
        return f"""Answer the question ONLY using the context below. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    
    def build_conversational(self, query: str, chunks: List[Dict], chat_history: List[Dict] = None) -> str:
        """Prompt with chat history"""
        context = "\n\n".join([c['content'] for c in chunks])
        history = ""
        if chat_history:
            history = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
            history = f"\nChat History:\n{history}\n"
        
        return f"""You are a helpful assistant. Use the context to answer the question.

Context:
{context}
{history}
Question: {query}

Answer:"""
    
    def build_custom(self, query: str, chunks: List[Dict], system_prompt: str = "") -> str:
        """Custom system prompt"""
        context = "\n\n".join([c['content'] for c in chunks])
        return f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
    
    def build_prompt(self, query: str, chunks: List[Dict], style: str = "simple", **kwargs) -> str:
        """Unified prompt builder"""
        fn = {
            'simple': self.build_simple,
            'sources': self.build_with_sources,
            'strict': self.build_strict,
            'conversational': self.build_conversational,
            'custom': self.build_custom
        }[style]
        return fn(query, chunks, **kwargs)
