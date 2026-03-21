import re
from typing import List, Dict

class ResponseProcessor:
    
    def clean_response(self, response: str) -> str:
        """Basic cleanup"""
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        return response.strip()
    
    def add_sources(self, response: str, chunks: List[Dict]) -> str:
        """Append source references"""
        sources = set(c.get('source', 'Unknown') for c in chunks)
        source_text = "\n".join(f"- {s}" for s in sources)
        return f"{response}\n\nSources:\n{source_text}"
    
    def format_markdown(self, response: str) -> str:
        """Format as markdown"""
        return f"## Answer\n\n{response}\n"
    
    def format_json(self, response: str, query: str, chunks: List[Dict]) -> Dict:
        """Format as structured JSON"""
        return {
            "query": query,
            "answer": response,
            "sources": list(set(c.get('source', '') for c in chunks)),
            "num_chunks_used": len(chunks),
            "confidence": self._estimate_confidence(chunks)
        }
    
    def truncate(self, response: str, max_length: int = 500) -> str:
        """Truncate response to max length"""
        if len(response) <= max_length:
            return response
        return response[:max_length].rsplit(' ', 1)[0] + "..."
    
    def remove_hallucination_phrases(self, response: str) -> str:
        """Remove common LLM filler phrases"""
        patterns = [
            r"^(As an AI language model,?\s*)",
            r"^(Based on the (provided )?context,?\s*)",
            r"^(According to the (given )?information,?\s*)",
            r"^(I think\s*)",
        ]
        for p in patterns:
            response = re.sub(p, '', response, flags=re.IGNORECASE)
        return response.strip()
    
    def _estimate_confidence(self, chunks: List[Dict]) -> str:
        """Estimate confidence based on chunk scores"""
        scores = [c.get('score', 0) for c in chunks]
        if not scores:
            return "low"
        avg = sum(scores) / len(scores)
        if avg > 0.8:
            return "high"
        elif avg > 0.5:
            return "medium"
        return "low"
    
    def process(self, response: str, query: str, chunks: List[Dict], output_format: str = "text", **kwargs) -> any:
        """Unified post-processing"""
        response = self.clean_response(response)
        response = self.remove_hallucination_phrases(response)
        
        if output_format == "json":
            return self.format_json(response, query, chunks)
        elif output_format == "markdown":
            return self.format_markdown(self.add_sources(response, chunks))
        else:
            return self.add_sources(response, chunks)




## Usage

# from response_processor import ResponseProcessor

# processor = ResponseProcessor()

# # After LLM generates raw answer
# result = processor.process(answer, query, results, output_format="json")
# print(result)

# # Or markdown with sources
# result = processor.process(answer, query, results, output_format="markdown")
# print(result)
