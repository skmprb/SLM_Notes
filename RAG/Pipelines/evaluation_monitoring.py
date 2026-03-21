import time
import json
import logging
from typing import List, Dict
from datetime import datetime
from pathlib import Path

logging.basicConfig(filename="data/rag_monitor.log", level=logging.INFO, format="%(asctime)s | %(message)s")

class RAGEvaluator:
    
    def relevance_score(self, query: str, chunks: List[Dict]) -> float:
        """Check keyword overlap between query and retrieved chunks"""
        query_words = set(query.lower().split())
        chunk_text = " ".join(c['content'] for c in chunks).lower()
        chunk_words = set(chunk_text.split())
        overlap = query_words & chunk_words
        return len(overlap) / len(query_words) if query_words else 0.0
    
    def answer_groundedness(self, answer: str, chunks: List[Dict]) -> float:
        """Check how much of the answer is grounded in context"""
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(c['content'] for c in chunks).lower().split())
        grounded = answer_words & context_words
        return len(grounded) / len(answer_words) if answer_words else 0.0
    
    def chunk_coverage(self, answer: str, chunks: List[Dict]) -> Dict:
        """Check which chunks contributed to the answer"""
        used = []
        for i, c in enumerate(chunks):
            keywords = set(c['content'].lower().split()[:20])
            answer_words = set(answer.lower().split())
            if keywords & answer_words:
                used.append(i)
        return {
            "total_chunks": len(chunks),
            "chunks_used": len(used),
            "coverage": len(used) / len(chunks) if chunks else 0.0
        }
    
    def evaluate(self, query: str, answer: str, chunks: List[Dict]) -> Dict:
        """Full evaluation"""
        return {
            "relevance": round(self.relevance_score(query, chunks), 3),
            "groundedness": round(self.answer_groundedness(answer, chunks), 3),
            "coverage": self.chunk_coverage(answer, chunks)
        }


class RAGMonitor:
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
    
    def track_query(self, query: str, answer: str, chunks: List[Dict], latency: float, evaluation: Dict) -> Dict:
        """Log a single RAG query"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:200],
            "num_chunks": len(chunks),
            "latency_seconds": round(latency, 3),
            "evaluation": evaluation
        }
        self.history.append(record)
        logging.info(json.dumps(record))
        return record
    
    def get_stats(self) -> Dict:
        """Get aggregate stats"""
        if not self.history:
            return {"message": "No queries tracked yet"}
        
        latencies = [r['latency_seconds'] for r in self.history]
        relevances = [r['evaluation']['relevance'] for r in self.history]
        groundedness = [r['evaluation']['groundedness'] for r in self.history]
        
        return {
            "total_queries": len(self.history),
            "avg_latency": round(sum(latencies) / len(latencies), 3),
            "avg_relevance": round(sum(relevances) / len(relevances), 3),
            "avg_groundedness": round(sum(groundedness) / len(groundedness), 3)
        }
    
    def save_logs(self, filename: str = "query_logs.json"):
        """Save all logs to file"""
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        return str(path)


## usage example

# import time
# from evaluation_monitor import RAGEvaluator, RAGMonitor

# evaluator = RAGEvaluator()
# monitor = RAGMonitor()

# # Track latency
# start = time.time()

# # ... your RAG pipeline runs here (query → retrieve → generate) ...

# latency = time.time() - start

# # Evaluate
# evaluation = evaluator.evaluate(query, answer, results)
# print(evaluation)
# # {'relevance': 0.85, 'groundedness': 0.72, 'coverage': {'total_chunks': 5, 'chunks_used': 3, 'coverage': 0.6}}

# # Monitor
# monitor.track_query(query, answer, results, latency, evaluation)

# # After multiple queries, check stats
# print(monitor.get_stats())
# # {'total_queries': 10, 'avg_latency': 1.234, 'avg_relevance': 0.82, 'avg_groundedness': 0.75}

# # Save logs
# monitor.save_logs()
