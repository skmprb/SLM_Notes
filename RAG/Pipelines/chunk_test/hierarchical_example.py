"""
HIERARCHICAL CHUNKING EXPLAINED
================================

Concept: Create a tree structure with parent (large) and child (small) chunks

Example Document:
-----------------
"Machine learning is a subset of AI. It uses algorithms to learn patterns. 
Deep learning is a type of ML. It uses neural networks with many layers.
Natural language processing helps computers understand text. It's used in chatbots."

Traditional Chunking:
--------------------
Chunk 1: "Machine learning is a subset of AI. It uses algorithms..."
Chunk 2: "Deep learning is a type of ML. It uses neural networks..."
Chunk 3: "Natural language processing helps computers..."

Problem: Each chunk is independent, no broader context

Hierarchical Chunking:
---------------------
Parent 1 (2000 chars):
├── Child 1.1 (500 chars): "Machine learning is a subset of AI..."
├── Child 1.2 (500 chars): "It uses algorithms to learn patterns..."
└── Child 1.3 (500 chars): "Deep learning is a type of ML..."

Parent 2 (2000 chars):
├── Child 2.1 (500 chars): "It uses neural networks with many layers..."
└── Child 2.2 (500 chars): "Natural language processing helps..."

Benefits:
---------
1. Precise Retrieval: Search child chunks (small, focused)
2. Rich Context: Return parent chunk to LLM (broader context)
3. Better Answers: LLM sees both specific info + surrounding context

RAG Workflow:
------------
1. Embed & Index: Store child chunks in vector DB
2. User Query: "What is deep learning?"
3. Retrieve: Find matching child chunk (1.3)
4. Augment: Send to LLM with parent context (Parent 1)
5. Generate: LLM has precise answer + related ML context

Use Cases:
----------
- Long documents (research papers, books)
- Complex topics needing context
- Multi-hop reasoning
- Re-ranking strategies
"""

from TextChunkingSplitting import TextChunker

# Simple example
text = """Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data and make predictions. Machine learning models improve over time with more data.

Deep learning is a specialized type of machine learning. It uses neural networks with multiple layers to process complex patterns. Deep learning excels at image and speech recognition.

Natural language processing is another AI field. It helps computers understand and generate human language. NLP powers chatbots, translation, and sentiment analysis."""

chunker = TextChunker()

# Create hierarchical chunks
result = chunker.hierarchical_chunking(
    text,
    parent_size=300,  # Small for demo
    child_size=100,
    overlap=20
)

print("=" * 80)
print("HIERARCHICAL CHUNKING VISUALIZATION")
print("=" * 80)

for parent in result['hierarchy']:
    print(f"\n📦 PARENT {parent['parent_id']} ({len(parent['parent_content'])} chars)")
    print(f"   Content: {parent['parent_content'][:150]}...")
    print(f"   Children: {parent['num_children']}")
    
    for child in parent['children']:
        print(f"\n   └── 📄 CHILD {child['child_id']}")
        print(f"       {child['content'][:100]}...")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total Parents: {result['total_parents']}")
print(f"Total Children: {result['total_children']}")
print(f"\nIn Vector DB: Store {result['total_children']} child embeddings")
print(f"On Retrieval: Return child + parent for context")
