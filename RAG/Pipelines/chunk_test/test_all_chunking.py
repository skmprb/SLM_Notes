from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor
from TextChunkingSplitting import TextChunker

collector = DataCollector()
preprocessor = DataPreprocessor()
chunker = TextChunker()

# Collect and preprocess data
data = collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
cleaned_data = preprocessor.preprocess(data)

print("=" * 80)
print("COMPREHENSIVE CHUNKING METHODS COMPARISON")
print("=" * 80)

# 1. Character-based
chunks_char = chunker.chunk_with_metadata(cleaned_data, method='characters', chunk_size=500, overlap=100)
print(f"\n1. Character-based: {len(chunks_char)} chunks")
print(f"   - Simple fixed-size splitting with overlap")

# 2. Word-based
chunks_word = chunker.chunk_with_metadata(cleaned_data, method='words', chunk_size=100, overlap=20)
print(f"\n2. Word-based: {len(chunks_word)} chunks")
print(f"   - Splits by word count")

# 3. Sentence-based
chunks_sent = chunker.chunk_with_metadata(cleaned_data, method='sentences', sentences_per_chunk=5)
print(f"\n3. Sentence-based: {len(chunks_sent)} chunks")
print(f"   - Groups sentences together")

# 4. Paragraph-based
chunks_para = chunker.chunk_with_metadata(cleaned_data, method='paragraph', max_chunk_size=1000)
print(f"\n4. Paragraph-based: {len(chunks_para)} chunks")
print(f"   - Respects paragraph boundaries")

# 5. Similarity-based (TRUE semantic)
chunks_similarity = chunker.chunk_with_metadata(
    cleaned_data, 
    method='similarity',
    similarity_threshold=0.5,
    max_chunk_size=1500
)
print(f"\n5. Similarity-based: {len(chunks_similarity)} chunks")
print(f"   - Uses embeddings to detect topic changes")

# 6. Recursive (smart splitting)
chunks_recursive = chunker.chunk_with_metadata(
    cleaned_data,
    method='recursive',
    chunk_size=1000,
    overlap=100
)
print(f"\n6. Recursive: {len(chunks_recursive)} chunks")
print(f"   - Tries separators in priority order (paragraphs → sentences → words)")

# 7. Hierarchical
chunks_hierarchical = chunker.chunk_with_metadata(
    cleaned_data,
    method='hierarchical',
    parent_size=2000,
    child_size=500,
    overlap=50
)
print(f"\n7. Hierarchical: {len(chunks_hierarchical)} child chunks")
print(f"   - Creates parent-child relationships")
print(f"   - Parents: {len(set(c['parent_id'] for c in chunks_hierarchical))}")

# Show hierarchical structure
print("\n" + "=" * 80)
print("HIERARCHICAL STRUCTURE EXAMPLE")
print("=" * 80)

# Group by parent
from collections import defaultdict
hierarchy_map = defaultdict(list)
for chunk in chunks_hierarchical:
    hierarchy_map[chunk['parent_id']].append(chunk)

# Show first parent and its children
first_parent_id = 0
print(f"\nParent Chunk {first_parent_id}:")
print(f"Content: {chunks_hierarchical[0]['parent_content'][:200]}...")
print(f"\nChildren ({len(hierarchy_map[first_parent_id])} total):")
for i, child in enumerate(hierarchy_map[first_parent_id][:3]):  # Show first 3 children
    print(f"\n  Child {i}:")
    print(f"  {child['content'][:150]}...")

# Compare first chunks from different methods
print("\n" + "=" * 80)
print("FIRST CHUNK COMPARISON")
print("=" * 80)

print(f"\n1. Character chunk ({len(chunks_char[0]['content'])} chars):")
print(f"   {chunks_char[0]['content'][:150]}...")

print(f"\n2. Paragraph chunk ({len(chunks_para[0]['content'])} chars):")
print(f"   {chunks_para[0]['content'][:150]}...")

print(f"\n3. Recursive chunk ({len(chunks_recursive[0]['content'])} chars):")
print(f"   {chunks_recursive[0]['content'][:150]}...")

print(f"\n4. Similarity chunk ({len(chunks_similarity[0]['content'])} chars):")
print(f"   {chunks_similarity[0]['content'][:150]}...")

print(f"\n5. Hierarchical child chunk ({len(chunks_hierarchical[0]['content'])} chars):")
print(f"   {chunks_hierarchical[0]['content'][:150]}...")

# Compare retrieval strategies
print("\n" + "=" * 80)
print("RETRIEVAL STRATEGY COMPARISON")
print("=" * 80)

print("""
1. Character/Word/Sentence: 
   - Fast, simple
   - May split mid-concept
   - Use for: Quick prototyping

2. Paragraph:
   - Respects structure
   - Assumes paragraphs = topics
   - Use for: Well-formatted documents

3. Recursive:
   - Respects natural boundaries (paragraphs → sentences → words)
   - Better than character splitting
   - Use for: General-purpose RAG, mixed content

4. Similarity (Semantic):
   - Detects topic boundaries via embeddings
   - Slower (needs embeddings)
   - Use for: High-quality RAG, topic-based retrieval

5. Hierarchical:
   - Best of both worlds
   - Retrieve child (precise) + parent (context)
   - Use for: Complex queries, re-ranking
   
   Example workflow:
   - Embed & store child chunks
   - On retrieval: return child + parent context to LLM
   - Gives precise answer with broader context
""")

# Show metadata comparison
print("\n" + "=" * 80)
print("METADATA COMPARISON")
print("=" * 80)

print("\n1. Standard chunk metadata (Character):")
print(chunks_char[0])

print("\n2. Recursive chunk metadata:")
print(chunks_recursive[0])

print("\n3. Similarity chunk metadata:")
print(chunks_similarity[0])

print("\n4. Hierarchical chunk metadata:")
print({k: v if k != 'parent_content' else f"{v[:100]}..." for k, v in chunks_hierarchical[0].items()})

# Performance summary
print("\n" + "=" * 80)
print("CHUNK COUNT SUMMARY")
print("=" * 80)
print(f"""
Method              | Chunks | Avg Size
--------------------|--------|----------
Character-based     | {len(chunks_char):6} | {sum(len(c['content']) for c in chunks_char)//len(chunks_char):4} chars
Word-based          | {len(chunks_word):6} | {sum(len(c['content']) for c in chunks_word)//len(chunks_word):4} chars
Sentence-based      | {len(chunks_sent):6} | {sum(len(c['content']) for c in chunks_sent)//len(chunks_sent):4} chars
Paragraph-based     | {len(chunks_para):6} | {sum(len(c['content']) for c in chunks_para)//len(chunks_para):4} chars
Recursive           | {len(chunks_recursive):6} | {sum(len(c['content']) for c in chunks_recursive)//len(chunks_recursive):4} chars
Similarity-based    | {len(chunks_similarity):6} | {sum(len(c['content']) for c in chunks_similarity)//len(chunks_similarity):4} chars
Hierarchical (kids) | {len(chunks_hierarchical):6} | {sum(len(c['content']) for c in chunks_hierarchical)//len(chunks_hierarchical):4} chars
""")
