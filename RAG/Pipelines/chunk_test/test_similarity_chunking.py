from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor
from TextChunkingSplitting import TextChunker

collector = DataCollector()
preprocessor = DataPreprocessor()
chunker = TextChunker()

# Collect and preprocess data
data = collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
cleaned_data = preprocessor.preprocess(data)

# Test different chunking methods
print("=" * 60)
print("TESTING DIFFERENT CHUNKING METHODS")
print("=" * 60)

# 1. Character-based (simple splitting)
chunks_char = chunker.chunk_with_metadata(cleaned_data, method='characters', chunk_size=500, overlap=100)
print(f"\n1. Character-based chunks: {len(chunks_char)}")

# 2. Paragraph-based (renamed from semantic)
chunks_para = chunker.chunk_with_metadata(cleaned_data, method='paragraph', max_chunk_size=1000)
print(f"2. Paragraph-based chunks: {len(chunks_para)}")

# 3. NEW: Similarity-based (TRUE semantic chunking)
chunks_similarity = chunker.chunk_with_metadata(
    cleaned_data, 
    method='similarity',
    similarity_threshold=0.5,  # Lower = more splits (stricter topic boundaries)
    max_chunk_size=1500
)
print(f"3. Similarity-based chunks: {len(chunks_similarity)}")

# Compare first chunks
print("\n" + "=" * 60)
print("FIRST CHUNK COMPARISON")
print("=" * 60)

print(f"\nCharacter chunk:\n{chunks_char[0]['content'][:200]}...")
print(f"\nParagraph chunk:\n{chunks_para[0]['content'][:200]}...")
print(f"\nSimilarity chunk:\n{chunks_similarity[0]['content'][:1500]}...")

# Show metadata
print("\n" + "=" * 60)
print("METADATA EXAMPLE")
print("=" * 60)
print(f"\nSimilarity chunk metadata:\n{chunks_similarity[0]}")
