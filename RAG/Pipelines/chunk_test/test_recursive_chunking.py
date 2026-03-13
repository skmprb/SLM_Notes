"""
RECURSIVE CHUNKING EXPLAINED
=============================

Concept: Try splitting by natural boundaries in order of priority
If a chunk is still too large, recursively try the next separator

Separator Hierarchy (default):
1. \\n\\n  (paragraphs) - Most semantic
2. \\n    (lines)
3. .     (sentences)
4. " "   (words)
5. ""    (characters) - Last resort

Example:
--------
Text: "Paragraph 1 is very long...\n\nParagraph 2...\n\nParagraph 3..."
Chunk size: 100

Step 1: Try splitting by \\n\\n (paragraphs)
  - If paragraph fits → keep it
  - If paragraph too large → recursively split by \\n (lines)
  
Step 2: If line still too large → split by . (sentences)
Step 3: If sentence still too large → split by " " (words)
Step 4: If word still too large → split by "" (characters)

Benefits:
---------
✓ Respects natural text structure
✓ Keeps semantic units together when possible
✓ Gracefully handles edge cases (very long sentences)
✓ Better than fixed-size splitting
✓ Similar to LangChain's RecursiveCharacterTextSplitter
"""

from TextChunkingSplitting import TextChunker

# Example 1: Well-structured text
text1 = """Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data.

Deep learning uses neural networks. These networks have multiple layers. They excel at pattern recognition.

Natural language processing helps computers understand text. It powers chatbots and translation systems."""

# Example 2: Text with long paragraphs
text2 = """This is a very long paragraph that exceeds our chunk size limit. It contains multiple sentences. Each sentence adds more information. The recursive chunker will split this intelligently. It will try paragraphs first, then sentences, then words if needed. This ensures we maintain semantic coherence as much as possible while respecting size constraints.

This is a shorter paragraph that fits within limits."""

chunker = TextChunker()

print("=" * 80)
print("RECURSIVE CHUNKING DEMONSTRATION")
print("=" * 80)

# Test 1: Default separators
print("\n1. DEFAULT SEPARATORS (paragraphs → lines → sentences → words → chars)")
print("-" * 80)
chunks1 = chunker.recursive_chunking(text1, chunk_size=150, overlap=20)
print(f"Chunks created: {len(chunks1)}")
for i, chunk in enumerate(chunks1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"  {chunk[:100]}...")

# Test 2: Long paragraph handling
print("\n\n2. HANDLING LONG PARAGRAPHS")
print("-" * 80)
chunks2 = chunker.recursive_chunking(text2, chunk_size=200, overlap=30)
print(f"Chunks created: {len(chunks2)}")
for i, chunk in enumerate(chunks2):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"  {chunk[:100]}...")

# Test 3: Custom separators (code-specific)
print("\n\n3. CUSTOM SEPARATORS (for code)")
print("-" * 80)
code_text = """def function1():
    print("Hello")
    return True

def function2():
    x = 10
    y = 20
    return x + y

class MyClass:
    def __init__(self):
        self.value = 0"""

code_separators = ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
chunks3 = chunker.recursive_chunking(code_text, separators=code_separators, chunk_size=100, overlap=10)
print(f"Chunks created: {len(chunks3)}")
for i, chunk in enumerate(chunks3):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"  {chunk[:80]}...")

# Test 4: Compare with character-based
print("\n\n4. COMPARISON: Recursive vs Character-based")
print("-" * 80)

test_text = """AI is transforming industries. Machine learning enables predictions.

Deep learning powers image recognition. Neural networks learn patterns."""

recursive_chunks = chunker.recursive_chunking(test_text, chunk_size=80, overlap=10)
char_chunks = chunker.chunk_by_characters(test_text, chunk_size=80, overlap=10)

print(f"\nRecursive chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"  {i}: {chunk}")

print(f"\nCharacter chunks: {len(char_chunks)}")
for i, chunk in enumerate(char_chunks):
    print(f"  {i}: {chunk}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)
print("""
✓ Recursive chunking respects sentence boundaries
✓ Character chunking may split mid-word or mid-sentence
✓ Recursive is more semantic while maintaining size constraints
✓ Custom separators allow domain-specific chunking (code, markdown, etc.)
""")
