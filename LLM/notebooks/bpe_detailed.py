from typing import List, Dict, Tuple
from collections import defaultdict

class SimpleBPE:
    """
    Simplified Byte Pair Encoding implementation with detailed step-by-step logging.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        print(f"🔧 Initialized BPE with target vocabulary size: {vocab_size}")
    
    def get_pairs(self, word: List[str]) -> Dict[Tuple[str, str], int]:
        """Extract all adjacent character pairs from a word."""
        pairs = defaultdict(int)
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def train(self, texts: List[str]):
        """Train BPE on a corpus by learning merge rules."""
        print(f"\n🚀 Starting BPE Training with vocab_size={self.vocab_size}")
        print(f"📚 Training corpus: {texts}")
        
        # Initialize with character vocabulary
        vocab = set()
        word_freqs = defaultdict(int)
        
        print("\n📊 Step 1: Counting word frequencies and collecting characters...")
        # Count word frequencies and collect characters
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[word] += 1
                vocab.update(word)  # Add all characters to vocab
        
        print(f"   📝 Word frequencies: {dict(word_freqs)}")
        print(f"   🔤 Initial character vocabulary: {sorted(vocab)}")
        print(f"   📏 Initial vocabulary size: {len(vocab)}")
        
        # Convert words to character lists for processing
        word_splits = {word: list(word) for word in word_freqs}
        print(f"\n🔨 Step 2: Initial word splits: {word_splits}")
        
        merge_count = 0
        # Iteratively merge most frequent pairs
        for i in range(self.vocab_size - len(vocab)):
            print(f"\n🔄 Merge iteration {i+1}:")
            pairs = defaultdict(int)
            
            # Count all pairs across all words (weighted by frequency)
            print("   🔍 Counting character pairs across all words...")
            for word, freq in word_freqs.items():
                word_pairs = self.get_pairs(word_splits[word])
                print(f"      Word '{word}' (freq={freq}): {word_splits[word]} -> pairs: {dict(word_pairs)}")
                for pair, count in word_pairs.items():
                    pairs[pair] += count * freq
            
            if not pairs:
                print("   ⚠️ No more pairs to merge. Stopping training.")
                break
            
            print(f"   📈 All pair frequencies: {dict(sorted(pairs.items(), key=lambda x: x[1], reverse=True))}")
            
            # Find most frequent pair to merge
            best_pair = max(pairs, key=pairs.get)
            print(f"   🏆 Most frequent pair: {best_pair} (frequency: {pairs[best_pair]})")
            
            # Apply merge to all words
            print("   🔧 Applying merge to all words...")
            for word in word_splits:
                old_split = word_splits[word].copy()
                new_word = []
                j = 0
                while j < len(word_splits[word]):
                    # Check if current position matches the pair to merge
                    if (j < len(word_splits[word]) - 1 and 
                        word_splits[word][j] == best_pair[0] and 
                        word_splits[word][j + 1] == best_pair[1]):
                        # Merge the pair
                        new_word.append(best_pair[0] + best_pair[1])
                        j += 2
                    else:
                        # Keep character as is
                        new_word.append(word_splits[word][j])
                        j += 1
                word_splits[word] = new_word
                if old_split != new_word:
                    print(f"      '{word}': {old_split} -> {new_word}")
            
            # Add merged token to vocabulary and record merge rule
            merged_token = best_pair[0] + best_pair[1]
            vocab.add(merged_token)
            self.merges[best_pair] = merged_token
            merge_count += 1
            
            print(f"   ✅ Merge #{merge_count}: {best_pair} -> '{merged_token}'")
            print(f"   📊 Current vocabulary size: {len(vocab)}")
        
        # Create final vocabulary mapping
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        
        print(f"\n🎉 Training Complete!")
        print(f"   📚 Final vocabulary size: {len(self.vocab)}")
        print(f"   🔤 Final vocabulary: {list(self.vocab.keys())}")
        print(f"   🔗 Learned {len(self.merges)} merge rules: {list(self.merges.items())}")
    
    def encode(self, text: str) -> List[str]:
        """Encode text using learned BPE merges."""
        print(f"\n🔤 Encoding text: '{text}'")
        words = text.split()
        encoded = []
        
        for word_idx, word in enumerate(words):
            print(f"\n   📝 Processing word {word_idx+1}: '{word}'")
            # Start with character-level tokenization
            word_tokens = list(word)
            print(f"      Initial tokens: {word_tokens}")
            
            iteration = 0
            # Apply merges iteratively
            while len(word_tokens) > 1:
                iteration += 1
                print(f"      🔄 Merge iteration {iteration}:")
                
                pairs = self.get_pairs(word_tokens)
                if not pairs:
                    print(f"         No pairs found. Stopping.")
                    break
                
                print(f"         Available pairs: {dict(pairs)}")
                
                # Find the pair that appears in our learned merges
                valid_pairs = [pair for pair in pairs if pair in self.merges]
                if not valid_pairs:
                    print(f"         No valid merge pairs found. Stopping.")
                    break
                
                print(f"         Valid merge pairs: {valid_pairs}")
                
                # Use the merge that was learned earliest (lowest index)
                bigram = min(valid_pairs, key=lambda pair: list(self.merges.keys()).index(pair))
                print(f"         Selected pair to merge: {bigram} -> '{self.merges[bigram]}'")
                
                # Apply the merge
                old_tokens = word_tokens.copy()
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == bigram[0] and 
                        word_tokens[i + 1] == bigram[1]):
                        # Apply merge
                        new_word.append(self.merges[bigram])
                        i += 2
                    else:
                        # Keep token as is
                        new_word.append(word_tokens[i])
                        i += 1
                word_tokens = new_word
                print(f"         {old_tokens} -> {word_tokens}")
            
            print(f"      ✅ Final tokens for '{word}': {word_tokens}")
            encoded.extend(word_tokens)
        
        print(f"   🎯 Complete encoding result: {encoded}")
        return encoded

# Demonstrate BPE with detailed output
if __name__ == "__main__":
    print("="*60)
    print("🧪 BPE DEMONSTRATION WITH DETAILED LOGGING")
    print("="*60)
    
    corpus = [
        "hello world",
        "hello there", 
        "world peace",
        "hello hello world"
    ]
    
    print(f"📚 Training corpus: {corpus}")
    bpe = SimpleBPE(vocab_size=20)
    bpe.train(corpus)
    
    print("\n" + "="*60)
    print("🔤 ENCODING EXAMPLES")
    print("="*60)
    
    test_texts = ["hello world", "hello", "world", "peace"]
    for text in test_texts:
        result = bpe.encode(text)
        print(f"\n📊 SUMMARY: '{text}' -> {result}")