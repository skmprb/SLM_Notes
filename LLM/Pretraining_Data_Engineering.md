# Pretraining Data Engineering

## 🎯 Overview

Data engineering for LLM pretraining involves processing massive text corpora to create high-quality training datasets. This includes deduplication, filtering, contamination detection, and balancing to ensure optimal model performance.

## 🔄 Data Deduplication

### Why Deduplication Matters

**Problems with Duplicates:**
- Models memorize repeated content
- Reduced effective dataset diversity
- Overfitting to common patterns
- Biased evaluation metrics

### Exact Deduplication

```python
import hashlib
from collections import defaultdict

class ExactDeduplicator:
    def __init__(self):
        self.seen_hashes = set()
        self.duplicate_count = 0
        
    def get_hash(self, text: str) -> str:
        """Generate hash for exact matching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove exact duplicates."""
        unique_texts = []
        
        for text in texts:
            text_hash = self.get_hash(text)
            
            if text_hash not in self.seen_hashes:
                self.seen_hashes.add(text_hash)
                unique_texts.append(text)
            else:
                self.duplicate_count += 1
        
        return unique_texts
```

### Near-Duplicate Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NearDuplicateDetector:
    def __init__(self, threshold=0.85, ngram_range=(1, 3)):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=10000,
            stop_words='english'
        )
        
    def find_near_duplicates(self, texts: List[str]) -> List[int]:
        """Find indices of near-duplicate texts."""
        # Vectorize texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find duplicates
        duplicates = set()
        n = len(texts)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > self.threshold:
                    duplicates.add(j)  # Keep first occurrence
        
        return list(duplicates)
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove near-duplicates."""
        duplicate_indices = self.find_near_duplicates(texts)
        return [text for i, text in enumerate(texts) if i not in duplicate_indices]
```

### MinHash-Based Deduplication

```python
import random
from typing import Set

class MinHashDeduplicator:
    def __init__(self, num_hashes=128, shingle_size=5):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.hash_functions = self._generate_hash_functions()
        
    def _generate_hash_functions(self):
        """Generate random hash functions."""
        random.seed(42)
        return [(random.randint(1, 2**32), random.randint(0, 2**32)) 
                for _ in range(self.num_hashes)]
    
    def get_shingles(self, text: str) -> Set[str]:
        """Generate character shingles."""
        text = text.lower().replace(' ', '')
        return {text[i:i+self.shingle_size] 
                for i in range(len(text) - self.shingle_size + 1)}
    
    def compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        signature = []
        
        for a, b in self.hash_functions:
            min_hash = float('inf')
            
            for shingle in shingles:
                hash_val = (a * hash(shingle) + b) % (2**32)
                min_hash = min(min_hash, hash_val)
            
            signature.append(min_hash)
        
        return signature
    
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def deduplicate(self, texts: List[str], threshold=0.8) -> List[str]:
        """Deduplicate using MinHash."""
        signatures = []
        
        # Compute signatures
        for text in texts:
            shingles = self.get_shingles(text)
            signature = self.compute_minhash(shingles)
            signatures.append(signature)
        
        # Find duplicates
        keep_indices = set(range(len(texts)))
        
        for i in range(len(texts)):
            if i not in keep_indices:
                continue
                
            for j in range(i + 1, len(texts)):
                if j not in keep_indices:
                    continue
                
                similarity = self.jaccard_similarity(signatures[i], signatures[j])
                if similarity > threshold:
                    keep_indices.discard(j)
        
        return [texts[i] for i in sorted(keep_indices)]
```

## 🔍 Data Filtering

### Quality-Based Filtering

```python
import re
from typing import Dict

class QualityFilter:
    def __init__(self):
        self.filters = {
            'length': self.filter_by_length,
            'language': self.filter_by_language,
            'repetition': self.filter_repetitive_content,
            'special_chars': self.filter_special_characters,
            'readability': self.filter_by_readability
        }
    
    def filter_by_length(self, text: str, min_length=50, max_length=100000) -> bool:
        """Filter by text length."""
        return min_length <= len(text) <= max_length
    
    def filter_by_language(self, text: str, target_lang='en') -> bool:
        """Filter by language (simplified)."""
        # In practice, use proper language detection library
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a']
        words = text.lower().split()
        
        if len(words) < 10:
            return True  # Too short to determine
        
        english_count = sum(1 for word in words[:50] if word in english_indicators)
        return english_count / min(len(words), 50) > 0.1
    
    def filter_repetitive_content(self, text: str, max_repetition=0.3) -> bool:
        """Filter highly repetitive content."""
        lines = text.split('\n')
        if len(lines) < 3:
            return True
        
        # Check for repeated lines
        line_counts = {}
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Only count substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1
        
        if not line_counts:
            return True
        
        max_count = max(line_counts.values())
        repetition_ratio = max_count / len(lines)
        
        return repetition_ratio <= max_repetition
    
    def filter_special_characters(self, text: str, max_special_ratio=0.3) -> bool:
        """Filter text with too many special characters."""
        if not text:
            return False
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = special_chars / len(text)
        
        return ratio <= max_special_ratio
    
    def filter_by_readability(self, text: str, min_score=30) -> bool:
        """Filter by readability score (simplified Flesch)."""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self.count_syllables(text)
        
        if sentences == 0 or words == 0:
            return False
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * words/sentences) - (84.6 * syllables/words)
        return score >= min_score
    
    def count_syllables(self, text: str) -> int:
        """Estimate syllable count."""
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in text.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        return max(1, syllables)
    
    def apply_filters(self, text: str, enabled_filters=None) -> Dict[str, bool]:
        """Apply multiple filters and return results."""
        if enabled_filters is None:
            enabled_filters = list(self.filters.keys())
        
        results = {}
        for filter_name in enabled_filters:
            if filter_name in self.filters:
                results[filter_name] = self.filters[filter_name](text)
        
        results['overall_pass'] = all(results.values())
        return results
```

### Content-Based Filtering

```python
class ContentFilter:
    def __init__(self):
        self.toxic_keywords = self.load_toxic_keywords()
        self.spam_patterns = self.load_spam_patterns()
        
    def load_toxic_keywords(self) -> Set[str]:
        """Load toxic keywords list."""
        # In practice, load from comprehensive dataset
        return {'hate', 'violence', 'explicit', 'harmful'}
    
    def load_spam_patterns(self) -> List[str]:
        """Load spam detection patterns."""
        return [
            r'click here',
            r'buy now',
            r'limited time',
            r'act fast',
            r'\$\d+',  # Price patterns
        ]
    
    def detect_toxic_content(self, text: str) -> bool:
        """Detect potentially toxic content."""
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in self.toxic_keywords 
                         if keyword in text_lower)
        
        # Simple threshold-based detection
        return toxic_count > 2
    
    def detect_spam(self, text: str) -> bool:
        """Detect spam content."""
        text_lower = text.lower()
        
        spam_indicators = 0
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                spam_indicators += 1
        
        return spam_indicators >= 2
    
    def filter_content(self, text: str) -> bool:
        """Apply content filtering."""
        return not (self.detect_toxic_content(text) or self.detect_spam(text))
```

## 🔬 Data Contamination

### Test Set Contamination Detection

```python
class ContaminationDetector:
    def __init__(self, test_datasets: Dict[str, List[str]]):
        self.test_datasets = test_datasets
        self.contamination_threshold = 0.8
        
    def exact_match_detection(self, training_text: str) -> Dict[str, bool]:
        """Detect exact matches with test sets."""
        contamination = {}
        
        for dataset_name, test_texts in self.test_datasets.items():
            is_contaminated = any(training_text.strip() == test_text.strip() 
                                for test_text in test_texts)
            contamination[dataset_name] = is_contaminated
        
        return contamination
    
    def substring_contamination(self, training_text: str, min_length=50) -> Dict[str, bool]:
        """Detect substring contamination."""
        contamination = {}
        
        for dataset_name, test_texts in self.test_datasets.items():
            is_contaminated = False
            
            for test_text in test_texts:
                # Check if substantial substring exists
                for i in range(len(test_text) - min_length + 1):
                    substring = test_text[i:i + min_length]
                    if substring in training_text:
                        is_contaminated = True
                        break
                
                if is_contaminated:
                    break
            
            contamination[dataset_name] = is_contaminated
        
        return contamination
    
    def n_gram_contamination(self, training_text: str, n=8) -> Dict[str, float]:
        """Detect n-gram level contamination."""
        def get_ngrams(text: str, n: int) -> Set[str]:
            words = text.split()
            return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}
        
        contamination_scores = {}
        training_ngrams = get_ngrams(training_text, n)
        
        for dataset_name, test_texts in self.test_datasets.items():
            max_overlap = 0
            
            for test_text in test_texts:
                test_ngrams = get_ngrams(test_text, n)
                
                if test_ngrams:
                    overlap = len(training_ngrams & test_ngrams) / len(test_ngrams)
                    max_overlap = max(max_overlap, overlap)
            
            contamination_scores[dataset_name] = max_overlap
        
        return contamination_scores
    
    def detect_contamination(self, training_text: str) -> Dict[str, any]:
        """Comprehensive contamination detection."""
        results = {
            'exact_matches': self.exact_match_detection(training_text),
            'substring_contamination': self.substring_contamination(training_text),
            'ngram_contamination': self.n_gram_contamination(training_text)
        }
        
        # Overall contamination decision
        results['is_contaminated'] = any([
            any(results['exact_matches'].values()),
            any(results['substring_contamination'].values()),
            any(score > self.contamination_threshold 
                for score in results['ngram_contamination'].values())
        ])
        
        return results
```

### Benchmark Contamination Analysis

```python
class BenchmarkContaminationAnalyzer:
    def __init__(self):
        self.benchmark_datasets = {
            'hellaswag': [],
            'arc': [],
            'mmlu': [],
            'gsm8k': [],
            'humaneval': []
        }
        
    def load_benchmark_data(self, dataset_name: str, data_path: str):
        """Load benchmark dataset for contamination checking."""
        # In practice, load actual benchmark data
        with open(data_path, 'r') as f:
            self.benchmark_datasets[dataset_name] = f.readlines()
    
    def analyze_training_corpus(self, corpus_path: str) -> Dict[str, any]:
        """Analyze entire training corpus for contamination."""
        contamination_stats = {
            'total_documents': 0,
            'contaminated_documents': 0,
            'contamination_by_benchmark': {},
            'contaminated_examples': []
        }
        
        detector = ContaminationDetector(self.benchmark_datasets)
        
        with open(corpus_path, 'r') as f:
            for line_num, line in enumerate(f):
                contamination_stats['total_documents'] += 1
                
                results = detector.detect_contamination(line.strip())
                
                if results['is_contaminated']:
                    contamination_stats['contaminated_documents'] += 1
                    contamination_stats['contaminated_examples'].append({
                        'line_number': line_num,
                        'text_preview': line[:100] + '...',
                        'contamination_details': results
                    })
                
                # Track per-benchmark contamination
                for benchmark in self.benchmark_datasets:
                    if benchmark not in contamination_stats['contamination_by_benchmark']:
                        contamination_stats['contamination_by_benchmark'][benchmark] = 0
                    
                    if results['exact_matches'].get(benchmark, False):
                        contamination_stats['contamination_by_benchmark'][benchmark] += 1
        
        # Calculate contamination rate
        contamination_stats['contamination_rate'] = (
            contamination_stats['contaminated_documents'] / 
            contamination_stats['total_documents']
        )
        
        return contamination_stats
```

## ⚖️ Dataset Balancing

### Domain Balancing

```python
class DatasetBalancer:
    def __init__(self):
        self.domain_classifiers = {}
        self.target_distribution = {}
        
    def classify_domain(self, text: str) -> str:
        """Classify text domain (simplified)."""
        # In practice, use trained domain classifier
        domain_keywords = {
            'news': ['reported', 'according', 'sources', 'breaking'],
            'academic': ['research', 'study', 'analysis', 'findings'],
            'web': ['website', 'click', 'link', 'homepage'],
            'books': ['chapter', 'novel', 'story', 'character'],
            'code': ['function', 'class', 'import', 'return']
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'other'
    
    def analyze_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Analyze current domain distribution."""
        domain_counts = {}
        
        for text in texts:
            domain = self.classify_domain(text)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        total = len(texts)
        return {domain: count/total for domain, count in domain_counts.items()}
    
    def balance_dataset(self, texts: List[str], target_dist: Dict[str, float]) -> List[str]:
        """Balance dataset according to target distribution."""
        # Group texts by domain
        domain_texts = {}
        for text in texts:
            domain = self.classify_domain(text)
            if domain not in domain_texts:
                domain_texts[domain] = []
            domain_texts[domain].append(text)
        
        # Calculate target counts
        total_target = len(texts)
        balanced_texts = []
        
        for domain, target_ratio in target_dist.items():
            target_count = int(total_target * target_ratio)
            
            if domain in domain_texts:
                available_texts = domain_texts[domain]
                
                if len(available_texts) >= target_count:
                    # Randomly sample if we have enough
                    import random
                    selected = random.sample(available_texts, target_count)
                else:
                    # Use all available and oversample if needed
                    selected = available_texts * (target_count // len(available_texts))
                    remaining = target_count % len(available_texts)
                    selected.extend(random.sample(available_texts, remaining))
                
                balanced_texts.extend(selected)
        
        return balanced_texts
```

### Quality-Based Sampling

```python
class QualityBasedSampler:
    def __init__(self):
        self.quality_scorer = QualityFilter()
        
    def compute_quality_score(self, text: str) -> float:
        """Compute overall quality score for text."""
        filter_results = self.quality_scorer.apply_filters(text)
        
        # Weight different quality aspects
        weights = {
            'length': 0.1,
            'language': 0.2,
            'repetition': 0.3,
            'special_chars': 0.2,
            'readability': 0.2
        }
        
        score = 0
        for aspect, passed in filter_results.items():
            if aspect in weights and passed:
                score += weights[aspect]
        
        return score
    
    def stratified_sampling(self, texts: List[str], sample_size: int, 
                          quality_bins=5) -> List[str]:
        """Sample texts maintaining quality distribution."""
        # Compute quality scores
        scored_texts = [(text, self.compute_quality_score(text)) for text in texts]
        
        # Sort by quality score
        scored_texts.sort(key=lambda x: x[1])
        
        # Create quality bins
        bin_size = len(scored_texts) // quality_bins
        bins = []
        
        for i in range(quality_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < quality_bins - 1 else len(scored_texts)
            bins.append(scored_texts[start_idx:end_idx])
        
        # Sample from each bin proportionally
        sampled_texts = []
        samples_per_bin = sample_size // quality_bins
        
        for bin_texts in bins:
            if len(bin_texts) >= samples_per_bin:
                import random
                sampled = random.sample(bin_texts, samples_per_bin)
            else:
                sampled = bin_texts
            
            sampled_texts.extend([text for text, score in sampled])
        
        return sampled_texts
```

### Temporal Balancing

```python
import datetime
from collections import defaultdict

class TemporalBalancer:
    def __init__(self):
        self.date_extractors = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'
        ]
    
    def extract_date(self, text: str) -> Optional[datetime.date]:
        """Extract date from text (simplified)."""
        import re
        
        for pattern in self.date_extractors:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group()
                    # Parse different formats
                    if '-' in date_str:
                        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    elif '/' in date_str:
                        return datetime.datetime.strptime(date_str, '%m/%d/%Y').date()
                    # Add more parsing logic as needed
                except:
                    continue
        
        return None
    
    def balance_temporal_distribution(self, texts: List[str], 
                                   target_years: List[int]) -> List[str]:
        """Balance dataset across target years."""
        # Group texts by year
        year_texts = defaultdict(list)
        undated_texts = []
        
        for text in texts:
            date = self.extract_date(text)
            if date:
                year_texts[date.year].append(text)
            else:
                undated_texts.append(text)
        
        # Calculate samples per year
        total_dated = sum(len(texts) for texts in year_texts.values())
        samples_per_year = total_dated // len(target_years)
        
        balanced_texts = []
        
        for year in target_years:
            if year in year_texts:
                available = year_texts[year]
                if len(available) >= samples_per_year:
                    import random
                    selected = random.sample(available, samples_per_year)
                else:
                    selected = available
                
                balanced_texts.extend(selected)
        
        # Add some undated texts
        if undated_texts:
            import random
            undated_sample_size = min(len(undated_texts), len(balanced_texts) // 10)
            balanced_texts.extend(random.sample(undated_texts, undated_sample_size))
        
        return balanced_texts
```

## 🔧 Complete Data Pipeline

```python
class PretrainingDataPipeline:
    def __init__(self, config: Dict[str, any]):
        self.config = config
        
        # Initialize components
        self.deduplicator = MinHashDeduplicator()
        self.quality_filter = QualityFilter()
        self.content_filter = ContentFilter()
        self.contamination_detector = ContaminationDetector(config.get('test_datasets', {}))
        self.balancer = DatasetBalancer()
        
        # Statistics
        self.stats = {
            'total_input': 0,
            'after_dedup': 0,
            'after_quality_filter': 0,
            'after_content_filter': 0,
            'after_contamination_filter': 0,
            'final_output': 0
        }
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts through the pipeline."""
        self.stats['total_input'] += len(texts)
        
        # Step 1: Deduplication
        texts = self.deduplicator.deduplicate(texts)
        self.stats['after_dedup'] += len(texts)
        
        # Step 2: Quality filtering
        texts = [text for text in texts 
                if self.quality_filter.apply_filters(text)['overall_pass']]
        self.stats['after_quality_filter'] += len(texts)
        
        # Step 3: Content filtering
        texts = [text for text in texts if self.content_filter.filter_content(text)]
        self.stats['after_content_filter'] += len(texts)
        
        # Step 4: Contamination filtering
        texts = [text for text in texts 
                if not self.contamination_detector.detect_contamination(text)['is_contaminated']]
        self.stats['after_contamination_filter'] += len(texts)
        
        # Step 5: Balancing (if configured)
        if self.config.get('balance_domains', False):
            target_dist = self.config.get('target_distribution', {})
            texts = self.balancer.balance_dataset(texts, target_dist)
        
        self.stats['final_output'] += len(texts)
        
        return texts
    
    def get_pipeline_stats(self) -> Dict[str, any]:
        """Get pipeline processing statistics."""
        stats = self.stats.copy()
        
        if stats['total_input'] > 0:
            stats['dedup_rate'] = 1 - (stats['after_dedup'] / stats['total_input'])
            stats['quality_filter_rate'] = 1 - (stats['after_quality_filter'] / stats['after_dedup'])
            stats['content_filter_rate'] = 1 - (stats['after_content_filter'] / stats['after_quality_filter'])
            stats['contamination_rate'] = 1 - (stats['after_contamination_filter'] / stats['after_content_filter'])
            stats['overall_retention_rate'] = stats['final_output'] / stats['total_input']
        
        return stats
```

## 📊 Quality Metrics and Monitoring

```python
class DataQualityMonitor:
    def __init__(self):
        self.metrics = {}
        
    def compute_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Compute dataset diversity metrics."""
        # Vocabulary diversity
        all_words = set()
        total_words = 0
        
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
            total_words += len(words)
        
        vocab_diversity = len(all_words) / total_words if total_words > 0 else 0
        
        # Length diversity
        lengths = [len(text.split()) for text in texts]
        length_std = np.std(lengths) if lengths else 0
        
        return {
            'vocabulary_diversity': vocab_diversity,
            'length_diversity': length_std,
            'unique_documents': len(set(texts)),
            'total_documents': len(texts)
        }
    
    def compute_quality_distribution(self, texts: List[str]) -> Dict[str, any]:
        """Compute quality score distribution."""
        quality_filter = QualityFilter()
        scores = []
        
        for text in texts:
            filter_results = quality_filter.apply_filters(text)
            score = sum(filter_results.values()) / len(filter_results)
            scores.append(score)
        
        return {
            'mean_quality': np.mean(scores),
            'std_quality': np.std(scores),
            'min_quality': np.min(scores),
            'max_quality': np.max(scores),
            'quality_distribution': np.histogram(scores, bins=10)[0].tolist()
        }
```

## 📚 Summary

### Key Components

**Data Deduplication**
- Exact matching with hashing
- Near-duplicate detection with TF-IDF
- MinHash for large-scale deduplication

**Data Filtering**
- Quality-based filtering (length, language, readability)
- Content-based filtering (toxicity, spam detection)
- Multi-criteria filtering pipelines

**Contamination Detection**
- Exact match detection with test sets
- Substring and n-gram contamination
- Benchmark-specific contamination analysis

**Dataset Balancing**
- Domain distribution balancing
- Quality-based stratified sampling
- Temporal distribution balancing

### Best Practices
- **Pipeline approach**: Sequential filtering with statistics tracking
- **Quality monitoring**: Continuous assessment of data quality metrics
- **Contamination prevention**: Rigorous testing against evaluation benchmarks
- **Balanced representation**: Ensuring diverse and representative datasets

### Impact on Model Performance
- **Deduplication**: Reduces memorization, improves generalization
- **Quality filtering**: Enhances model capabilities and reduces noise
- **Contamination removal**: Ensures valid evaluation and benchmarking
- **Balancing**: Prevents domain bias and improves robustness

Proper data engineering is crucial for training high-quality LLMs that generalize well and perform reliably across diverse tasks and domains.