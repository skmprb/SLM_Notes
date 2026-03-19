# Applications of Large Language Models

## 🎯 Overview

Large Language Models have revolutionized numerous applications across industries. Their ability to understand and generate human-like text has enabled breakthrough capabilities in text generation, question answering, summarization, translation, code generation, and conversational AI.

## ✍️ Text Generation

### Core Capabilities

**Creative Writing**
- Story generation and continuation
- Poetry and creative content
- Screenplay and dialogue writing
- Marketing copy and content creation

**Technical Writing**
- Documentation generation
- Report writing
- Email composition
- Proposal and presentation content

### Implementation Approaches

**1. Conditional Text Generation**
```python
class ConditionalTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_with_prompt(self, prompt: str, max_length: int = 100, 
                           temperature: float = 0.7, top_p: float = 0.9):
        """Generate text conditioned on a prompt."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Remove prompt from output
    
    def generate_with_style_control(self, content: str, style: str):
        """Generate text with specific style control."""
        style_prompts = {
            'formal': f"Rewrite the following in a formal, professional tone:\n{content}\n\nFormal version:",
            'casual': f"Rewrite the following in a casual, friendly tone:\n{content}\n\nCasual version:",
            'technical': f"Rewrite the following in technical, precise language:\n{content}\n\nTechnical version:"
        }
        
        prompt = style_prompts.get(style, f"Rewrite:\n{content}\n\nRewritten:")
        return self.generate_with_prompt(prompt)
```

**2. Structured Text Generation**
```python
class StructuredTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_article(self, topic: str, sections: List[str]):
        """Generate structured article with multiple sections."""
        article_parts = []
        
        # Generate title
        title_prompt = f"Write a compelling title for an article about {topic}:"
        title = self.generate_with_prompt(title_prompt, max_length=20)
        article_parts.append(f"# {title.strip()}")
        
        # Generate each section
        for section in sections:
            section_prompt = f"Write a detailed section about {section} in the context of {topic}:"
            section_content = self.generate_with_prompt(section_prompt, max_length=200)
            article_parts.append(f"\n## {section}\n{section_content}")
        
        return "\n".join(article_parts)
    
    def generate_product_description(self, product_info: Dict[str, str]):
        """Generate product description from structured information."""
        prompt = f"""
        Product: {product_info['name']}
        Category: {product_info['category']}
        Key Features: {', '.join(product_info['features'])}
        Target Audience: {product_info['target_audience']}
        
        Write an engaging product description:
        """
        
        return self.generate_with_prompt(prompt, max_length=150)
```

### Quality Control

**1. Content Filtering**
```python
class ContentQualityFilter:
    def __init__(self):
        self.quality_metrics = {
            'coherence': self.check_coherence,
            'relevance': self.check_relevance,
            'factuality': self.check_factuality,
            'toxicity': self.check_toxicity
        }
        
    def evaluate_generated_text(self, text: str, context: str = "") -> Dict[str, float]:
        """Evaluate quality of generated text."""
        scores = {}
        
        for metric, check_function in self.quality_metrics.items():
            scores[metric] = check_function(text, context)
        
        scores['overall'] = sum(scores.values()) / len(scores)
        return scores
    
    def check_coherence(self, text: str, context: str = "") -> float:
        """Check text coherence (simplified)."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence check based on sentence length variation
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        # Penalize extreme length variations
        length_std = np.std(lengths)
        coherence_score = max(0, 1 - (length_std / np.mean(lengths)))
        return coherence_score
    
    def should_regenerate(self, text: str, context: str = "", threshold: float = 0.7) -> bool:
        """Determine if text should be regenerated."""
        scores = self.evaluate_generated_text(text, context)
        return scores['overall'] < threshold
```

## ❓ Question Answering

### QA System Types

**1. Extractive QA**
```python
class ExtractiveQA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def answer_question(self, question: str, context: str) -> Dict[str, any]:
        """Extract answer from given context."""
        inputs = self.tokenizer.encode_plus(
            question, context,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        # Find best answer span
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence
        start_prob = torch.softmax(start_logits, dim=1)[0][start_idx].item()
        end_prob = torch.softmax(end_logits, dim=1)[0][end_idx].item()
        confidence = (start_prob + end_prob) / 2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'start_position': start_idx.item(),
            'end_position': end_idx.item()
        }
```

**2. Generative QA**
```python
class GenerativeQA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def answer_question(self, question: str, context: str = "") -> str:
        """Generate answer using language model."""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                temperature=0.3,  # Lower temperature for factual answers
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        
        return answer
    
    def multi_hop_qa(self, question: str, documents: List[str]) -> Dict[str, any]:
        """Answer questions requiring multiple reasoning steps."""
        reasoning_steps = []
        current_question = question
        
        for step in range(3):  # Maximum 3 reasoning steps
            # Find most relevant document
            best_doc = self.find_most_relevant_document(current_question, documents)
            
            # Generate intermediate answer
            intermediate_answer = self.answer_question(current_question, best_doc)
            reasoning_steps.append({
                'step': step + 1,
                'question': current_question,
                'document': best_doc[:100] + "...",
                'answer': intermediate_answer
            })
            
            # Check if we have final answer
            if self.is_final_answer(intermediate_answer, question):
                break
            
            # Generate follow-up question
            current_question = self.generate_followup_question(
                question, intermediate_answer
            )
        
        return {
            'final_answer': reasoning_steps[-1]['answer'],
            'reasoning_steps': reasoning_steps
        }
```

### Knowledge-Augmented QA

```python
class KnowledgeAugmentedQA:
    def __init__(self, qa_model, knowledge_base, retriever):
        self.qa_model = qa_model
        self.knowledge_base = knowledge_base
        self.retriever = retriever
        
    def answer_with_retrieval(self, question: str, top_k: int = 5) -> Dict[str, any]:
        """Answer question using retrieved knowledge."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Combine retrieved context
        combined_context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Generate answer
        answer = self.qa_model.answer_question(question, combined_context)
        
        return {
            'answer': answer,
            'sources': [doc['source'] for doc in retrieved_docs],
            'retrieved_documents': retrieved_docs
        }
    
    def verify_answer(self, question: str, answer: str, sources: List[str]) -> Dict[str, any]:
        """Verify answer against multiple sources."""
        verification_results = []
        
        for source in sources:
            verification_prompt = f"""
            Question: {question}
            Proposed Answer: {answer}
            Source: {source}
            
            Is the proposed answer supported by the source? Answer with Yes/No and explanation:
            """
            
            verification = self.qa_model.answer_question(verification_prompt)
            verification_results.append({
                'source': source[:100] + "...",
                'verification': verification,
                'supported': verification.lower().startswith('yes')
            })
        
        support_ratio = sum(1 for v in verification_results if v['supported']) / len(verification_results)
        
        return {
            'verification_results': verification_results,
            'support_ratio': support_ratio,
            'is_reliable': support_ratio >= 0.6
        }
```

## 📄 Summarization

### Summarization Types

**1. Extractive Summarization**
```python
class ExtractiveSummarizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def extract_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """Extract most important sentences from text."""
        sentences = self.split_into_sentences(text)
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            score = self.score_sentence(sentence, text)
            sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in sentence_scores[:num_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return summary_sentences
    
    def score_sentence(self, sentence: str, full_text: str) -> float:
        """Score sentence importance (simplified)."""
        # Word frequency scoring
        words = sentence.lower().split()
        full_words = full_text.lower().split()
        word_freq = Counter(full_words)
        
        sentence_score = sum(word_freq[word] for word in words if word in word_freq)
        return sentence_score / len(words) if words else 0
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
```

**2. Abstractive Summarization**
```python
class AbstractiveSummarizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def summarize(self, text: str, max_length: int = 150, 
                 summary_type: str = 'general') -> str:
        """Generate abstractive summary."""
        # Create prompt based on summary type
        prompts = {
            'general': f"Summarize the following text:\n\n{text}\n\nSummary:",
            'key_points': f"Extract the key points from:\n\n{text}\n\nKey points:",
            'executive': f"Write an executive summary of:\n\n{text}\n\nExecutive Summary:"
        }
        
        prompt = prompts.get(summary_type, prompts['general'])
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                min_length=inputs.shape[1] + 20,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_response[len(prompt):].strip()
        
        return summary
    
    def multi_document_summarization(self, documents: List[str], 
                                   max_length: int = 200) -> str:
        """Summarize multiple documents."""
        # First, summarize each document individually
        individual_summaries = []
        for doc in documents:
            summary = self.summarize(doc, max_length=100)
            individual_summaries.append(summary)
        
        # Then, create a meta-summary
        combined_summaries = "\n\n".join(individual_summaries)
        meta_prompt = f"Create a comprehensive summary from these individual summaries:\n\n{combined_summaries}\n\nComprehensive Summary:"
        
        inputs = self.tokenizer.encode(meta_prompt, return_tensors='pt', max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.3,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(meta_prompt):].strip()
```

### Summary Quality Assessment

```python
class SummaryEvaluator:
    def __init__(self):
        self.metrics = {
            'coverage': self.evaluate_coverage,
            'coherence': self.evaluate_coherence,
            'conciseness': self.evaluate_conciseness,
            'faithfulness': self.evaluate_faithfulness
        }
    
    def evaluate_summary(self, summary: str, original_text: str) -> Dict[str, float]:
        """Evaluate summary quality across multiple dimensions."""
        scores = {}
        
        for metric, evaluator in self.metrics.items():
            scores[metric] = evaluator(summary, original_text)
        
        scores['overall'] = sum(scores.values()) / len(scores)
        return scores
    
    def evaluate_coverage(self, summary: str, original_text: str) -> float:
        """Evaluate how well summary covers original content."""
        summary_words = set(summary.lower().split())
        original_words = set(original_text.lower().split())
        
        if not original_words:
            return 0.0
        
        coverage = len(summary_words.intersection(original_words)) / len(original_words)
        return min(coverage * 2, 1.0)  # Scale up since summaries are shorter
    
    def evaluate_faithfulness(self, summary: str, original_text: str) -> float:
        """Evaluate factual consistency with original text."""
        # Simplified faithfulness check
        summary_sentences = summary.split('.')
        faithful_count = 0
        
        for sentence in summary_sentences:
            if sentence.strip():
                # Check if key concepts in sentence appear in original
                sentence_words = set(sentence.lower().split())
                original_words = set(original_text.lower().split())
                
                overlap = len(sentence_words.intersection(original_words))
                if overlap >= len(sentence_words) * 0.5:  # 50% overlap threshold
                    faithful_count += 1
        
        return faithful_count / len(summary_sentences) if summary_sentences else 0.0
```

## 🌍 Translation

### Neural Machine Translation

```python
class NeuralMachineTranslator:
    def __init__(self, model, tokenizer, source_lang: str, target_lang: str):
        self.model = model
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def translate(self, text: str, context: str = "") -> Dict[str, any]:
        """Translate text from source to target language."""
        # Prepare translation prompt
        if context:
            prompt = f"Context: {context}\nTranslate from {self.source_lang} to {self.target_lang}: {text}\nTranslation:"
        else:
            prompt = f"Translate from {self.source_lang} to {self.target_lang}: {text}\nTranslation:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + len(text.split()) * 2,  # Allow for length variation
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = full_response[len(prompt):].strip()
        
        # Calculate confidence (simplified)
        confidence = self.estimate_translation_confidence(text, translation)
        
        return {
            'translation': translation,
            'confidence': confidence,
            'source_language': self.source_lang,
            'target_language': self.target_lang
        }
    
    def batch_translate(self, texts: List[str]) -> List[Dict[str, any]]:
        """Translate multiple texts efficiently."""
        results = []
        
        for text in texts:
            result = self.translate(text)
            results.append(result)
        
        return results
    
    def estimate_translation_confidence(self, source: str, translation: str) -> float:
        """Estimate translation confidence (simplified)."""
        # Length ratio check
        source_len = len(source.split())
        translation_len = len(translation.split())
        
        if source_len == 0:
            return 0.0
        
        length_ratio = translation_len / source_len
        
        # Reasonable length ratios for most language pairs
        if 0.5 <= length_ratio <= 2.0:
            length_score = 1.0
        else:
            length_score = max(0, 1 - abs(length_ratio - 1))
        
        # Check for untranslated words (simplified)
        source_words = set(source.lower().split())
        translation_words = set(translation.lower().split())
        untranslated_ratio = len(source_words.intersection(translation_words)) / len(source_words)
        
        # Lower untranslated ratio is better for different languages
        translation_score = 1 - untranslated_ratio
        
        return (length_score + translation_score) / 2
```

### Multilingual Applications

```python
class MultilingualProcessor:
    def __init__(self, translators: Dict[str, NeuralMachineTranslator]):
        self.translators = translators
        self.supported_languages = list(translators.keys())
        
    def detect_language(self, text: str) -> str:
        """Detect language of input text (simplified)."""
        # In practice, use a proper language detection library
        language_indicators = {
            'en': ['the', 'and', 'is', 'in', 'to'],
            'es': ['el', 'la', 'y', 'en', 'de'],
            'fr': ['le', 'la', 'et', 'de', 'à'],
            'de': ['der', 'die', 'und', 'in', 'zu']
        }
        
        text_words = text.lower().split()
        language_scores = {}
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in text_words if word in indicators)
            language_scores[lang] = score
        
        return max(language_scores, key=language_scores.get) if language_scores else 'en'
    
    def translate_to_multiple_languages(self, text: str, 
                                      target_languages: List[str]) -> Dict[str, str]:
        """Translate text to multiple target languages."""
        source_lang = self.detect_language(text)
        translations = {}
        
        for target_lang in target_languages:
            if target_lang in self.supported_languages:
                translator_key = f"{source_lang}_{target_lang}"
                if translator_key in self.translators:
                    result = self.translators[translator_key].translate(text)
                    translations[target_lang] = result['translation']
        
        return translations
    
    def cross_lingual_qa(self, question: str, context: str, 
                        answer_language: str = 'en') -> Dict[str, any]:
        """Answer questions across languages."""
        # Detect languages
        question_lang = self.detect_language(question)
        context_lang = self.detect_language(context)
        
        # Translate to common language (English) if needed
        if question_lang != 'en':
            question_en = self.translators[f"{question_lang}_en"].translate(question)['translation']
        else:
            question_en = question
            
        if context_lang != 'en':
            context_en = self.translators[f"{context_lang}_en"].translate(context)['translation']
        else:
            context_en = context
        
        # Answer in English
        qa_system = GenerativeQA(self.model, self.tokenizer)  # Assume we have access
        answer_en = qa_system.answer_question(question_en, context_en)
        
        # Translate answer to desired language
        if answer_language != 'en':
            answer = self.translators[f"en_{answer_language}"].translate(answer_en)['translation']
        else:
            answer = answer_en
        
        return {
            'question': question,
            'answer': answer,
            'question_language': question_lang,
            'answer_language': answer_language,
            'intermediate_translations': {
                'question_en': question_en,
                'context_en': context_en,
                'answer_en': answer_en
            }
        }
```

## 💻 Code Generation

### Code Generation Systems

```python
class CodeGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_function(self, description: str, language: str = 'python') -> Dict[str, any]:
        """Generate function from natural language description."""
        prompt = f"""
        Language: {language}
        Task: {description}
        
        Generate a complete function with docstring:
        
        ```{language}
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 300,
                temperature=0.2,  # Lower temperature for more deterministic code
                do_sample=True,
                stop_sequences=['```'],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = full_response[len(prompt):].strip()
        
        # Extract function name and validate syntax
        function_name = self.extract_function_name(code, language)
        is_valid = self.validate_syntax(code, language)
        
        return {
            'code': code,
            'function_name': function_name,
            'language': language,
            'is_valid_syntax': is_valid,
            'description': description
        }
    
    def generate_class(self, class_description: str, methods: List[str], 
                      language: str = 'python') -> str:
        """Generate complete class with multiple methods."""
        prompt = f"""
        Create a {language} class: {class_description}
        
        Required methods:
        {chr(10).join(f"- {method}" for method in methods)}
        
        ```{language}
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 500,
                temperature=0.2,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()
    
    def fix_code_errors(self, code: str, error_message: str, 
                       language: str = 'python') -> str:
        """Fix code based on error message."""
        prompt = f"""
        Fix the following {language} code that has an error:
        
        Error: {error_message}
        
        Original code:
        ```{language}
        {code}
        ```
        
        Fixed code:
        ```{language}
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 400,
                temperature=0.1,  # Very low temperature for bug fixes
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()
    
    def validate_syntax(self, code: str, language: str) -> bool:
        """Validate code syntax."""
        try:
            if language.lower() == 'python':
                compile(code, '<string>', 'exec')
                return True
            # Add other language validators as needed
            return True  # Assume valid for unsupported languages
        except SyntaxError:
            return False
    
    def extract_function_name(self, code: str, language: str) -> str:
        """Extract function name from generated code."""
        if language.lower() == 'python':
            import re
            match = re.search(r'def\s+(\w+)\s*\(', code)
            return match.group(1) if match else None
        
        return None  # Add support for other languages
```

### Code Understanding and Documentation

```python
class CodeAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def explain_code(self, code: str, language: str = 'python') -> str:
        """Generate explanation for given code."""
        prompt = f"""
        Explain what this {language} code does:
        
        ```{language}
        {code}
        ```
        
        Explanation:
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                temperature=0.3,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()
    
    def generate_docstring(self, function_code: str, language: str = 'python') -> str:
        """Generate docstring for function."""
        prompt = f"""
        Generate a comprehensive docstring for this {language} function:
        
        ```{language}
        {function_code}
        ```
        
        Docstring:
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_dash():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.2,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()
    
    def suggest_improvements(self, code: str, language: str = 'python') -> List[str]:
        """Suggest code improvements."""
        prompt = f"""
        Suggest improvements for this {language} code:
        
        ```{language}
        {code}
        ```
        
        Suggestions:
        1.
        """
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                temperature=0.4,
                do_sample=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        suggestions_text = full_response[len(prompt):].strip()
        
        # Parse suggestions into list
        suggestions = []
        for line in suggestions_text.split('\n'):
            if line.strip() and (line.strip().startswith(tuple('123456789')) or line.strip().startswith('-')):
                suggestions.append(line.strip())
        
        return suggestions
```

## 🤖 Conversational Agents

### Dialogue Management

```python
class ConversationalAgent:
    def __init__(self, model, tokenizer, personality: str = "helpful"):
        self.model = model
        self.tokenizer = tokenizer
        self.personality = personality
        self.conversation_history = []
        self.context_window = 10  # Keep last 10 exchanges
        
    def respond(self, user_input: str, context: Dict[str, any] = None) -> Dict[str, any]:
        """Generate response to user input."""
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Build conversation context
        conversation_context = self.build_context(context)
        
        # Generate response
        response = self.generate_response(conversation_context)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Maintain context window
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window * 2:]
        
        return {
            'response': response,
            'conversation_id': id(self),
            'turn_number': len(self.conversation_history) // 2
        }
    
    def build_context(self, additional_context: Dict[str, any] = None) -> str:
        """Build conversation context for model input."""
        # System prompt based on personality
        system_prompts = {
            "helpful": "You are a helpful, knowledgeable assistant.",
            "creative": "You are a creative, imaginative assistant who loves storytelling.",
            "technical": "You are a technical expert who provides detailed, accurate information.",
            "casual": "You are a friendly, casual conversationalist."
        }
        
        context_parts = [system_prompts.get(self.personality, system_prompts["helpful"])]
        
        # Add additional context if provided
        if additional_context:
            for key, value in additional_context.items():
                context_parts.append(f"{key}: {value}")
        
        # Add conversation history
        for exchange in self.conversation_history[-self.context_window:]:
            role = exchange["role"].title()
            content = exchange["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts) + "\nAssistant:"
    
    def generate_response(self, context: str) -> str:
        """Generate response using the model."""
        inputs = self.tokenizer.encode(context, return_tensors='pt', max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(context):].strip()
        
        return response
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
```

### Multi-Turn Dialogue

```python
class MultiTurnDialogueManager:
    def __init__(self, agent: ConversationalAgent):
        self.agent = agent
        self.dialogue_state = {}
        self.intent_tracker = IntentTracker()
        
    def process_turn(self, user_input: str, session_id: str) -> Dict[str, any]:
        """Process a single dialogue turn."""
        # Track intent and entities
        intent_info = self.intent_tracker.analyze_intent(user_input)
        
        # Update dialogue state
        self.update_dialogue_state(session_id, intent_info, user_input)
        
        # Generate contextual response
        context = {
            'session_id': session_id,
            'current_intent': intent_info['intent'],
            'entities': intent_info['entities'],
            'dialogue_state': self.dialogue_state.get(session_id, {})
        }
        
        response_info = self.agent.respond(user_input, context)
        
        return {
            **response_info,
            'intent': intent_info['intent'],
            'entities': intent_info['entities'],
            'dialogue_state': self.dialogue_state.get(session_id, {})
        }
    
    def update_dialogue_state(self, session_id: str, intent_info: Dict, user_input: str):
        """Update dialogue state based on user input."""
        if session_id not in self.dialogue_state:
            self.dialogue_state[session_id] = {
                'active_topics': [],
                'user_preferences': {},
                'conversation_goals': []
            }
        
        state = self.dialogue_state[session_id]
        
        # Update active topics
        if intent_info['intent'] not in state['active_topics']:
            state['active_topics'].append(intent_info['intent'])
        
        # Extract and store user preferences
        for entity in intent_info['entities']:
            if entity['type'] == 'preference':
                state['user_preferences'][entity['category']] = entity['value']

class IntentTracker:
    def __init__(self):
        self.intent_patterns = {
            'question': ['what', 'how', 'why', 'when', 'where', '?'],
            'request': ['please', 'can you', 'could you', 'would you'],
            'complaint': ['problem', 'issue', 'wrong', 'error', 'broken'],
            'compliment': ['great', 'awesome', 'excellent', 'perfect', 'love']
        }
    
    def analyze_intent(self, text: str) -> Dict[str, any]:
        """Analyze user intent from text."""
        text_lower = text.lower()
        
        # Simple intent classification
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent] = score
        
        predicted_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'general'
        
        # Simple entity extraction (placeholder)
        entities = self.extract_entities(text)
        
        return {
            'intent': predicted_intent,
            'confidence': intent_scores.get(predicted_intent, 0) / len(self.intent_patterns[predicted_intent]),
            'entities': entities
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text (simplified)."""
        # This would typically use NER models
        entities = []
        
        # Simple pattern matching for demonstration
        import re
        
        # Extract dates
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
        dates = re.findall(date_pattern, text)
        for date in dates:
            entities.append({'type': 'date', 'value': date})
        
        # Extract numbers
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        for number in numbers:
            entities.append({'type': 'number', 'value': number})
        
        return entities
```

## 📚 Summary

Large Language Models enable powerful applications across diverse domains:

### Key Applications
- **Text Generation**: Creative writing, content creation, style transfer
- **Question Answering**: Extractive and generative QA, knowledge-augmented systems
- **Summarization**: Extractive and abstractive summarization, multi-document processing
- **Translation**: Neural machine translation, multilingual processing
- **Code Generation**: Function generation, code explanation, debugging assistance
- **Conversational Agents**: Dialogue management, multi-turn conversations, intent tracking

### Technical Considerations
- **Quality Control**: Content filtering, evaluation metrics, confidence estimation
- **Context Management**: Conversation history, dialogue state tracking
- **Multimodal Integration**: Cross-modal understanding and generation
- **Personalization**: User preferences, adaptive responses

### Future Directions
- **Improved Accuracy**: Better factual consistency and reduced hallucination
- **Enhanced Interactivity**: More natural dialogue and better context understanding
- **Specialized Applications**: Domain-specific fine-tuning and optimization
- **Integration Capabilities**: Seamless integration with existing systems and workflows

These applications demonstrate the transformative potential of LLMs across industries, from creative industries to software development, education, and customer service.