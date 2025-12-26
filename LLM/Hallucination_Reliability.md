# Hallucination and Reliability in Large Language Models

## Overview

Hallucination in LLMs refers to the generation of content that appears plausible but is factually incorrect, unsupported by training data, or inconsistent with provided context. Reliability encompasses the model's ability to produce consistent, accurate, and trustworthy outputs across various scenarios.

## Hallucination Types

### 1. Factual Hallucinations

Generation of false factual information that contradicts established knowledge.

```python
class FactualHallucinationDetector:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.fact_checker = FactChecker()
    
    def detect_factual_errors(self, generated_text, context=None):
        """Detect factual inconsistencies in generated text"""
        claims = self.extract_claims(generated_text)
        errors = []
        
        for claim in claims:
            # Check against knowledge base
            kb_result = self.kb.verify(claim)
            if kb_result == "FALSE":
                errors.append({
                    'claim': claim,
                    'type': 'factual_contradiction',
                    'confidence': kb_result.confidence
                })
            
            # Cross-reference with context
            if context and not self.is_supported_by_context(claim, context):
                errors.append({
                    'claim': claim,
                    'type': 'context_unsupported',
                    'severity': 'high'
                })
        
        return errors
    
    def extract_claims(self, text):
        """Extract verifiable claims from text"""
        # Use NER and dependency parsing
        claims = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            if self.contains_factual_claim(sent):
                claims.append({
                    'text': sent.text,
                    'entities': [(ent.text, ent.label_) for ent in sent.ents],
                    'relations': self.extract_relations(sent)
                })
        
        return claims
```

### 2. Contextual Hallucinations

Information that contradicts or is unsupported by the provided context.

```python
class ContextualConsistencyChecker:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.entailment_model = pipeline('text-classification', 
                                       model='microsoft/deberta-large-mnli')
    
    def check_contextual_consistency(self, context, generated_text):
        """Check if generated text is consistent with context"""
        context_embeddings = self.similarity_model.encode([context])
        gen_sentences = self.split_sentences(generated_text)
        
        inconsistencies = []
        
        for sentence in gen_sentences:
            # Semantic similarity check
            sent_embedding = self.similarity_model.encode([sentence])
            similarity = cosine_similarity(context_embeddings, sent_embedding)[0][0]
            
            if similarity < 0.3:  # Low similarity threshold
                # Check for entailment
                result = self.entailment_model(f"{context} [SEP] {sentence}")
                if result['label'] == 'CONTRADICTION':
                    inconsistencies.append({
                        'sentence': sentence,
                        'type': 'contextual_contradiction',
                        'similarity': similarity,
                        'entailment_score': result['score']
                    })
        
        return inconsistencies
    
    def split_sentences(self, text):
        """Split text into sentences for analysis"""
        import nltk
        return nltk.sent_tokenize(text)
```

### 3. Logical Hallucinations

Internally inconsistent statements or logical contradictions within the generated text.

```python
class LogicalConsistencyAnalyzer:
    def __init__(self):
        self.logic_parser = LogicParser()
        self.contradiction_detector = ContradictionDetector()
    
    def detect_logical_inconsistencies(self, text):
        """Detect logical contradictions within text"""
        statements = self.extract_logical_statements(text)
        contradictions = []
        
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                if self.are_contradictory(stmt1, stmt2):
                    contradictions.append({
                        'statement1': stmt1,
                        'statement2': stmt2,
                        'type': 'logical_contradiction',
                        'positions': (i, j)
                    })
        
        return contradictions
    
    def extract_logical_statements(self, text):
        """Extract statements that can be logically analyzed"""
        statements = []
        doc = self.nlp(text)
        
        for sent in doc.sents:
            # Look for definitive statements
            if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' 
                   for token in sent):
                logical_form = self.logic_parser.parse(sent.text)
                if logical_form:
                    statements.append({
                        'text': sent.text,
                        'logical_form': logical_form,
                        'certainty': self.extract_certainty(sent)
                    })
        
        return statements
```

## Faithfulness

### Faithfulness Metrics

Measuring how well generated text adheres to source information.

```python
class FaithfulnessEvaluator:
    def __init__(self):
        self.nli_model = pipeline('text-classification',
                                model='facebook/bart-large-mnli')
        self.qa_model = pipeline('question-answering',
                               model='deepset/roberta-base-squad2')
    
    def evaluate_faithfulness(self, source_text, generated_text):
        """Comprehensive faithfulness evaluation"""
        scores = {}
        
        # 1. Entailment-based faithfulness
        scores['entailment'] = self.entailment_faithfulness(source_text, generated_text)
        
        # 2. Question-answering faithfulness
        scores['qa_faithfulness'] = self.qa_faithfulness(source_text, generated_text)
        
        # 3. Factual consistency
        scores['factual_consistency'] = self.factual_consistency(source_text, generated_text)
        
        # 4. Information coverage
        scores['coverage'] = self.information_coverage(source_text, generated_text)
        
        return scores
    
    def entailment_faithfulness(self, source, generated):
        """Check if generated text is entailed by source"""
        result = self.nli_model(f"{source} [SEP] {generated}")
        entailment_score = result['score'] if result['label'] == 'ENTAILMENT' else 0
        return entailment_score
    
    def qa_faithfulness(self, source, generated):
        """Generate questions from generated text and answer from source"""
        questions = self.generate_questions(generated)
        faithfulness_scores = []
        
        for question in questions:
            # Answer from source
            source_answer = self.qa_model(question=question, context=source)
            
            # Answer from generated text
            gen_answer = self.qa_model(question=question, context=generated)
            
            # Compare answers
            similarity = self.compute_answer_similarity(
                source_answer['answer'], gen_answer['answer']
            )
            faithfulness_scores.append(similarity)
        
        return np.mean(faithfulness_scores) if faithfulness_scores else 0
    
    def generate_questions(self, text):
        """Generate questions from text for faithfulness checking"""
        # Use T5 or similar model for question generation
        question_generator = pipeline('text2text-generation',
                                    model='valhalla/t5-small-qg-hl')
        
        questions = []
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            try:
                result = question_generator(f"generate question: {sentence}")
                questions.append(result[0]['generated_text'])
            except:
                continue
        
        return questions
```

### Attribution and Source Tracking

```python
class AttributionTracker:
    def __init__(self):
        self.source_embeddings = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def track_attributions(self, sources, generated_text):
        """Track which parts of generated text come from which sources"""
        # Encode all sources
        for i, source in enumerate(sources):
            self.source_embeddings[i] = self.embedding_model.encode(
                nltk.sent_tokenize(source)
            )
        
        # Analyze generated text
        gen_sentences = nltk.sent_tokenize(generated_text)
        attributions = []
        
        for gen_sent in gen_sentences:
            gen_embedding = self.embedding_model.encode([gen_sent])
            
            best_match = self.find_best_source_match(gen_embedding)
            attributions.append({
                'sentence': gen_sent,
                'source_id': best_match['source_id'],
                'source_sentence': best_match['sentence'],
                'confidence': best_match['similarity']
            })
        
        return attributions
    
    def find_best_source_match(self, gen_embedding):
        """Find best matching source sentence"""
        best_match = {'similarity': 0, 'source_id': None, 'sentence': None}
        
        for source_id, source_embs in self.source_embeddings.items():
            similarities = cosine_similarity(gen_embedding, source_embs)[0]
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            
            if max_sim > best_match['similarity']:
                best_match.update({
                    'similarity': max_sim,
                    'source_id': source_id,
                    'sentence_idx': max_sim_idx
                })
        
        return best_match
```

## Grounding

### Knowledge Grounding

Ensuring generated content is grounded in reliable knowledge sources.

```python
class KnowledgeGroundingSystem:
    def __init__(self, knowledge_graphs, fact_databases):
        self.knowledge_graphs = knowledge_graphs
        self.fact_databases = fact_databases
        self.entity_linker = EntityLinker()
        self.relation_extractor = RelationExtractor()
    
    def ground_generation(self, text, domain=None):
        """Ground generated text in knowledge sources"""
        grounding_results = {
            'entities': self.ground_entities(text),
            'relations': self.ground_relations(text),
            'facts': self.ground_facts(text, domain),
            'overall_score': 0
        }
        
        # Calculate overall grounding score
        grounding_results['overall_score'] = self.calculate_grounding_score(
            grounding_results
        )
        
        return grounding_results
    
    def ground_entities(self, text):
        """Ground entities in knowledge graphs"""
        entities = self.entity_linker.extract_and_link(text)
        grounded_entities = []
        
        for entity in entities:
            kg_matches = []
            for kg in self.knowledge_graphs:
                matches = kg.find_entity(entity['text'])
                kg_matches.extend(matches)
            
            grounded_entities.append({
                'entity': entity,
                'kg_matches': kg_matches,
                'confidence': max([m['confidence'] for m in kg_matches]) if kg_matches else 0
            })
        
        return grounded_entities
    
    def ground_relations(self, text):
        """Ground relations in knowledge bases"""
        relations = self.relation_extractor.extract(text)
        grounded_relations = []
        
        for relation in relations:
            # Check if relation exists in knowledge graphs
            for kg in self.knowledge_graphs:
                if kg.has_relation(relation['subject'], relation['predicate'], relation['object']):
                    grounded_relations.append({
                        'relation': relation,
                        'grounded': True,
                        'source': kg.name
                    })
                    break
            else:
                grounded_relations.append({
                    'relation': relation,
                    'grounded': False,
                    'source': None
                })
        
        return grounded_relations
    
    def ground_facts(self, text, domain):
        """Ground factual claims in fact databases"""
        claims = self.extract_factual_claims(text)
        grounded_facts = []
        
        for claim in claims:
            fact_check_results = []
            
            # Check against domain-specific databases
            if domain and domain in self.fact_databases:
                result = self.fact_databases[domain].verify(claim)
                fact_check_results.append(result)
            
            # Check against general fact databases
            for db in self.fact_databases.values():
                result = db.verify(claim)
                fact_check_results.append(result)
            
            grounded_facts.append({
                'claim': claim,
                'verification_results': fact_check_results,
                'consensus': self.calculate_consensus(fact_check_results)
            })
        
        return grounded_facts
```

### Retrieval-Augmented Grounding

```python
class RetrievalGroundingSystem:
    def __init__(self, retrieval_system, reranker):
        self.retrieval_system = retrieval_system
        self.reranker = reranker
        self.citation_generator = CitationGenerator()
    
    def ground_with_retrieval(self, query, generated_text, top_k=10):
        """Ground generated text using retrieval system"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieval_system.retrieve(query, top_k=top_k)
        
        # Rerank based on generated text
        reranked_docs = self.reranker.rerank(generated_text, retrieved_docs)
        
        # Check grounding for each sentence
        sentences = nltk.sent_tokenize(generated_text)
        grounding_results = []
        
        for sentence in sentences:
            sentence_grounding = self.check_sentence_grounding(
                sentence, reranked_docs[:5]
            )
            grounding_results.append(sentence_grounding)
        
        # Generate citations
        citations = self.citation_generator.generate_citations(
            generated_text, reranked_docs, grounding_results
        )
        
        return {
            'sentence_grounding': grounding_results,
            'supporting_documents': reranked_docs,
            'citations': citations,
            'overall_grounding_score': np.mean([r['score'] for r in grounding_results])
        }
    
    def check_sentence_grounding(self, sentence, documents):
        """Check if sentence is grounded in retrieved documents"""
        best_support = {'score': 0, 'document': None, 'passage': None}
        
        for doc in documents:
            # Split document into passages
            passages = self.split_into_passages(doc['content'])
            
            for passage in passages:
                # Check entailment
                entailment_score = self.check_entailment(passage, sentence)
                
                if entailment_score > best_support['score']:
                    best_support.update({
                        'score': entailment_score,
                        'document': doc,
                        'passage': passage
                    })
        
        return {
            'sentence': sentence,
            'score': best_support['score'],
            'supporting_document': best_support['document'],
            'supporting_passage': best_support['passage'],
            'grounded': best_support['score'] > 0.7
        }
```

## Reliability Assessment Framework

```python
class ReliabilityAssessment:
    def __init__(self):
        self.hallucination_detector = FactualHallucinationDetector()
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.grounding_system = KnowledgeGroundingSystem()
        self.uncertainty_estimator = UncertaintyEstimator()
    
    def assess_reliability(self, context, generated_text, sources=None):
        """Comprehensive reliability assessment"""
        assessment = {
            'hallucination_analysis': {},
            'faithfulness_scores': {},
            'grounding_analysis': {},
            'uncertainty_analysis': {},
            'overall_reliability': 0
        }
        
        # 1. Hallucination Detection
        assessment['hallucination_analysis'] = {
            'factual_errors': self.hallucination_detector.detect_factual_errors(generated_text, context),
            'contextual_inconsistencies': self.check_contextual_consistency(context, generated_text),
            'logical_contradictions': self.detect_logical_inconsistencies(generated_text)
        }
        
        # 2. Faithfulness Evaluation
        if sources:
            assessment['faithfulness_scores'] = self.faithfulness_evaluator.evaluate_faithfulness(
                ' '.join(sources), generated_text
            )
        
        # 3. Grounding Analysis
        assessment['grounding_analysis'] = self.grounding_system.ground_generation(generated_text)
        
        # 4. Uncertainty Estimation
        assessment['uncertainty_analysis'] = self.uncertainty_estimator.estimate_uncertainty(
            generated_text, context
        )
        
        # 5. Calculate Overall Reliability Score
        assessment['overall_reliability'] = self.calculate_overall_reliability(assessment)
        
        return assessment
    
    def calculate_overall_reliability(self, assessment):
        """Calculate weighted overall reliability score"""
        weights = {
            'hallucination': 0.3,
            'faithfulness': 0.25,
            'grounding': 0.25,
            'uncertainty': 0.2
        }
        
        # Calculate component scores (higher is better)
        hallucination_score = 1.0 - self.calculate_hallucination_penalty(
            assessment['hallucination_analysis']
        )
        
        faithfulness_score = np.mean(list(assessment['faithfulness_scores'].values())) \
            if assessment['faithfulness_scores'] else 0.5
        
        grounding_score = assessment['grounding_analysis']['overall_score']
        
        uncertainty_score = 1.0 - assessment['uncertainty_analysis']['overall_uncertainty']
        
        overall_score = (
            weights['hallucination'] * hallucination_score +
            weights['faithfulness'] * faithfulness_score +
            weights['grounding'] * grounding_score +
            weights['uncertainty'] * uncertainty_score
        )
        
        return max(0, min(1, overall_score))
```

## Mitigation Strategies

### Real-time Hallucination Prevention

```python
class HallucinationPrevention:
    def __init__(self, model, knowledge_base):
        self.model = model
        self.knowledge_base = knowledge_base
        self.fact_checker = RealTimeFactChecker()
    
    def generate_with_prevention(self, prompt, max_length=512):
        """Generate text with real-time hallucination prevention"""
        generated_tokens = []
        current_text = prompt
        
        for _ in range(max_length):
            # Get next token probabilities
            logits = self.model.get_next_token_logits(current_text)
            
            # Apply hallucination prevention filters
            filtered_logits = self.apply_prevention_filters(
                current_text, logits
            )
            
            # Sample next token
            next_token = self.sample_token(filtered_logits)
            generated_tokens.append(next_token)
            current_text += next_token
            
            # Check for stopping conditions
            if next_token == self.model.eos_token:
                break
        
        return ''.join(generated_tokens)
    
    def apply_prevention_filters(self, context, logits):
        """Apply filters to prevent hallucination"""
        # 1. Fact-checking filter
        logits = self.fact_checking_filter(context, logits)
        
        # 2. Consistency filter
        logits = self.consistency_filter(context, logits)
        
        # 3. Uncertainty-based filter
        logits = self.uncertainty_filter(context, logits)
        
        return logits
    
    def fact_checking_filter(self, context, logits):
        """Filter tokens that would create factual errors"""
        # Get top-k candidate tokens
        top_k_indices = torch.topk(logits, k=50).indices
        
        filtered_logits = logits.clone()
        
        for token_idx in top_k_indices:
            token = self.model.tokenizer.decode([token_idx])
            hypothetical_text = context + token
            
            # Quick fact check
            if self.fact_checker.is_likely_false(hypothetical_text):
                filtered_logits[token_idx] *= 0.1  # Heavily penalize
        
        return filtered_logits
```

This comprehensive framework provides tools and techniques for detecting, measuring, and mitigating hallucinations while ensuring reliability and proper grounding of LLM outputs. The implementation focuses on practical approaches that can be integrated into production systems.