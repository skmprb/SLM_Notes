import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextChunker:
    
    def chunk_by_characters(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        # This method splits text into overlapping chunks of specified size
        # Parameters: text (input string), chunk_size (max characters per chunk), overlap (characters shared between consecutive chunks)
        
        # Initialize empty list to store the text chunks
        chunks = []
        # Set starting position for the first chunk
        start = 0
        # Get total length of input text for boundary checking
        text_len = len(text)
        
        # Continue creating chunks until we've processed the entire text
        while start < text_len:
            # Calculate end position for current chunk
            end = start + chunk_size
            # Extract substring from start to end position
            chunk = text[start:end]
            # Add the chunk to our list of chunks
            chunks.append(chunk)
            # Move start position forward, accounting for overlap (step back by overlap amount)
            start += chunk_size - overlap
        # Return the list of all created chunks
        return chunks
    
    def chunk_by_words(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size  - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[str]:
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n',text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_tokens(self, text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
        char_size = max_tokens * 4
        char_overlap = overlap_tokens * 4
        return self.chunk_by_characters(text, char_size, char_overlap)
    
    def paragraph_chunking(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        paragraphs = self.chunk_by_paragraphs(text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def similarity_chunking(self, text: str, similarity_threshold: float = 0.5, max_chunk_size: int = 1500, model_name: str = 'all-MiniLM-L6-v2') -> List[str]:
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [text]
        
        # Generate embeddings for each sentence
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences)
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Find split points where similarity drops below threshold
        chunks = []
        current_chunk = sentences[0]
        
        for i, sentence in enumerate(sentences[1:]):
            if similarities[i] < similarity_threshold or len(current_chunk) + len(sentence) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def hierarchical_chunking(self, text: str, parent_size: int = 2000, child_size: int = 500, overlap: int = 50) -> Dict:
        # Create parent chunks (large context)
        parent_chunks = self.chunk_by_characters(text, chunk_size=parent_size, overlap=overlap)
        
        hierarchy = []
        child_id_counter = 0
        
        for parent_id, parent_chunk in enumerate(parent_chunks):
            # Create child chunks from each parent
            child_chunks = self.chunk_by_characters(parent_chunk, chunk_size=child_size, overlap=overlap)
            
            children = []
            for child_chunk in child_chunks:
                children.append({
                    'child_id': child_id_counter,
                    'content': child_chunk,
                    'parent_id': parent_id
                })
                child_id_counter += 1
            
            hierarchy.append({
                'parent_id': parent_id,
                'parent_content': parent_chunk,
                'children': children,
                'num_children': len(children)
            })
        
        return {
            'hierarchy': hierarchy,
            'total_parents': len(parent_chunks),
            'total_children': child_id_counter
        }
    
    def recursive_chunking(self, text: str, separators: List[str] = None, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if separators is None:
            # Default hierarchy: paragraphs -> sentences -> words -> characters
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
            # Base case: no more separators or text is small enough
            if not separators or len(text) <= chunk_size:
                return [text] if text else []
            
            # Try current separator
            separator = separators[0]
            remaining_separators = separators[1:]
            
            # Split by current separator
            if separator == "":
                # Character-level split (last resort)
                splits = list(text)
            else:
                splits = text.split(separator)
            
            # Merge splits into chunks
            chunks = []
            current_chunk = ""
            
            for split in splits:
                if not split:
                    continue
                    
                # Add separator back (except for character-level)
                split_with_sep = split + separator if separator != "" else split
                
                # If single split is too large, recursively split with next separator
                if len(split_with_sep) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Recursive call with next separator
                    sub_chunks = split_text(split_with_sep, remaining_separators, chunk_size, overlap)
                    chunks.extend(sub_chunks)
                    continue
                
                # Try to add to current chunk
                if len(current_chunk) + len(split_with_sep) <= chunk_size:
                    current_chunk += split_with_sep
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Handle overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + split_with_sep
                    else:
                        current_chunk = split_with_sep
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        return split_text(text, separators, chunk_size, overlap)
    
    def chunk_with_metadata(self, data: Dict, method:str = 'words', **kwargs) -> List[Dict]:
        
        text = data.get('content', '')
        
        if method == 'characters':
            chunks = self.chunk_by_characters(text, **kwargs)
        elif method == 'words':
            chunks = self.chunk_by_words(text, **kwargs)
        elif method == 'sentences':
            chunks = self.chunk_by_sentences(text, **kwargs)
        elif method == 'paragraphs':
            chunks = self.chunk_by_paragraphs(text)
        elif method == 'tokens':
            chunks = self.chunk_by_tokens(text, **kwargs)
        elif method == 'paragraph':
            chunks = self.paragraph_chunking(text, **kwargs)
        elif method == 'similarity':
            chunks = self.similarity_chunking(text, **kwargs)
        elif method == 'recursive':
            chunks = self.recursive_chunking(text, **kwargs)
        elif method == 'hierarchical':
            # Hierarchical returns different structure, handle separately
            hierarchy_data = self.hierarchical_chunking(text, **kwargs)
            result = []
            for parent in hierarchy_data['hierarchy']:
                for child in parent['children']:
                    result.append({
                        'chunk_id': child['child_id'],
                        'content': child['content'],
                        'parent_id': child['parent_id'],
                        'parent_content': parent['parent_content'],
                        'source': data.get('source', ''),
                        'type': data.get('type', ''),
                        'total_chunks': hierarchy_data['total_children']
                    })
            return result
        else:
            chunks = [text]
            
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'chunk_id' : i,
                'content' : chunk,
                'source' : data.get('source', ''),
                'type' : data.get('type', ''),
                'total_chunks' : len(chunks)
            })
        
        return result
    


