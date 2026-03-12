import re
from typing import List, Dict

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
    
    def semantic_chunking(self,text: str, max_chunk_size: int = 1000) -> List[str]:
        
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
        elif method == 'semantic':
            chunks = self.semantic_chunking(text, **kwargs)
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
            
