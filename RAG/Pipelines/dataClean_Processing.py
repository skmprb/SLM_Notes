import re
from typing import Dict, List

class DataPreprocessor:
    
    def clean_text(self,text:str) -> str:
        
        text = re.sub(r'\s+', ' ', text) # Remove extra spaces
        text = re.sub(r'\n+', ' ', text) # Remove newlines
        text = text.strip() # Remove leading and trailing spaces
        return text
    
    def remove_special_chars(self, text: str) -> str:
        # The regex pattern [^\w\s\.\,\!\?\-] breaks down as follows:
        # ^ at the start of [] means "NOT" (negation)
        # \w matches word characters (letters, digits, underscore)
        # \s matches whitespace characters (spaces, tabs, newlines)
        # \. matches literal period/dot (escaped because . is a special regex character)
        # \, matches literal comma (escaped for consistency, though not strictly needed)
        # \! matches literal exclamation mark (escaped for consistency)
        # \? matches literal question mark (escaped because ? is a special regex character)
        # \- matches literal hyphen/dash (escaped because - can be special in character classes)
        # So this pattern matches any character that is NOT a word character, whitespace, or basic punctuation
        # it removes special characters while preserving letters, numbers, spaces, and common punctuation marks.
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text) # Remove special characters except basic punctuation
        return text
    
    
    def lowercase(self,text: str) -> str:
        return text.lower()
    
    def remove_urls(self, text: str) -> str:
        # The regex pattern r'http\S+|www\.\S+' breaks down as follows:
        # http\S+ matches URLs starting with "http" followed by one or more non-whitespace characters
        # | is the OR operator, allowing either pattern to match
        # www\.\S+ matches URLs starting with "www." followed by one or more non-whitespace characters
        # \. matches a literal dot (escaped because . is a special regex character that matches any character)
        # \S+ matches one or more non-whitespace characters (letters, numbers, symbols, but not spaces/tabs/newlines)
        # This pattern will match and remove both http/https URLs and www URLs from the text
        return re.sub(r'http\S+|www\.\S+', '', text)
    
    def remove_emails(self, text: str) -> str:
        # The regex pattern r'\S+@\S+' breaks down as follows:
        # \S+ matches one or more non-whitespace characters (letters, numbers, symbols, but not spaces/tabs/newlines)
        # @ matches the literal @ symbol
        # \S+ matches one or more non-whitespace characters after the @ symbol
        # This pattern will match email addresses by finding sequences of non-whitespace characters
        # separated by an @ symbol, effectively removing email addresses from the text
        return re.sub(r'\S+@\S+', '', text) # Remove email addresses
    
    def remove_numbers(self, text: str) -> str:
        # Modified to only remove phone numbers, not all numbers like costs or quantities
        # Phone number patterns to match:
        # - (123) 456-7890 format
        # - 123-456-7890 format  
        # - 123.456.7890 format
        # - 123 456 7890 format
        # - +1 123 456 7890 international format
        # - 1234567890 10-digit format
        phone_patterns = [
            r'\+?1?\s*\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',  # (123) 456-7890
            r'\+?1?\s*\d{3}[-.\s]\d{3}[-.\s]\d{4}',     # 123-456-7890 or 123.456.7890
            r'\b\d{10}\b',                               # 1234567890 (10 digits only)
            r'\+1\s*\d{3}\s*\d{3}\s*\d{4}'             # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        
        return text
        
    # def remove_numbers(self, text: str) -> str:
    #     """Remove standalone numbers"""
    #     return re.sub(r'\b\d+\b', '', text)       
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        
        return text    
    def preprocess(self, data: Dict, operations: List[str] = None) -> Dict:
        if operations is None:
            operations = ['clean_text', 'remove_urls', 'remove_emails']
            
        text = data['content']
        for op in operations:
            if hasattr(self, op):
                text = getattr(self, op)(text)
                
        return {**data, 'content': text, 'preprocessed': True}
    
    def batch_preprocess(self, data_list: List[Dict], operations: List[str] = None) -> List[Dict]:
        return [self.preprocess(data, operations) for data in data_list]
