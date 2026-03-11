import re
from typing import Dict, List
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
import fitz
import docx
import json
import pandas as pd

class DocumentParser:
    def parse_pdf_pypdf2(self, pdf_path: str) -> Dict:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = []
            for i, page in enumerate(reader.pages):
                pages.append({
                    'page_num' : i+1,
                    'text': page.extract_text()
                })
        return {'source': pdf_path, 'pages': pages, 'type': 'pdf'}
    
    def parse_pdf_pdfplumber(self, pdf_path: str) -> Dict:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                pages.append({
                    'page_num' : i+1,
                    'text': page.extract_text(),
                    'tables': page.extract_tables()
                })
        return {'source': pdf_path, 'pages': pages, 'type': 'pdf'}
    
    def parse_pdf_pymupdf(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            pages.append({
                'page_num' : i+1,
                'text': page.get_text(),
                'images': len(page.get_images())
            })
        metadata = doc.metadata
        return {'source': pdf_path, 'pages': pages, 'type': 'pdf', 'metadata': metadata}
    
    def parse_html(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        title = soup.find('title').text if soup.find('title') else ''
        headings = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
        paragraphs = [p.text.strip() for p in soup.find_all('p')]
        links = [a.get('href') for a in soup.find_all('a', href = True)]
        return {
            'source': 'html_content',
            'title': title,
            'headings': headings,
            'paragraphs': paragraphs,
            'links': links,
            'full_text': soup.get_text(separator='\n', strip=True),
            'type': 'html'
        }
    
    def parse_docx(self, docx_path: str) -> Dict:
        doc = docx.Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        tables = []
        for table in doc.tables:
            table_data = [[cell.text for cell in row.cells] for row in table.rows]
            tables.append(table_data)
        
        return {
            'source': docx_path,
            'paragraphs': paragraphs,
            'tables': tables,
            'type': 'docx'
        }
        
    def parse_json(self, json_path: str) -> Dict:
        with open(json_path, 'r') as file:
            data = json.load(file)
        return {'source': json_path, 'data': data, 'type': 'json'}
    
    def parse_csv(self, csv_path: str) -> Dict:
        df = pd.read_csv(csv_path)
        return {
            'source': csv_path,
            'columns' : df.columns.tolist(),
            'rows': df.to_dict('records'),
            'type': 'csv',
            'text': df.to_string()
        }
    
    def parse_markdown(self, md_path: str) -> Dict:
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        
        code_blocks = re.findall(r'```(.*?)```', content)
        
        return{
            'source': md_path,
            'headings': headings,
            'code_blocks': code_blocks,
            'full_text': content,
            'type': 'markdown'
        }
    
    def extract_text_from_parsed(self, parsed_data: Dict) -> str:
        
        doc_type = parsed_data.get('type')
        
        if doc_type == 'pdf':
            return '\n'.join([page['text'] for page in parsed_data['pages']])
        elif doc_type == 'html':
            return parsed_data['full_text']
        elif doc_type == 'docx':
            return '\n'.join(parsed_data['paragraphs'])
        elif doc_type == 'json':
            return json.dumps(parsed_data['data'], indent=2)
        elif doc_type == 'csv':
            return parsed_data['text']
        elif doc_type == 'markdown':
            return parsed_data['full_text']
        else:
            return str(parsed_data)
        
        