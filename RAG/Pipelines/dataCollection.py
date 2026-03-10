import os
import requests 
from pathlib import Path
import PyPDF2
from typing import List, Dict
from bs4 import BeautifulSoup
import spacy
import pdfplumber
import docx
from pypdf import PdfReader as PyPDFReader
import fitz

class DataCollector:
    # Initialize the DataCollector class with a default output directory
    # This constructor sets up the basic configuration for data collection operations
    def __init__(self, output_dir="data/raw"):
        # Convert the output directory path string to a Path object for better path handling
        # Path objects provide cross-platform compatibility and useful path manipulation methods
        self.output_dir = Path(output_dir)
        # Create the output directory and any necessary parent directories if they don't exist
        # parents=True creates intermediate directories, exist_ok=True prevents errors if directory already exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_pdf(self, pdf_path: str) -> Dict:
        # Open the PDF file in binary read mode to access its raw content
        with open(pdf_path, 'rb') as file:
            # Create a PyPDF2 PdfReader object to parse and read the PDF file
            reader = PyPDF2.PdfReader(file)
            # Extract text from all pages by iterating through each page and joining the text with newlines
            text = "\n".join([page.extract_text() for page in reader.pages])
        # Return a dictionary containing the source path, file type, and extracted text content
        return {"source": pdf_path, "type":"pdf", "content":text}
    
    def collect_url(self,url:str) -> Dict:
        """
        The function `collect_url` takes a URL as input, retrieves the content from the URL, and returns
        a dictionary containing the source URL, type, and the text content extracted from the URL.
        
        :param url: The `url` parameter in the `collect_url` function is a string that represents the
        URL of a webpage from which you want to collect content. The function sends a GET request to the
        URL, extracts the text content from the webpage using BeautifulSoup, and returns a dictionary
        containing the source URL, content
        :type url: str
        :return: The function `collect_url` is returning a dictionary with three key-value pairs:
        1. "source": the input URL
        2. "type": the string "url"
        3. "content": the text content extracted from the URL after parsing it with BeautifulSoup
        """
        
        # Send an HTTP GET request to the specified URL and store the response object
        response = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        # Parse the HTML content from the response using BeautifulSoup with the html.parser
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract all text content from the parsed HTML, using newlines as separators and stripping whitespace
        text = soup.get_text(separator='\n',strip=True)
        # Return a dictionary containing the source URL, content type, and extracted text
        return {"source": url, "type":"url", "content":text}    
    
    def collect_text_file(self, file_path:str) -> Dict:
        # Open the text file at the specified path in read mode with UTF-8 encoding
        # UTF-8 encoding ensures proper handling of special characters and international text
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the entire content of the file into a string variable
            text = f.read()
        # Return a dictionary containing the source file path, file type, and the complete text content
        return {"source": file_path, "type":"text", "content":text}
    
    def collect_api(self, api_url:str, headers: Dict = None) -> Dict:
        # Send an HTTP GET request to the specified API URL with optional custom headers
        # If no headers are provided, use an empty dictionary as default
        response = requests.get(api_url, headers=headers or {})
        # Return a dictionary containing the source API URL, content type, and the response text content
        return {"source": api_url, "type":"api", "content":response.text}
    
    def save_collected_data(self, data: List[Dict], filename:str = "collected_data.txt"):
        # Construct the full output file path by combining the instance's output directory with the provided filename
        output_path = self.output_dir / filename
        # Open the output file in write mode with UTF-8 encoding to handle special characters properly
        with open(output_path, 'w', encoding='utf-8') as f:
            # Iterate through each data item (dictionary) in the provided data list
            for item in data:
                # Write the source information and type of each data item to the file
                f.write(f"Source: {item['source']}\nType: {item['type']}\n")
                # Write the actual content followed by a separator line of 80 dashes and blank lines for readability
                f.write(f"Content:\n{item['content']}\n{'-'*80}\n\n")
        # Return the full path of the saved file as a string for reference
        return str(output_path)
    
    def collect_pdf_spacy(self, pdf_path: str) -> Dict:
        # Load the English language model from spaCy for natural language processing
        # This model provides tokenization, part-of-speech tagging, and other NLP capabilities
        nlp = spacy.load("en_core_web_sm")
        # Open the PDF file in binary read mode to access its raw content
        with open(pdf_path, 'rb') as file:
            # Create a PyPDF2 PdfReader object to parse and read the PDF file
            reader = PyPDF2.PdfReader(file)
            # Extract text from all pages by iterating through each page and joining the text with newlines
            text = "\n".join([page.extract_text() for page in reader.pages])
        # Process the extracted text through spaCy's NLP pipeline to create a Doc object
        # This enables advanced text analysis capabilities like entity recognition and linguistic analysis
        doc = nlp(text)
        # Return a dictionary containing the source path, processing type, and the processed text content
        return {"source": pdf_path, "type":"pdf_spacy", "content":doc.text}    

    def collect_docx(self, docx_path: str) -> Dict:
        # Open the DOCX file using the python-docx library to read its content
        doc = docx.Document(docx_path)
        # Extract text from all paragraphs in the document and join them with newlines
        text = "\n".join([para.text for para in doc.paragraphs])
        # Return a dictionary containing the source path, file type, and extracted text content
        return {"source": docx_path, "type":"docx", "content":text}
    
    def collect_pdf_pdfplumber(self, pdf_path: str) -> Dict:
        # Open the PDF file using pdfplumber to access its content
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages by iterating through each page and joining the text with newlines
            text = "\n".join([page.extract_text() for page in pdf.pages])
        # Return a dictionary containing the source path, file type, and extracted text content
        return {"source": pdf_path, "type":"pdf_pdfplumber", "content":text}
    
    def collect_pdf_pymupdf(self, pdf_path: str) -> Dict:
        # Open the PDF file using PyMuPDF (fitz) to access its content
        doc = fitz.open(pdf_path)
        # Extract text from all pages by iterating through each page and joining the text with newlines
        text = "\n".join([page.get_text() for page in doc])
        # Return a dictionary containing the source path, file type, and extracted text content
        return {"source": pdf_path, "type":"pdf_pymupdf", "content":text}
    
    def collect_url_selenium(self, url:str) -> Dict:
    
        # Import the webdriver module from selenium to control web browsers programmatically
        from selenium import webdriver
        # Import the Service class to manage the ChromeDriver service
        from selenium.webdriver.chrome.options import Options
    
        # Create a ChromeOptions object to configure browser settings
        options = Options()
        # Add the headless argument to run Chrome without a visible browser window
        options.add_argument("--headless")  # Run in headless mode (without opening a
        # Initialize a Chrome WebDriver instance with the configured options
        driver = webdriver.Chrome( options=options)    
        # Navigate to the specified URL using the WebDriver
        driver.get(url)
        # Find the body element of the webpage and extract all visible text content
        text = driver.find_element("tag name", "body").text
        # Close the browser and terminate the WebDriver session to free up resources
        driver.quit()
        # Return a dictionary containing the source URL, collection method type, and extracted text content
        return {"source": url, "type":"url_selenium", "content":text} 
   
    def collect_csv(self, csv_path: str) -> Dict:
        import pandas as pd
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        # Convert the DataFrame to a string representation for storage
        text = df.to_string(index=False)
        # Return a dictionary containing the source path, file type, and the string representation of the CSV content
        return {"source": csv_path, "type":"csv", "content":text}
    
    def collect_json(self, json_path: str) -> Dict:
        import json
        # Open the JSON file and load its content into a Python dictionary
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Return a dictionary containing the source path, file type, and the string representation of the JSON content
        return {"source": json_path, "type":"json", "content":data}
    
    
