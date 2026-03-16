# # Data collection tests

# from dataCollection import DataCollector

# collector = DataCollector(output_dir="data/raw")

# collected_data = []

# collected_data.append(collector.collect_pdf(r"C:\Users\Administrator\Downloads\sravanResume.pdf"))

# collected_data.append(collector.collect_url("https://google.github.io/adk-docs/get-started/python/"))

# collected_data.append(collector.collect_text_file(r"C:\Users\Administrator\Desktop\agent.txt"))

# #collected_data.append(collector.collect_api("https://api.example.com/data"))

# output_file = collector.save_collected_data(collected_data)
# print(f"Data saved to :{output_file}")


# #testing the DataPreprocessor

# from dataCollection import DataCollector
# from dataClean_Processing import DataPreprocessor

# collector = DataCollector()
# # collected_data = [
# #     collector.collect_pdf(r"C:\Users\Administrator\Downloads\sravanResume.pdf"),
# #     collector.collect_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
# # ]
# collected_data = [
#     collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
# ]


# preprocessor = DataPreprocessor()
# cleaned_data = preprocessor.batch_preprocess(
#     collected_data,
#     operations = ['clean_text', 'remove_urls', 'remove_emails', 'remove_numbers','lowercase']
# )

# print(f"Processed {len(cleaned_data)} documents")

# print(cleaned_data[0]['content'][:500])



# from documentParsing import DocumentParser
# parser = DocumentParser()

# pdf_data = parser.parse_pdf_pypdf2(r"C:\Users\Administrator\Downloads\sravanResume.pdf")

# html_data = parser.parse_html("<html><head><title>Test</title></head><body><h1>Heading</h1><p>Paragraph with a <a href='https://example.com'>link</a>.</p></body></html>")
# # docx_data = parser.parse_docx(r"C:\Users\Administrator\Downloads\sample.docx")
# # json_data = parser.parse_json(r"C:\Users\Administrator\Downloads\sample.json")
# # csv_data = parser.parse_csv(r"C:\Users\Administrator\Downloads\sample.csv")

# text = parser.extract_text_from_parsed(pdf_data)
# print(text)


#--------------------- chunking and splitting tests ---------------------

# from dataCollection import DataCollector
# from dataClean_Processing import DataPreprocessor
# from TextChunkingSplitting import TextChunker

# collector = DataCollector()
# preprocessor = DataPreprocessor()
# chunker = TextChunker()

# data = collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
# cleaned_data = preprocessor.preprocess(data)

# chunks_char = chunker.chunk_with_metadata(cleaned_data, method='characters', chunk_size = 500, overlap = 100)
# chunks_semantic = chunker.chunk_with_metadata(cleaned_data, method='semantic', max_chunk_size = 1000)

# print(f"Character-based chunks: {len(chunks_char)}")
# print(f"Semantic chunks: {len(chunks_semantic)}")
# print(f"\nFirst character chunk:\n{chunks_char[0]}...")
# print(f"\nFirst semantic chunk:\n{chunks_semantic[0]}...")


#---------------------------- embedding generation tests ----------------#

from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor
from TextChunkingSplitting import TextChunker
from embedding_generator import EmbeddingGenerator

collector = DataCollector()
preprocessor = DataPreprocessor()
chunker = TextChunker()
embedder = EmbeddingGenerator()

data = collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
cleaned_data = preprocessor.preprocess(data)
chunks = chunker.chunk_with_metadata(cleaned_data, method = 'characters', chunk_size = 500, overlap = 100)

chunks = embedder.generate_embeddings(chunks, provider="tfidf", batch_size=16)
# chunks = embedder.generate_embeddings(chunks, provider="openai", batch_size=16)
# chunks = embedder.generate_embeddings(chunks, provider="huggingface", batch_size=16)
# chunks = embedder.generate_embeddings(chunks, provider="cohere", batch_size=16)
# chunks = embedder.generate_embeddings(chunks, provider="bedrock", batch_size=16)

print(f"Generated embeddings for {len(chunks)} chunks")
print(f"Embedding dim : {len(chunks[0]['embedding'])}")