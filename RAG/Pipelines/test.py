# Data collection tests

from dataCollection import DataCollector

collector = DataCollector(output_dir="data/raw")

collected_data = []

collected_data.append(collector.collect_pdf(r"C:\Users\Administrator\Downloads\sravanResume.pdf"))

collected_data.append(collector.collect_url("https://google.github.io/adk-docs/get-started/python/"))

collected_data.append(collector.collect_text_file(r"C:\Users\Administrator\Desktop\agent.txt"))

#collected_data.append(collector.collect_api("https://api.example.com/data"))

output_file = collector.save_collected_data(collected_data)
print(f"Data saved to :{output_file}")


#testing the DataPreprocessor

from dataCollection import DataCollector
from dataClean_Processing import DataPreprocessor

collector = DataCollector()
# collected_data = [
#     collector.collect_pdf(r"C:\Users\Administrator\Downloads\sravanResume.pdf"),
#     collector.collect_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
# ]
collected_data = [
    collector.collect_text_file(r"C:\Users\Administrator\Documents\sravan\Learning\RAG\data\raw\collected_data.txt")
]


preprocessor = DataPreprocessor()
cleaned_data = preprocessor.batch_preprocess(
    collected_data,
    operations = ['clean_text', 'remove_urls', 'remove_emails', 'remove_numbers','lowercase']
)

print(f"Processed {len(cleaned_data)} documents")

print(cleaned_data[0]['content'][:500])

