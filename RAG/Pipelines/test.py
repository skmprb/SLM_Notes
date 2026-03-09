## Data collection tests

from dataCollection import DataCollector

collector = DataCollector(output_dir="data/raw")

collected_data = []

collected_data.append(collector.collect_pdf(r"C:\Users\Administrator\Downloads\sravanResume.pdf"))

collected_data.append(collector.collect_url("https://google.github.io/adk-docs/get-started/python/"))

collected_data.append(collector.collect_text_file(r"C:\Users\Administrator\Desktop\agent.txt"))

#collected_data.append(collector.collect_api("https://api.example.com/data"))

output_file = collector.save_collected_data(collected_data)
print(f"Data saved to :{output_file}")
