from typing import Dict

class LLMGenerator:
    
    def generate_openai(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate using OpenAI"""
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def generate_bedrock(self, prompt: str, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0") -> str:
        """Generate using AWS Bedrock"""
        import boto3, json
        client = boto3.client('bedrock-runtime')
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            })
        )
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    
    def generate_huggingface(self, prompt: str, model_name: str = "google/flan-t5-base") -> str:
        """Generate using HuggingFace"""
        from transformers import pipeline
        generator = pipeline("text2text-generation", model=model_name)
        return generator(prompt, max_length=512)[0]['generated_text']
    
    def generate_cohere(self, prompt: str, model: str = "command-r-plus") -> str:
        """Generate using Cohere"""
        import cohere
        client = cohere.Client()
        response = client.chat(message=prompt, model=model)
        return response.text
    
    def generate_ollama(self, prompt: str, model: str = "llama3") -> str:
        """Generate using Ollama (local)"""
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json()['response']
    
    def generate(self, prompt: str, provider: str = "openai", **kwargs) -> str:
        """Unified generation"""
        fn = {
            'openai': self.generate_openai,
            'bedrock': self.generate_bedrock,
            'huggingface': self.generate_huggingface,
            'cohere': self.generate_cohere,
            'ollama': self.generate_ollama
        }[provider]
        return fn(prompt, **kwargs)


#usage example
# from prompt_builder import PromptBuilder
# from llm_generator import LLMGenerator

# builder = PromptBuilder()
# llm = LLMGenerator()

# # Build prompt with retrieved chunks
# prompt = builder.build_prompt(query, results, style="sources")

# # Generate answer (pick any provider)
# answer = llm.generate(prompt, provider="openai")
# # answer = llm.generate(prompt, provider="bedrock")
# # answer = llm.generate(prompt, provider="ollama")
# # answer = llm.generate(prompt, provider="huggingface")

# print(answer)
