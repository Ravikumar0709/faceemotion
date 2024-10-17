 import pandas as pd
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Initialize OpenAI LLM and embeddings
llm = OpenAI(model="gemini-1.5-flash")
embeddings = OpenAIEmbeddings(model="gemini")

# Define areas of interest
areas = ['products', 'offers', 'benefits', 'plans', 'support']

# Define prompt templates and instructions
prompt_templates = {
    'products': "Generate a FAQ question and answer about products in direct-broadcast digital television.",
    'offers': "Generate a FAQ question and answer about offers in direct-broadcast digital television.",
    'benefits': "Generate a FAQ question and answer about benefits in direct-broadcast digital television.",
    'plans': "Generate a FAQ question and answer about plans in direct-broadcast digital television.",
    'support': "Generate a FAQ question and answer about support in direct-broadcast digital television."
}

# Generate FAQs
def generate_faqs(area, count=6):
    faqs = []
    for _ in range(count):
        try:
            prompt = prompt_templates[area]
            response = llm(prompt)
            question, answer = response.split('\n', 1)  # Assuming first line is question and second is answer
            faqs.append({'Area': area, 'Question': question, 'Answer': answer.strip()})
        except Exception as e:
            print(f"Error generating FAQ for {area}: {e}")
    return faqs

# Collect all FAQs
all_faqs = []
for area in areas:
    all_faqs.extend(generate_faqs(area))

# Create a DataFrame and save as CSV
faq_df = pd.DataFrame(all_faqs)
faq_df.to_csv('faqs.csv', index=False)

print("FAQs generated and saved to faqs.csv.")
