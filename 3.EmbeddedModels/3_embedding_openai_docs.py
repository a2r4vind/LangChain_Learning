from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "New Delhi is the capital of India",
    "Mumbai is the financial capital of India",
    "Bangalore is the IT hub of India"
]
result = embedding.embed_documents(documents) # for multiple queries in documents use embed_documents

print(str(result))