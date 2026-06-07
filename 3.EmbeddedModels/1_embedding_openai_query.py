from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("New Delhi is the capital of India") # for single query use embed_query

print(str(result))