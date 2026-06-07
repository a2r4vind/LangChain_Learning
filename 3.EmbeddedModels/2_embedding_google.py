from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_genai_api_key=os.getenv("GOOGLE_API_KEY")
)

text = "New Delhi is the capital of India"

vector = embeddings.embed_query(text, output_dimensionality=32) # default dimensions is 3072

print(f"Generated Vector Dimensions: {len(vector)}")
print(f"Generated Vector: {str(vector)}")