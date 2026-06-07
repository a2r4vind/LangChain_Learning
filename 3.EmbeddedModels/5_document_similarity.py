from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv(override=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_genai_api_key=os.getenv("GOOGLE_API_KEY")
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style and leadership skills.",
    "Sachin Tendulkar is a former Indian cricketer widely regarded as one of the greatest batsmen in the history of cricket.",
    "M.S. Dhoni is a former Indian cricketer and captain of the Indian national team, known for his calm demeanor and finishing abilities in limited-overs cricket.",
    "Rohit Sharma is an Indian cricketer known for his elegant batting style and ability to score big centuries in limited-overs cricket.",
    "Vaibhav Sooryavanshi is an Indian cricketer who has shown promise as a batsman and has represented India in domestic cricket.",
    "Jasprit Bumrah is an Indian cricketer known for his unique bowling action and ability to bowl yorkers at the death in limited-overs cricket."
]

# query = "Tell me about Virat Kohli"
query = "Tell me about Sooryavanshi"

document_embeddings = embeddings.embed_documents(documents, output_dimensionality=300)
query_embedding = embeddings.embed_query(query, output_dimensionality=300)

scores = cosine_similarity([query_embedding], document_embeddings)[0] # always pass a 2D array to cosine_similarity function

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0] # sort the scores in descending order so largest score will be at the top

print(f"Query: {query}")
print(f"Most Similar Document: {documents[index]}")
print(f"Similarity Score: {score}")