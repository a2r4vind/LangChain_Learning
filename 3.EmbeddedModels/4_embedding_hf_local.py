from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "New Delhi is the capital of India"

documents = [
    "New Delhi is the capital of India",
    "Mumbai is the financial capital of India",
    "Bangalore is the IT hub of India"
]

# vector = embedding.embed_query(text)
vector = embedding.embed_documents(documents)

print(str(vector))