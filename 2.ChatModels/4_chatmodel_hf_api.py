from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv(override=True)

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    max_new_tokens=100,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)
 
result = model.invoke("What is the capital of India?")

print(result.content)