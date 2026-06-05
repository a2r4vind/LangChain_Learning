from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# create an instance of the ChatOpenAI
model = ChatOpenAI(model="gpt-4")

result = model.invoke("What is the capital of India?")

print(result.content)
print(result)