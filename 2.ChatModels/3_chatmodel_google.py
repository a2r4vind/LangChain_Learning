from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# create an instance of the ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = model.invoke("What is the capital of India?")

print(result.content)