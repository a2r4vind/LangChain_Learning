from langchain_openai import OpenAI
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# create an instance of the OpenAI LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of Australia?")

print(result)