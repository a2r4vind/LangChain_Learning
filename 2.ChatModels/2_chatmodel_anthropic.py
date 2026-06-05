from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# create an instance of the ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

result = model.invoke("What is the capital of India?")

print(result.content)