import os
from dotenv import load_dotenv
from huggingface_hub import login

# Force python to refresh and read the updated .env file 
load_dotenv(override=True)

# Fetch the variable
token = os.getenv("HF_TOKEN")

if token:
    # Explicitly authenticate the runtime container using the token string
    login(token=token)
    print("Successfully authenticated via Python code using the new token!")
else:
    print("Error: HF_TOKEN environment variable could not be read.")
