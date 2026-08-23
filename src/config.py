import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Safety check
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Create the Gemini client (new SDK style)
client = genai.Client(api_key=api_key)

# Send one test message
response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="In one sentence, what is retrieval-augmented generation?",
)

print(response.text)