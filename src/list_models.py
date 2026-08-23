import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Models available to your account:\n")
for m in client.models.list():
    # Only show models that can generate content
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"  {m.name}")