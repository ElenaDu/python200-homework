
# --- Completions API ---
# API Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Send a prompt to the model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)

print("Model response:", response.choices[0].message.content)
print("Model used:", response.model)
print("Total tokens:", response.usage.total_tokens)
