
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

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    print(f"\nTemperature: {temperature}")
    print(response.choices[0].message.content)

# Observation:
# I noticed that the responses were quite different in both length and format. 
# With temperature set to 0, the model provided a list of several suggestions, while at 0.7 and 1.5 it returned just
# one name each. Higher temperatures allow for more variation and creativity, while lower temperatures generally make the output
# more predictable and focused.
# If I needed a consistent and reproducible output, I would choose temperature=0 because it favors more predictable responses and
# reduces randomness in the model's answers.


# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)
for choice in response.choices:
    print(choice.message.content)



# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print("Response:", response.choices[0].message.content)

# Observation:
# The response was cut off because max_tokens was set to 15, which limited the length of the generated text.
# In a real application, max_tokens is useful for controlling response length, managing API costs, and preventing the model
# from generating unnecessarily long answers.

