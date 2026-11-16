
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


# --- System Messages and Personas ---

# System Q1

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Patient Python tutor:")
print(response.choices[0].message.content)

messages = [
    {
        "role": "system",
        "content": "You are a dramatic pirate who teaches Python. Explain programming concepts using pirate language, funny comparisons, and occasional pirate expressions. Keep explanations accurate and easy to understand."
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nPirate Python instructor:")
print(response.choices[0].message.content)

# Observation:
# Changing the system message significantly changed the model's personality and communication style.
# The pirate instructor used humor, pirate expressions, and creative analogies to explain the same concept.
# However, although the technical explanation and Python examples remained accurate, the response was not as easy to follow.


# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Assistant response:", response.choices[0].message.content)

# Observation:
# The model knows Jordan's name because the conversation history was included in the messages list.
# Although the API is stateless and does not remember previous calls, it can use the context
# provided with each new request.

