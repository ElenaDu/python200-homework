
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



# --- Prompt Engineering ---

# Prompt Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Reviews:
{reviews}

Return the sentiment for each review, labeled by review number.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("Zero-Shot Sentiment Classification:")
print(response.choices[0].message.content)

# Prompt Q2

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review below as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:
{reviews}

Return each classification labeled with the review number.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("One-Shot Sentiment Classification:")
print(response.choices[0].message.content)

# Observation:
# Adding one example changed the output format. In Q1, the model returned a simple numbered list, while in Q2, it included each
# review followed by its sentiment label.
# Both approaches produced correct classifications, but the example helped guide the output into a more structured and consistent format.


# Prompt Q3

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Examples:

Review: "The product is easy to use and works perfectly."
Sentiment: positive

Review: "The app is slow, unreliable, and full of bugs."
Sentiment: negative

Review: "The customer service was excellent, but the product was disappointing."
Sentiment: mixed

Now classify these reviews:

{reviews}

Return each classification labeled with the review number.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("Few-Shot Sentiment Classification:")
print(response.choices[0].message.content)

# Comparison:
# Zero-shot prompting is useful for simple tasks where instructions alone are enough. 
# One-shot prompting is helpful when the model needs an example of the expected format or style.
# Few-shot prompting is useful for more complex tasks where multiple examples can help the model recognize patterns and produce more consistent results.


# Prompt Q4

prompt = """
Solve the following problem. Show your reasoning step by step
before giving your final answer. Clearly label the final answer.

A data engineer earns $85,000 per year. She gets a 12% raise,
then 6 months later takes a new job that pays $7,500 more per
year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("Chain of Thought Response:")
print(response.choices[0].message.content)

# Observation:
# Asking the model to solve a problem step by step helps it break a complex task into smaller, manageable steps.
# This makes the calculation process easier to follow and verify, and can help reduce errors in multi-step problems.


# Prompt Q5

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the following review:

{review}

Return only valid JSON with these three keys:
- "sentiment": positive, negative, or mixed
- "confidence": a number between 0 and 1
- "reason": one sentence explaining the sentiment

Do not include Markdown formatting or any text outside the JSON.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

raw_response = response.choices[0].message.content

print("Raw JSON response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nParsed Results:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nInvalid JSON. Raw response:")
    print(raw_response)


# Prompt Q6

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("Instruction Text Response:")
print(response.choices[0].message.content)


# Second prompt: regular prose, not instructions

user_text = "The sun was setting behind the mountains. The sky turned orange and pink, and birds were returning to their nests."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nRegular Prose Response:")
print(response.choices[0].message.content)


# Observation:
# Delimiters help prevent confusion between instructions and input data.
# They clearly mark the beginning and end of the text the model should process,
# reducing the risk of treating instructions inside the input as actual prompts.


# --- Local Models with Ollama ---
# Ollama Q1 

# OpenAI API response:
prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("OpenAI API response:")
print(response.choices[0].message.content)

# Ollama response:
# A large language model is an AI system trained on vast amounts of text to understand and generate human-like language, 
# enabling tasks like writing or answering questions with accuracy and fluency.
# It processes and interprets complex patterns in the training data to perform tasks like comprehension, 
# creativity, and natural language generation.  
#
# OpenAI API response:
#
# A large language model is a type of artificial intelligence that uses deep
# learning techniques to understand and generate human language by processing
# vast amounts of text data. These models are designed to predict the next word
# in a sequence, enabling them to perform a variety of language tasks such as
# translation, summarization, and conversation.
#
# Comparison:
# Both models explained how large language models understand and generate human-like text and described some of their applications.
# The Ollama response focused on understanding patterns in training data and generating language,
# while the OpenAI response provided more detail about deep learning and next-word prediction.
# 
# Advantage of a local model:
# It can run on my own computer without sending prompts to an external API or paying per-token API fees.
# 
# Disadvantage of a local model:
# Its performance and response quality may be more limited by the computer's hardware and the size of the model.
