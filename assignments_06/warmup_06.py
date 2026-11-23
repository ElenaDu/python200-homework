from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")



# --- RAG Concepts ---

# Concepts Q1

# Scenario A: RAG
# RAG is the best approach because the legal team needs to search hundreds of internal PDFs that are updated quarterly.
# The system can retrieve relevant information from the latest documents without retraining the model.

# Scenario B: Fine-tuning
# Fine-tuning is the best approach because the startup has 3,000 examples of its desired writing style.
# These examples can be used to train the model to consistently reproduce its distinctive brand voice.


# Scenario C: Prompt engineering
# Prompt engineering is sufficient because the analyst only needs to work with one short report.
# She can include the report directly in the prompt along with her question, without building a separate retrieval system.


