# Step 1: Setup

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load API key from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OpenAI API key not found. Check your .env file."

print("API key loaded successfully.")


docs_dir = Path(
    "../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs"
)

assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents

documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print(f"\nLoaded {len(documents)} documents.")

print("\nDocument filenames:")

for document in documents:
    print(document.metadata["file_name"])


# Step 3: Build the Index and Query Engine

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)


index = VectorStoreIndex.from_documents(documents)


query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")

# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    source_node = response.source_nodes[0]

    print("\nTop retrieved source:")
    print(f"Document: {source_node.node.metadata['file_name']}")
    print(f"Similarity score: {source_node.score}")
    print(f"Chunk text: {source_node.node.get_content()[:200]}")


# The assistant provided accurate answers to all five questions.
# It correctly answered questions about weekend hours, dairy-free milk options, the loyalty program, company history, and catering
# and wholesale services.
#
# However, the top retrieved source was not always the most relevant.
# For example, the weekend hours question retrieved our_story.txt instead of faq.txt, and the dairy-free milk question retrieved
# seasonal_specials.txt instead of menu.txt.
#
# The company history and catering questions had the most relevant top sources.
# I was surprised that the assistant could still provide correct answers even when the top retrieved document was not the
# best match. This suggests that the other retrieved chunks may have provided the necessary information.


# Step 5: Find a Failure

question = "What is the most common password?"

print("\n" + "=" * 60)
print("STEP 5: FIND A FAILURE")
print(f"Question: {question}")

response = query_engine.query(question)

print("\nFull answer:")
print(response)

print("\nAll three retrieved source nodes:")

for i, source_node in enumerate(response.source_nodes, 1):
    print(f"\nSource {i}")
    print(f"Document: {source_node.node.metadata['file_name']}")
    print(f"Similarity score: {source_node.score}")
    print(f"Chunk text: {source_node.node.get_content()[:200]}")

# I asked "What is the most common password?" because the question is ambiguous and the answer is not available in the Groundwork
# documents.
#
# The model interpreted the question as asking about the coffee shop's Wi-Fi password.
# It retrieved faq.txt, which mentions that customers can ask a barista for the current Wi-Fi password.
# However, the actual password is not provided in the documents.
#
# The similarity scores were low, especially for the first source (0.158).
# Despite the weak retrieval results, the model provided a confident answer instead of explicitly saying that the
# information was unavailable.


# Step 6: Reflection

# 1. LlamaIndex makes building a RAG system much easier because it automates chunking, embedding, and indexing.
# My implementation required relatively little code to load documents, build the index, and create a query engine.
# In comparison, the keyword-based RAG implementation required more manual code for tokenization, keyword matching, scoring,
# and selecting documents. Using a framework saves time and makes it easier to build more advanced systems.
# 

# 2. One possible use case is an internal HR assistant.
# Employees could ask questions about company policies, benefits, vacation rules, and onboarding procedures.
# The assistant could retrieve information from internal company documents instead of requiring employees to search through multiple files.

# 3. One failure that RAG cannot completely prevent is hallucination.
# Even when retrieval works correctly, the model can misunderstand the retrieved information, make assumptions, or generate
# unsupported details. RAG can reduce hallucinations, but it cannot guarantee that every answer will be correct.
