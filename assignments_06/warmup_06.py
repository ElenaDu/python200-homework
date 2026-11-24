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


# Concepts Q2

# A confidently wrong answer is more harmful than an answer that admits uncertainty because people may trust the information and act on it
# without verifying whether it is correct.
#
# For example, someone asks an AI assistant when they must stop for a school bus displaying flashing red lights on the opposite side of
# the road.
# The answer may depend on the road configuration and the state's definition of a divided highway.
# Requirements for physical dividers and exceptions for opposing  traffic vary by state.
# If the AI confidently provides an incorrect answer, a driver might illegally pass a stopped school bus,
# potentially putting children crossing the road at risk.

# The tone of a response affects how much people trust the model.
# Confident and authoritative language can make incorrect information sound reliable, encouraging people to act on it without checking.
# Admitting uncertainty signals that the information needs verification.



# Concepts Q3

# 1. Extract text from source documents
# Read source documents and extract their text for processing.

# 2. Split text into chunks
# Divide long documents into smaller pieces to make retrieval more precise.

# 3. Convert text chunks into embeddings
# Transform each text chunk into a numerical vector representing its meaning.

# 4. Receive the user's query
# Accept the question submitted by the user.

# 5. Embed the user's query
# Convert the user's question into a numerical vector using the same embedding model.

# 6. Retrieve the most relevant chunks
# Compare the query embedding with document embeddings and select the most relevant chunks.

# 7. Inject retrieved chunks into the prompt
# Add the retrieved text to the prompt to provide context for the LLM.

# 8. Generate a response from the LLM
# Use the user's question and retrieved context to generate an answer.


# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Question 1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# loyalty.txt was selected because it had the same keyword overlap score as hours.txt and hiring.txt.
#  Since the results are sorted in reverse alphabetical order, loyalty.txt was selected first.


# Question 2
query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# No relevant document was found because none of the query keywords matched the document contents. 
# Keyword RAG could not identify the menu as potentially relevant.
#  Semantic retrieval would do better because it understands the meaning of words and related concepts.


# Question 3

query = "How do I sign up for rewards?"

# Prediction: I think loyalty.txt will be selected because the question
# is about signing up for rewards, which relates to the loyalty program.

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:", results[0][0])

# My prediction was incorrect. The function did not find any matching keywords because the query and loyalty.txt use different words.
# Keyword retrieval only matches exact words, not their meanings.


# --- Semantic ---

# Question 1
#
# 1. A vector embedding is a numerical representation of text that captures its meaning. 
#    It allows computers to compare texts based on their meaning rather than just matching individual words.
#   
# 2. The chunk with a cosine similarity score of 0.85 is more relevant because it is closer in meaning to the query.
#    A higher score means the texts have a stronger semantic relationship.
#    
# 3. Semantic search can find relevant information even when the exact words are different because it looks at the meaning and context
#    of the text, rather than relying on exact keyword matches.


# Question 2
#
# | Feature                 | Keyword RAG                    | Semantic RAG                     |
# |-------------------------|--------------------------------|----------------------------------|
# | What is compared?       | Exact word overlap             | Meaning of text using embeddings |
# | What is retrieved?      | Full document                  | Most relevant text chunks        |
# | Can it handle synonyms? | No                             | Yes                              |
# | Storage format          | Plain text dictionary          | Vector database                  |
# | Relevance score         | Number of overlapping keywords | Cosine similarity score          |



# --- LlamaIndex ---


from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

# Load Brightleaf Solar PDF documents
documents = SimpleDirectoryReader(
    "../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
).load_data()

print(f"\nLoaded {len(documents)} PDF documents")



# Question 1

# Build an in-memory vector index
index = VectorStoreIndex.from_documents(documents)

# Create a query engine that retrieves the 3 most relevant chunks
query_engine = index.as_query_engine(similarity_top_k=3)


questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


for question in questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nRetrieved source nodes:")

    for i, source_node in enumerate(response.source_nodes, 1):
        print(f"\nSource {i}")
        print(f"Similarity score: {source_node.score}")
        print(f"Chunk text: {source_node.node.get_content()[:150]}")




# Query 1: Employee benefits
# The first retrieved chunk was highly relevant because it contained information about employee benefits. 
# The other two chunks were less relevant because they discussed the company's partnership and general overview.
# The model gave a confident and detailed answer without hedging. 
# It was unexpected that two of the three retrieved chunks were not directly related to employee benefits.

# Query 2: Security policies
# The first retrieved chunk was highly relevant because it contained  information about network and data security.
# The other two chunks were less relevant because they discussed employee benefits and the company's general overview.
# The model gave a confident and specific answer without hedging.
# It was unexpected that the employee benefits chunk was retrieved for a question about security policies.


# Question 2

question = "What employee benefits does BrightLeaf offer?"

# Run with similarity_top_k=1
query_engine_1 = index.as_query_engine(similarity_top_k=1)
response_1 = query_engine_1.query(question)

print("\nAnswer with similarity_top_k=1:")
print(response_1)

for node in response_1.source_nodes:
    print(f"Score: {node.score}")


# Run with similarity_top_k=5
query_engine_5 = index.as_query_engine(similarity_top_k=5)
response_5 = query_engine_5.query(question)

print("\nAnswer with similarity_top_k=5:")
print(response_5)

for node in response_5.source_nodes:
    print(f"Score: {node.score}")



# With similarity_top_k=1, the model retrieved only the most relevant chunk and provided a detailed answer about employee benefits.
# With similarity_top_k=5, the answer contained mostly the same information, but was presented in a slightly different format.
# The additional retrieved chunks had lower similarity scores and did not significantly improve the answer.
# This shows that retrieving more context is not always better because it can introduce irrelevant information and potentially distract the model.


# Question 3

question = "What is it like to work at BrightLeaf?"

# Run the query
response = query_engine.query(question)

print("\n" + "=" * 60)
print("LLAMAINDEX QUESTION 3")
print(f"Question: {question}")

print("\nAnswer:")
print(response)

print("\nAll retrieved source nodes:")

for i, source_node in enumerate(response.source_nodes, 1):
    print(f"\nSource {i}")
    print(f"Similarity score: {source_node.score}")
    print(f"Chunk text:\n{source_node.node.get_content()}")


# Observations:
# I expected this query to be challenging because it was vague and could involve information from multiple documents.
# The model provided a detailed answer about employee benefits, work flexibility, and professional development.
# The first retrieved chunk was highly elevant, and the second provided useful information about the company's mission. 
# The third chunk was less relevant because it focused on business partnerships rather than employee experience.
# To improve the system, I would use a reranker to select more relevant chunks and experiment with different chunking strategies.


# Question 4

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# Configure the LLM that will act as the evaluator (judge)
judge_llm = OpenAI(model="gpt-4o-mini", temperature=0)

# Instantiate both evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


# Questions to evaluate
evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is the capital of France?",
]

for q in evaluation_questions:
    print("\n" + "=" * 60)
    print(f"Question: {q}")

    response = query_engine.query(q)

    print("\nAnswer:")
    print(response)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=q,
        response=response,
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=q,
        response=response,
    )

    print("\nEvaluation scores:")
    print(f"Faithfulness score: {faithfulness_result.score}")
    print(f"Relevancy score: {relevancy_result.score}")


# The employee benefits question received a faithfulness score of 1.0 and a relevancy score of 1.0. 
# The answer was fully supported by the retrieved BrightLeaf documents and directly addressed the question.
# 
# The question about the capital of France received scores of 0.0 for both faithfulness and relevancy. 
# Although the answer was factually correct, the information was not present in the retrieved BrightLeaf documents. 
# This demonstrates that a response can be factually correct but still fail the faithfulness evaluation.

# LLM-as-a-judge uses a language model to evaluate the quality of generated responses.
# It is useful for RAG evaluation because open-ended answers can be phrased in many different ways, making
# simple accuracy metrics difficult to apply. An LLM judge can evaluate whether an answer is supported by the retrieved context
# and whether it addresses the user's question.

