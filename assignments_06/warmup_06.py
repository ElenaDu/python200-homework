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


# Keyword Question 1

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
