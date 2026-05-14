import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import math
import re
import joblib


# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------
def clean_text(text):

    text = str(text)

    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------
# CHECK NaN
# ---------------------------------------------------
def has_nan(embedding):

    for value in embedding:

        if math.isnan(value):
            return True

    return False


# ---------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------
def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    data = r.json()

    # Error handling
    if "embeddings" not in data:

        print("\nERROR FROM OLLAMA:")
        print(data)

        return []

    return data["embeddings"]


# ---------------------------------------------------
# LLM INFERENCE
# ---------------------------------------------------
def inference(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    data = r.json()

    return data


# ---------------------------------------------------
# LOAD EMBEDDINGS DATAFRAME
# ---------------------------------------------------
df = joblib.load("embeddings.joblib")

print("\nEmbeddings Loaded Successfully")

# ---------------------------------------------------
# CLEAN EXISTING DATAFRAME
# ---------------------------------------------------
df = df[df["embedding"].notnull()]

valid_rows = []

for index, row in df.iterrows():

    embedding = row["embedding"]

    # Skip invalid embeddings
    if embedding is None:
        continue

    if has_nan(embedding):
        print(f"Skipped NaN embedding at index {index}")
        continue

    valid_rows.append(row)

df = pd.DataFrame(valid_rows)

print(f"\nValid Rows: {len(df)}")


# ---------------------------------------------------
# USER QUERY
# ---------------------------------------------------
incoming_query = input("\nAsk a Question: ")

incoming_query = clean_text(incoming_query)

# ---------------------------------------------------
# QUESTION EMBEDDING
# ---------------------------------------------------
question_embedding_list = create_embedding([incoming_query])

if len(question_embedding_list) == 0:

    print("Failed to create question embedding")
    exit()

question_embedding = question_embedding_list[0]

# Check query embedding
if has_nan(question_embedding):

    print("Question embedding contains NaN")
    exit()

# ---------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------
embeddings_matrix = np.vstack(df["embedding"].values)

similarities = cosine_similarity(
    embeddings_matrix,
    [question_embedding]
).flatten()

# ---------------------------------------------------
# TOP RESULTS
# ---------------------------------------------------
top_results = 5

max_indx = similarities.argsort()[::-1][:top_results]

new_df = df.iloc[max_indx]

print("\nTop Matching Chunks:\n")

print(
    new_df[
        ["title", "number", "start", "end", "text"]
    ]
)

# ---------------------------------------------------
# PROMPT
# ---------------------------------------------------
prompt = f"""
I am teaching Python programming in my course.

Here are relevant subtitle chunks from my Python course videos:

{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}

------------------------------------------------------

User Question:
"{incoming_query}"

Instructions:
- Answer in a human friendly way.
- Tell the user which video contains the answer.
- Mention timestamps whenever possible.
- Guide the user properly.
- Explain concepts clearly if needed.
- Do NOT mention JSON, embeddings, chunks, dataframe or vector search.
- If the question is unrelated to Python or the course, politely say you can only answer course related questions.
"""

# ---------------------------------------------------
# SAVE PROMPT
# ---------------------------------------------------
with open("prompt.txt", "w", encoding="utf-8") as f:

    f.write(prompt)

# ---------------------------------------------------
# GENERATE RESPONSE
# ---------------------------------------------------
response_data = inference(prompt)

response = response_data.get("response", "No response generated")

# ---------------------------------------------------
# PRINT RESPONSE
# ---------------------------------------------------
print("\nAI RESPONSE:\n")

print(response)

# ---------------------------------------------------
# SAVE RESPONSE
# ---------------------------------------------------
with open("response.txt", "w", encoding="utf-8") as f:

    f.write(response)

print("\nResponse saved to response.txt")