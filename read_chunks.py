import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# -----------------------------
# CREATE EMBEDDINGS FUNCTION
# -----------------------------
def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    data = r.json()

    # Debugging response
    if "embeddings" not in data:
        print("\nERROR FROM OLLAMA:")
        print(data)
        return []

    return data["embeddings"]


# -----------------------------
# BATCH FUNCTION
# -----------------------------
def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# -----------------------------
# MAIN CODE
# -----------------------------
jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0

for json_file in jsons:

    print(f"\nProcessing File: {json_file}")

    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content.get("chunks", [])

    # -----------------------------
    # CLEAN TEXTS
    # -----------------------------
    valid_chunks = []

    for chunk in chunks:

        text = chunk.get("text")

        # Skip invalid data
        if text is None:
            print("Skipped None text")
            continue

        if not isinstance(text, str):
            print("Skipped non-string text")
            continue

        text = text.strip()

        if text == "":
            print("Skipped empty text")
            continue

        chunk["text"] = text
        valid_chunks.append(chunk)

    print(f"Valid Chunks: {len(valid_chunks)}")

    # -----------------------------
    # CREATE EMBEDDINGS IN BATCHES
    # -----------------------------
    batch_size = 10

    for small_batch in batch(valid_chunks, batch_size):

        texts = [c["text"] for c in small_batch]

        embeddings = create_embedding(texts)

        # If embedding failed
        if len(embeddings) == 0:
            print("Embedding generation failed")
            continue

        # -----------------------------
        # STORE RESULTS
        # -----------------------------
        for i, chunk in enumerate(small_batch):

            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]

            my_dicts.append(chunk)

            chunk_id += 1
            

            print(f"Stored Chunk ID: {chunk['chunk_id']}")
    break

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame.from_records(my_dicts)

print("\nFINAL DATAFRAME:")
print(df)
incoming_query = input("Ask a Question :")
question_embedding = create_embedding([incoming_query])
print(question_embedding)
# -----------------------------
# OPTIONAL: SAVE TO CSV
# -----------------------------
#df.to_csv("embeddings_output.csv", index=False)

#print("\nSaved embeddings_output.csv")
joblib.dump(df, "embeddings.joblib")

print("embeddings.joblib saved successfully")