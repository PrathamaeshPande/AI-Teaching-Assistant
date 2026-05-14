import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    
    # Print full error response from Ollama for debugging
    if r.status_code != 200:
        print(f"Ollama Error {r.status_code}: {r.text}")
        r.raise_for_status()
    
    data = r.json()
    return data.get("embeddings") or data.get("embedding")


jsons = [f for f in os.listdir("jsons") if f.endswith('.json')]
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    
    print(f"Creating Embeddings for {json_file}")
    
    chunks = content['chunks']
    
    # Process one chunk at a time to avoid overloading Ollama
    for i, chunk in enumerate(chunks):
        embeddings = create_embedding([chunk['text']])  # single chunk per request
        
        my_dicts.append({
            "chunk_id":  chunk_id,
            "number":    content.get("number"),
            "title":     content.get("title"),
            "text":      chunk["text"],
            "embedding": embeddings[0]
        })
        chunk_id += 1
        print(f"  chunk {i+1}/{len(chunks)} done")

df = pd.DataFrame(my_dicts, columns=["chunk_id", "number", "title", "text", "embedding"])
print(df)