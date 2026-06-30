# AI Teaching Assistant using RAG and Local LLMs

A local Retrieval-Augmented Generation (RAG) based AI assistant that answers course-related questions using semantic search and local LLM inference. The system retrieves relevant video transcript chunks and generates contextual responses with timestamps and video references.

---

# Features

* Local/offline AI inference using Ollama
* Semantic search using vector embeddings
* Context-aware question answering
* Transcript chunking and preprocessing
* Timestamp-based video guidance
* Cosine similarity based retrieval
* Automatic subtitle/transcript processing
* End-to-end RAG pipeline implementation

---

# Project Workflow

1. Convert course videos into audio files
2. Generate transcripts using Whisper speech-to-text
3. Split transcripts into smaller chunks
4. Generate embeddings using BGE-M3
5. Store embeddings and metadata locally
6. Convert user query into embeddings
7. Retrieve relevant chunks using cosine similarity
8. Send retrieved context to Llama 3.2
9. Generate grounded AI response with timestamps

---

# Tech Stack

Python, Ollama, Llama 3.2, BGE-M3, Whisper, Retrieval-Augmented Generation (RAG), NLP, Semantic Search, Vector Embeddings, Cosine Similarity, Pandas, NumPy, Scikit-learn, Joblib, JSON, Prompt Engineering, Local LLMs

---

# How To Run

## Step 1 — Collect Course Videos

Create a folder named `videos` and place all course videos inside it.


## Step 2 — Extract Audio From Videos

Run:
python video_process.py

This script extracts audio from all videos and stores them inside the `audios/` folder.


## Step 3 — Generate Transcripts

Run:
python stt.py

This script uses Whisper speech-to-text to generate transcripts and stores them as JSON files.


## Step 4 — Create Chunks and Embeddings

Run:
python create_chunks.py


This script:

* Cleans transcript text
* Splits transcripts into chunks
* Generates embeddings using BGE-M3
* Stores embeddings and metadata locally

Output files:

* `embeddings.joblib`
* `embeddings_output.csv`

---

## Step 5 — Install and Start Ollama

Install Ollama:

https://ollama.com

Pull required models:


ollama pull llama3.2
ollama pull bge-m3

Start Ollama:


ollama serve


## Step 6 — Start The AI Teaching Assistant

Run:
python process_in.py

The assistant will ask:

Ask a Question:


Example questions:

* What is inheritance in Python?
* Which video explains APIs?
* Explain list comprehension
* Where are decorators discussed?



# Future Improvements

* Streamlit web interface
* FAISS / ChromaDB vector database integration
* Multi-course support
* Chat history and memory
* Docker deployment
* Real-time streaming responses
* User authentication system
* Hybrid search implementation


# Author
Prathamesh Pande
