# AI Learning Mentor

AI Learning Mentor is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its contents.

The system retrieves relevant information from the uploaded document using semantic and keyword search, reranks the retrieved passages, and generates grounded answers with page citations.

## Features

- PDF upload and text extraction
- Token-based document chunking
- OpenAI embeddings
- FAISS vector search
- BM25 keyword search
- Hybrid retrieval
- LLM-based reranking
- Retrieval-Augmented Generation
- Source page citations
- Conversational follow-up questions
- Query rewriting
- Persistent chat history using SQLite
- FastAPI REST API
- Browser-based frontend
- Application logging and latency monitoring
- Docker containerization
- Persistent Docker volume storage

## Architecture

```text
                     User
                       |
                       v
                 Web Frontend
                       |
                       v
                 FastAPI API
                       |
          +------------+------------+
          |                         |
          v                         v
      SQLite                    RAG Pipeline
   Chat History                     |
                                    v
                             Query Rewriting
                                    |
                                    v
                           Hybrid Retrieval
                         /                  \
                        v                    v
                FAISS Vector Search     BM25 Search
                         \                  /
                          \                /
                           v              v
                           Candidate Merge
                                  |
                                  v
                              Reranker
                                  |
                                  v
                         Relevant PDF Chunks
                                  |
                                  v
                           OpenAI Generation
                                  |
                                  v
                       Answer + Source Pages
```

## Project Structure

```text
ai-learning-mentor/
│
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── database.py
│   ├── document_service.py
│   ├── embeddings.py
│   ├── models.py
│   ├── observability.py
│   └── rag_service.py
│
├── frontend/
│   └── index.html
│
├── data/
│   ├── learning_mentor.db
│   ├── pdf_chunks.json
│   ├── pdf_faiss.index
│   └── logs/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .env
└── README.md
```

The `data/` directory and `.env` file are excluded from Git.

## RAG Pipeline

### 1. PDF Processing

The uploaded PDF is processed using PyMuPDF.

Text is extracted page-by-page so source-page information can be retained.

### 2. Chunking

Document text is split into overlapping token-based chunks using `tiktoken`.

Overlapping chunks help preserve context across chunk boundaries.

### 3. Embeddings

Each chunk is converted into a numerical embedding using OpenAI's embedding model.

### 4. FAISS Vector Search

Embeddings are normalized and stored in a FAISS index.

FAISS retrieves chunks that are semantically similar to the user's question.

### 5. BM25 Search

BM25 performs traditional keyword-based retrieval.

This complements semantic search by improving retrieval of exact terms and technical keywords.

### 6. Hybrid Retrieval

FAISS and BM25 scores are normalized and combined.

```text
Hybrid Score =
0.7 × Vector Score
+
0.3 × BM25 Score
```

The weights are configurable.

### 7. Reranking

The highest-ranked retrieval candidates are sent to an LLM reranker.

Each chunk receives a relevance score between 0 and 10.

Only the strongest chunks are selected for final generation.

### 8. Answer Generation

The selected chunks are sent to the language model as context.

The model is instructed to answer only using information from the uploaded PDF.

The API returns:

- generated answer
- source pages
- rewritten search query

## Conversational RAG

The application stores recent conversation history.

For follow-up questions such as:

```text
Why is it important?
```

the system rewrites the question into a standalone search query using previous messages before performing retrieval.

## Observability

The application logs important RAG operations including:

- HTTP request latency
- query rewriting
- hybrid retrieval
- reranking latency
- retrieved pages
- generation latency
- total RAG latency
- errors

Logs are stored under:

```text
data/logs/app.log
```

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-learning-mentor
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env`

Create:

```text
.env
```

and add:

```text
OPENAI_API_KEY=your-openai-api-key
```

Never commit this file.

### 5. Start FastAPI

```bash
uvicorn backend.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the Docker image:

```bash
docker build -t ai-learning-mentor .
```

Create persistent storage:

```bash
docker volume create learning-mentor-data
```

Run:

```bash
docker run --env-file .env -p 8000:8000 -v learning-mentor-data:/app/data ai-learning-mentor
```

Open:

```text
http://localhost:8000
```

The Docker volume preserves:

- SQLite conversations
- FAISS index
- chunk metadata
- application logs

across container restarts.

## Technology Stack

### Backend

- Python
- FastAPI
- Pydantic
- SQLite

### AI / RAG

- OpenAI API
- OpenAI Embeddings
- FAISS
- BM25
- tiktoken

### Document Processing

- PyMuPDF

### Infrastructure

- Docker

### Frontend

- HTML
- CSS
- JavaScript

## API Endpoints

### Health

```text
GET /health
```

Returns application and knowledge-base status.

### Upload PDF

```text
POST /upload
```

Processes and indexes a PDF.

### Ask Question

```text
POST /ask
```

Example request:

```json
{
  "session_id": "example-session",
  "question": "What is self-attention?"
}
```

Example response:

```json
{
  "answer": "Self-attention...",
  "source_pages": [21, 22],
  "search_query": "What is self-attention?"
}
```

### Get Conversation

```text
GET /history/{session_id}
```

### Delete Conversation

```text
DELETE /history/{session_id}
```

## Key Concepts Demonstrated

This project demonstrates practical implementation of:

- embeddings
- cosine similarity
- vector databases
- semantic search
- keyword retrieval
- hybrid search
- document chunking
- reranking
- prompt grounding
- RAG
- conversational memory
- structured LLM outputs
- RAG evaluation
- FastAPI
- REST APIs
- Docker
- persistence
- logging and observability

## Future Improvements

Planned improvements include:

- multi-document support
- user authentication
- document selection
- streaming responses
- React frontend
- PostgreSQL
- production vector database
- caching
- rate limiting
- cost tracking
- automated RAG evaluation
- cloud deployment
- CI/CD