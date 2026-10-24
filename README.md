# Grocery RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for FreshBasket Super Mart, powered by LangChain, Qdrant, Google Gemini embeddings, and Groq LLM.

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=grocery-store
PORT=5000
```

### 3. Ingest the knowledge base

```bash
npm run ingest
```

This reads `knowledge.pdf`, splits it into chunks, and uploads them to Qdrant. Safe to re-run — it deletes and recreates the collection each time.

### 4. Start the development server

```bash
npm run dev
```

## API Usage

### Health check

```bash
curl http://localhost:5000/
```

### Ask a question

```bash
curl -X POST http://localhost:5000/ai \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the price of organic apples?"}'
```

**Response:**

```json
{
  "answer": "Organic apples are priced at $X.XX per kg."
}
```
