import "dotenv/config";
import express from "express";
import { ChatGroq } from "@langchain/groq";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { COLLECTION_NAME, EMBEDDING_MODEL } from "./config.js";

// ── App setup ────────────────────────────────────────────────────────────────
const app = express();
const port = Number(process.env.PORT) || 5000;
app.use(express.json());

// ── LLM ──────────────────────────────────────────────────────────────────────
const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.1,
  maxTokens: 1000,
  maxRetries: 2,
});

// ── Embeddings (RETRIEVAL_QUERY for search-time) ─────────────────────────────
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: EMBEDDING_MODEL,
  taskType: TaskType.RETRIEVAL_QUERY,
});

// ── Vector store (fail fast if collection is missing) ────────────────────────
let vectorStore;
try {
  vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: COLLECTION_NAME,
  });
} catch (err) {
  console.error("Collection not found. Run `npm run ingest` first.");
  process.exit(1);
}

// ── Routes ───────────────────────────────────────────────────────────────────
app.get("/", (_req, res) => {
  return res.json({ message: "Grocery RAG chatbot API is running" });
});

app.post("/ai", async (req, res) => {
  // Input validation
  const { input } = req.body ?? {};

  if (typeof input !== "string" || input.trim().length === 0) {
    return res.status(400).json({ error: "input must be a non-empty string" });
  }

  const trimmed = input.trim();

  if (trimmed.length > 1000) {
    return res
      .status(400)
      .json({ error: "input must be 1000 characters or fewer" });
  }

  try {
    // 1. Find the 4 chunks closest to the question
    const results = await vectorStore.similaritySearch(trimmed, 4);
    const context = results.map((doc) => doc.pageContent).join("\n\n");

    // 2. Give those chunks to the LLM so it answers from the PDF
    const response = await llm.invoke([
      [
        "system",
        `You are a helpful assistant for FreshBasket Super Mart. Answer the question using only the context below. If the answer is not in the context, say you don't know.\n\nContext:\n${context}`,
      ],
      ["human", trimmed],
    ]);

    return res.status(200).json({ answer: response.content });
  } catch (err) {
    console.error("POST /ai error:", err);
    return res.status(500).json({ error: "Something went wrong" });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});