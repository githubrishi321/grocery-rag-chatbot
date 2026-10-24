import "dotenv/config";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { COLLECTION_NAME, EMBEDDING_MODEL } from "./config.js";

// ── Constants ────────────────────────────────────────────────────────────────
const BATCH_SIZE = 10;
const BATCH_DELAY_MS = 10_000; // 10 seconds between batches

// ── Helpers ──────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ── Main ─────────────────────────────────────────────────────────────────────
async function ingest() {
  // 1. Parse the PDF
  console.log("📄 Reading knowledge.pdf …");
  const buffer = fs.readFileSync("./knowledge.pdf");
  const pdfParser = new PDFParse({ data: buffer });
  const { text } = await pdfParser.getText();

  // 2. Split into chunks (size/overlap unchanged: 500 / 100)
  console.log("✂️  Splitting text into chunks …");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });
  const docs = await splitter.createDocuments([text]);
  console.log(`   ${docs.length} chunks created.`);

  // 3. Delete existing collection so ingest is idempotent
  const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
  });

  const { collections } = await qdrantClient.getCollections();
  const exists = collections.some((c) => c.name === COLLECTION_NAME);

  if (exists) {
    console.log(`🗑️  Collection "${COLLECTION_NAME}" already exists — deleting …`);
    await qdrantClient.deleteCollection(COLLECTION_NAME);
    console.log("   Deleted.");
  }

  // 4. Embeddings (RETRIEVAL_DOCUMENT for ingestion)
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: EMBEDDING_MODEL,
    taskType: TaskType.RETRIEVAL_DOCUMENT,
  });

  // 5. Upload first batch by creating the collection, then remaining batches
  console.log("⬆️  Uploading chunks to Qdrant …");

  const firstBatch = docs.slice(0, BATCH_SIZE);
  console.log(`   Batch 1 / ${Math.ceil(docs.length / BATCH_SIZE)} (chunks 1–${firstBatch.length}) …`);

  const vectorStore = await QdrantVectorStore.fromDocuments(firstBatch, embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: COLLECTION_NAME,
  });

  const totalBatches = Math.ceil(docs.length / BATCH_SIZE);

  for (let batchIndex = 1; batchIndex < totalBatches; batchIndex++) {
    const start = batchIndex * BATCH_SIZE;
    const end = Math.min(start + BATCH_SIZE, docs.length);
    const batch = docs.slice(start, end);

    // Pause before each subsequent batch (not after the last one)
    await sleep(BATCH_DELAY_MS);

    console.log(
      `   Batch ${batchIndex + 1} / ${totalBatches} (chunks ${start + 1}–${end}) …`
    );
    await vectorStore.addDocuments(batch);
  }

  console.log(`\n✅ Done! ${docs.length} chunks stored in collection "${COLLECTION_NAME}".`);
}

ingest().catch((err) => {
  console.error("Ingestion failed:", err);
  process.exit(1);
});
