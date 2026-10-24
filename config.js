// Shared configuration used by both index.js and ingest.js
export const COLLECTION_NAME = process.env.COLLECTION_NAME ?? "grocery-store";
export const EMBEDDING_MODEL = "gemini-embedding-001";
