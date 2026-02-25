# Zeva RAG: Carbon Compliance Chatbot

## Overview

This project provides a Retrieval-Augmented Generation (RAG) chatbot, "Zeva", focused on carbon compliance and sustainability regulations (e.g., EU CBAM 2026). It consists of two main components:

- **Document Embedding Pipeline** (`update_embeddings.py`): Scans, chunks, and embeds supported documents (PDF, DOCX, PPTX) into a Chroma vector database using OpenAI embeddings.
- **Streamlit Chat App** (`zeva_app_streamlit.py`): An interactive chatbot UI that retrieves relevant context from your document database and answers compliance-related questions using an LLM.

---

## 1. Document Embedding Pipeline

**Script:** `update_embeddings.py`

- **Purpose:** Detects new/changed/deleted documents, splits them into chunks, computes embeddings, and updates the Chroma vector store.
- **Supported Formats:** `.docx`, `.pdf`, `.pptx` (in the project root).
- **How it works:**
  - Scans for supported files and tracks their state in a manifest.
  - Detects changes (add, modify, delete, rename) using file hashes and metadata.
  - Chunks documents (using `langchain` splitters).
  - Embeds chunks with OpenAI's `text-embedding-3-small` model.
  - Updates the Chroma vector DB (`chroma_chat/`).
- **Usage:**
  ```bash
  python update_embeddings.py
  ```
  Ensure your OpenAI API key is set in a `.env` file.

---

## 2. Zeva Chatbot App

**Script:** `zeva_app_streamlit.py`

- **Purpose:** Provides a Streamlit-based chat interface to query your document knowledge base.
- **Features:**
  - Uses the Chroma vector DB for retrieval-augmented generation.
  - Specialized prompt for carbon compliance and sustainability.
  - Clean, modern chat UI with persistent chat history.
  - Answers are strictly based on your uploaded documents.
- **Usage:**
  ```bash
  streamlit run zeva_app_streamlit.py
  ```
  - Ask questions about carbon regulations, emission data, compliance gaps, etc.
  - Click "New Chat" in the sidebar to reset the conversation.

---

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-key-here
     ```
3. **Add your documents** (PDF, DOCX, PPTX) to the project root.

---

## Folder Structure

- `update_embeddings.py` — Document ingestion and embedding.
- `zeva_app_streamlit.py` — Streamlit chatbot app.
- `chroma_chat/` — Chroma vector DB (auto-generated).
- `.docx_manifest.json` — Tracks document state (auto-generated).
- `requirements.txt` — Python dependencies.

---

## Notes

- Only files in the project root are processed (no subfolders).
- The chatbot will only answer based on the ingested documents.
- For best results, keep your documents up to date and re-run the embedding script after changes.

---
