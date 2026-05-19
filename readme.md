# Enterprise Two-Stage Hybrid RAG Pipeline with Neural Reranking

A production-ready, high-precision Retrieval-Augmented Generation (RAG) system engineered to solve the limitations of standard vector search. This architecture utilizes a two-stage retrieval methodology combining dense semantic search and sparse keyword matching, paired with a local cross-encoder neural reranking layer for maximum context alignment and strict data citation enforcement.

---

## System Architecture

Standard RAG architectures often suffer from missing exact keyword phrases (like product serial IDs) or pulling irrelevant data chunks due to loose vector distance metrics. This pipeline overcomes those limitations using a decoupled two-stage architecture:

1. **Stage 1: Hybrid Multi-Stream Retrieval**
   * **Dense Semantic Stream:** Encodes text matching using the local `BAAI/bge-small-en-v1.5` transformer model, mapping abstract human intent inside a vector space via ChromaDB.
   * **Sparse Keyword Stream:** Indexes exact structural alphanumeric tokens using a fast, local BM25 engine.
   * **Rank Fusion:** Fuses both retrieval streams deterministically using a custom **Reciprocal Rank Fusion (RRF)** scoring algorithm.

2. **Stage 2: Neural Reranking & Synthesis**
   * **Cross-Encoder Filtering:** Passes the top merged document blocks into a local `ms-marco-MiniLM-L-6-v2` network. Unlike independent vector comparison, it forces token-to-token cross-attention scoring to re-rank the absolute best context windows to the top.
   * **Citation-Enforced Synthesis:** Passes optimized contexts to `gemini-2.0-flash` with strict Pydantic contract guardrails and deterministic regex parsing to track, verify, and isolate specific text snippets, eliminating hallucinations.

---

## Tech Stack & Key Components

* **Framework:** Python, FastAPI, LangChain
* **Databases:** ChromaDB (Local Vector Store), File-system serialized BM25
* **Local ML Models:** `BAAI/bge-small-en-v1.5` (Embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reranker)
* **Cloud Synthesis Model:** Google Gemini API (`gemini-2.0-flash`)
* **Testing & Evaluation:** Pytest, HTTPX, Pydantic

---

##  Repository Structure

```text
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI application server & startup model pre-loading
│   ├── models.py        # Pydantic data schemas & response contract structures
│   ├── ingest.py        # Document loading, chunking, and dual-index building
│   ├── retrieval.py     # Multi-stream search, custom RRF calculation, & reranking
│   └── generation.py    # LLM prompt routing, generation, & regex citation tracking
├── data/                # Source markdown (.md) knowledge files
├── eval/
│   ├── eval_dataset.json  # Golden testing evaluation dataset 
│   └── test_rag_pipeline.py # Integration test runner & contract verification
├── .env.example         # Template for environment variables
├── .gitignore           # Safeguards local DB caches and private API keys
├── requirements.txt     # Complete environment dependencies
└── README.md