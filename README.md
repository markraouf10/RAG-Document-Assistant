# 📄 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF documents and ask questions about them in natural language. Every answer is grounded in your documents and cites the exact source — no hallucination.

Built from scratch without frameworks — no LangChain, no LlamaIndex — to understand every layer of the pipeline.

> *Built as part of the EFG Holding AI Engineering Internship (2026)*

---

## Sample Knowledge Base

The repo ships with 3 foundational AI research papers as a sample dataset:

- *Attention Is All You Need* — Vaswani et al. (2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* — Devlin et al. (2018)
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — Lewis et al. (2020)

You can replace these with any PDF documents of your own or add to them.

---

## Demo

Upload a PDF → ask a question → get a grounded, cited answer.

```
Question: What is multi-head attention?

Answer: Multi-head attention runs multiple attention functions in parallel on different 
learned linear projections of queries, keys and values, then concatenates and projects 
the results. This allows the model to jointly attend to information from different 
representation subspaces at different positions.
[source: Attention is All You Need, Page 4]
```

If the answer isn't in your documents, the system says so instead of making something up.

---

## Features

- 📄 **Upload any PDF** from the UI sidebar — chunked, embedded, and searchable instantly
- 🎯 **Grounded answers** — every response cites the source document and page number
- 🚫 **Honest not-found handling** — says "I could not find that" — no hallucinating 
- 💾 **Persistent storage** — Chroma persists your documents across sessions
- 🔄 **Safe re-ingestion** — upsert prevents duplicate chunks on re-upload
- 📊 **Evaluated** — LLM-as-a-Judge pipeline with cross-family scoring

---

## Pipeline

```
PDF ──▶ Extract text (pypdf) ──▶ Recursive chunking ──▶ Cohere embeddings ──▶ Chroma DB
                                                                                   │
Question ──────────────────────────────────────────────────────── top-8 retrieval │
                                                                                   ▼
                                                         Prompt = question + chunks
                                                                                   ▼
                                                         Groq generates cited answer
                                                                                   ▼
                                                              Answer + citations → UI
```

### Components

| # | Component | What it does |
|---|---|---|
| 1 | **Ingestion** | Load PDFs, extract text, recursive chunking with metadata (filename, page) |
| 2 | **Embedding** | Cohere `embed-v4.0` with `search_document` / `search_query` prefixes |
| 3 | **Vector Store** | Chroma local persistent DB — upsert for safe re-ingestion |
| 4 | **Retrieval** | Top-8 cosine similarity search |
| 5 | **Generation** | Grounded prompt — model instructed to answer only from retrieved context |
| 6 | **Interface** | Streamlit chat UI with live PDF upload in sidebar |
| 7 | **Evaluation** | LLM-as-a-Judge — Gemini judges Groq outputs (cross-family for unbiased scoring) |

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11+ |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | Cohere (`embed-v4.0`) |
| Vector DB | Chroma (local, persistent) |
| PDF Extraction | pypdf |
| UI | Streamlit |
| Evaluation Judge | Google Gemini (`gemini-3.6-flash`) |

---

## Setup

### 1. Clone
```bash
git clone https://github.com/markraouf10/RAG-Document-Assistant.git
cd RAG-Document-Assistant
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. API Keys
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
GEMINI_API_KEY=your_gemini_key
```

| Key | Get it from |
|---|---|
| Groq | [console.groq.com](https://console.groq.com) |
| Cohere | [cohere.com](https://cohere.com) |
| Gemini | [aistudio.google.com](https://aistudio.google.com) |

### 4. Add your documents
Drop PDF files into the `data/` folder — or upload them directly from the UI sidebar at runtime.

### 5. Ingest
```bash
python ingest.py
```

### 6. Run
```bash
streamlit run ui.py
```

### 7. (Optional) Run evaluation
```bash
python evaluate.py
```

---

## Project Structure

```
├── data/                  ← PDF documents (sample: 3 foundational AI papers)
├── eval_logs/
│   └── eval_results.json  ← Evaluation results
│
├── chunking.py            ← Recursive PDF chunking with metadata
├── embeddings.py          ← Embedding generation (Cohere)
├── ingest.py              ← Chunk → embed → upsert into Chroma
├── rag.py                 ← Retrieval + generation pipeline
├── ui.py                  ← Streamlit UI with live PDF upload
├── evaluate.py            ← LLM-as-a-Judge evaluation pipeline
├── eval_dataset.py        ← 15 Q&A pairs with ground truth (based on sample papers)
│
├── system_prompt.md       ← Configurable system prompt (no code changes needed)
├── requirements.txt
├── .env                   ← API keys (never committed)
└── .gitignore
```

---

## Evaluation

Evaluated using **LLM-as-a-Judge** with Google Gemini as the cross-family judge across 15 Q&A pairs sourced directly from the sample knowledge base.

**Why cross-family judging?**
Using a different model family (Gemini) to judge Groq's outputs avoids self-evaluation bias — same-family models tend to favor outputs that match their own style.

| Version | Chunk Size | Overlap | Top-K | Faithfulness | Relevance |
|---|---|---|---|---|---|
| v1 (baseline) | 1000 | 200 | 5 | 4.60 / 5 | 4.53 / 5 |
| v2 (tuned) | 800 | 300 | 8 | **5.00 / 5** | **5.00 / 5** |

**What drove the improvements:**
- Smaller chunks (800) → more focused embeddings, less noise per chunk
- More overlap (300) → key phrases no longer cut at chunk boundaries
- Higher top-k (8) → retriever recovers specific detail answers missed at k=5

All changes were driven by analyzing the eval results.

---

## Acceptance Criteria

- ✅ Answers cite the specific source document and page number
- ✅ When the answer isn't in the documents, the system says so — no fabrication
- ✅ Every pipeline component can be explained and justified
- ✅ Documented evaluation showing measurable improvement from v1 to v2
- ✅ Clone and run from this README alone

---

## Known Limitations

| Limitation | Root Cause | Potential Fix |
|---|---|---|
| Vague queries fail ("tell me about X") | Vague text embeds poorly | Query rewriting with LLM before retrieval |
| Scanned PDFs not supported | pypdf extracts text layer only | OCR (Tesseract / Google Cloud Vision) |
| Images and diagrams ignored | Text-only extraction | Vision Language Model for figure parsing |
| Acronym-only queries can miss | Vocabulary gap | Hybrid retrieval (BM25 + semantic) |

---

## What's Next

1. **Hybrid retrieval** — BM25 + semantic search to fix the vocabulary gap problem
2. **Reranking** — cross-encoder reranking after initial retrieval for higher precision
3. **Query rewriting** — LLM rewrites vague queries before embedding
4. **Conversation memory** — follow-up questions that reference earlier turns

---

## Key Learnings

- **Build without frameworks first** — implementing everything from scratch reveals what LangChain actually abstracts and why each layer exists
- **Retrieval is the foundation** — most RAG failures start at retrieval, not generation
- **Evaluation drives iteration** — without measuring, tuning is just guessing
- **Cross-family judging matters** — a different model family gives more honest evaluation scores

---

**Author:** Mark Raouf  
*Built as part of the EFG Holding AI Engineering Internship (2026)*