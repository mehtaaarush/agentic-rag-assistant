# Agentic RAG Research Assistant

An agentic Retrieval-Augmented Generation (RAG) assistant that answers natural-language questions over a corpus of computer-vision and deep-learning research papers. Unlike a standard RAG pipeline, this system uses an LLM-powered agent that **decides** which tool to use for each query — retrieving from the local paper corpus, searching the live web, or combining both — and synthesizes a grounded, cited answer.

Built with LangChain, LangGraph, FAISS, and Google Gemini.

---

## Key Features

- **Semantic document retrieval** over 12 CV/DL research papers using a local FAISS vector store and Sentence-Transformers embeddings (no embedding API cost).
- **Agentic tool routing (ReAct pattern):** a LangGraph agent autonomously chooses between a document-retrieval tool and a live web-search tool (Tavily) based on the question.
- **Query reformulation & multi-tool synthesis:** the agent rewrites vague queries into better search terms and can combine local + web results to answer comparative questions.
- **Source-grounded answers with citations** and an explicit anti-hallucination guardrail (the assistant says when it lacks information instead of inventing an answer).
- **Custom LLM-as-judge evaluation harness** measuring faithfulness and answer relevancy across a curated question set.

---

## Architecture

```
                          ┌──────────────────┐
   User question  ───────▶│   LangGraph      │
                          │   ReAct Agent    │
                          └────────┬─────────┘
                                   │  decides which tool
                    ┌──────────────┴───────────────┐
                    ▼                               ▼
          ┌───────────────────┐          ┌────────────────────┐
          │  search_papers    │          │    search_web      │
          │  (FAISS retrieval)│          │  (Tavily web API)  │
          └─────────┬─────────┘          └─────────┬──────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          ┌──────────────────┐
                          │  Gemini LLM      │
                          │  grounded answer │
                          │  + citations     │
                          └──────────────────┘
```

**Ingestion (one-time):** PDFs → text extraction → recursive fixed-size chunking (1000 chars, 200 overlap) → local embeddings → FAISS index saved to disk.

**Query (per question):** question → agent decides tool(s) → retrieve context → grounded prompt → Gemini generates cited answer.

---

## Tech Stack

| Component            | Technology                                      |
|---------------------|-------------------------------------------------|
| LLM                 | Google Gemini (`gemini-flash-latest`)           |
| Agent framework     | LangGraph (ReAct agent)                         |
| Orchestration       | LangChain                                       |
| Vector store        | FAISS (local, flat index)                       |
| Embeddings          | Sentence-Transformers (`all-MiniLM-L6-v2`)      |
| Web search tool     | Tavily                                          |
| PDF parsing         | PyPDF                                           |

---

## Project Structure

```
agentic-rag-assistant/
├── data/                 # source PDFs (research papers)
├── src/
│   ├── config.py         # environment + API test
│   ├── ingest.py         # build the FAISS vector store from PDFs
│   ├── query.py          # basic RAG query pipeline
│   ├── agent.py          # agentic RAG with tool routing (main app)
│   └── evaluate.py       # LLM-as-judge evaluation harness
├── faiss_index/          # generated vector store (git-ignored)
├── .env                  # API keys (git-ignored)
├── requirements.txt
└── README.md
```

---

## Web version

`web/` is a hosted version of the same agent, deployed on Vercel:

- **Retrieval in the browser.** The papers are chunked exactly as `src/ingest.py` does (1,000 characters, 200 overlap) and embedded once with all-MiniLM-L6-v2. The page loads the same 8-bit model with transformers.js, embeds each query the agent writes, and ranks the chunks by cosine similarity.
- **The agent on a Vercel function.** `web/api/agent.js` runs the Gemini ReAct loop with the same two tools and system prompt. It hands `search_papers` calls to the browser and runs `search_web` itself, with Tavily if `TAVILY_API_KEY` is set and Google Search grounding through Gemini otherwise, so a Gemini key is the only one required.
- **A route log.** The page shows every tool the agent chose, the query it wrote and what came back, beside the cited answer.

The web corpus is twelve arXiv papers: Attention Is All You Need, ViT, Swin Transformer, SwinIR, DINO, DINOv2, MAE, GANs, CLIP, ResNet, LoFTR and DDPM. The full text is not committed; rebuild it from the PDFs:

```bash
python web/scripts/build_corpus.py path/to/papers/
cd web && npm install --no-save @huggingface/transformers@3.8.1 && node scripts/embed.mjs
npx vercel deploy --prod      # then: npx vercel env add GEMINI_API_KEY production
```

---

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone https://github.com/mehtaaarush/agentic-rag-assistant.git
   cd agentic-rag-assistant
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Mac/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add API keys** — create a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_ai_studio_key
   TAVILY_API_KEY=your_tavily_key
   ```
   Get keys from [Google AI Studio](https://aistudio.google.com/apikey) and [Tavily](https://tavily.com) (both have free tiers).

4. **Add documents** — place PDF papers in the `data/` folder.

5. **Build the vector store**
   ```bash
   python src/ingest.py
   ```

6. **Run the agent**
   ```bash
   python src/agent.py
   ```

---

## Usage Example

```
Your question: Compare the Swin Transformer to newer vision architectures from 2025

  -> Agent chose tool: search_papers  (query: Swin Transformer shifted window ...)
  -> Agent chose tool: search_web     (query: MambaVision VMamba 2025 architecture ...)

[grounded comparison synthesizing the local Swin Transformer paper
 with current web results, including source citations]
```

The agent recognizes that "Swin Transformer" is covered by the local papers but "2025 architectures" are not, so it uses **both** tools and combines the results.

---

## Evaluation

The system includes a custom **LLM-as-judge** evaluation harness (`src/evaluate.py`) that scores each answer on:

- **Faithfulness** — are the answer's claims supported by the retrieved context? (anti-hallucination)
- **Answer Relevancy** — does the answer actually address the question?

This is the same core technique frameworks like RAGAS use, implemented directly to keep the evaluation lightweight and transparent.

**Results on the current corpus:** faithfulness of **1.0** (answers strictly grounded in retrieved context, no hallucination). Evaluation also surfaced a retrieval-recall gap on abstractly-phrased queries — a known limitation of pure dense (semantic) retrieval — motivating hybrid (BM25 + dense) search as the next iteration.

---

## Design Decisions & Learnings

- **Local embeddings over an API:** using Sentence-Transformers locally removes per-chunk embedding cost, network latency, and keeps documents private.
- **Recursive chunking with overlap:** preserves semantic coherence at chunk boundaries; 200-char overlap prevents ideas from being split across chunks.
- **Anti-hallucination guardrail:** the prompt instructs the model to answer only from retrieved context and to admit when information is missing — verified by the 1.0 faithfulness score.
- **Pure dense retrieval (current):** simple and effective, but evaluation revealed recall gaps on abstract queries. **Planned improvement:** hybrid search combining BM25 keyword matching with dense vectors, A/B tested against the current setup using the built-in eval harness.

---

## Future Work

- Hybrid retrieval (BM25 + dense) with re-ranking to close the recall gap.
- Expand the evaluation set and add context-precision / context-recall metrics.
- Conversation memory for multi-turn follow-up questions.
