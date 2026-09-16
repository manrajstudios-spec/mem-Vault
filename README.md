# mem-Valut

**Stop starting every AI chat from zero.**

mem-Valut is a local-first memory layer for AI assistants — it saves your data, documents, and preferences so that chatting with an AI doesn't mean re-explaining yourself every single time. It's the missing "memory" piece: a system that remembers your conversations, your documents, and (eventually) your preferences, tasks, and decisions, so the AI you talk to actually knows *you* over time.

> ⚠️ **Status: early, in-development prototype.** This is a progress share, not a usable release. See [What Works Today](#what-works-today) below for the honest breakdown.

---

## Why

Most AI chat is stateless — close the tab and it forgets everything. mem-Valut is an attempt to fix that at the memory layer: instead of dumping every chat/doc into one flat vector store, it clusters related information into semantic **groups** that grow and merge over time, and retrieves against those groups using a blend of embedding similarity and keyword overlap. The goal is a memory system that organizes itself the way your own memory does — related things cluster together, and new information merges into what's already there instead of just piling up.

## What Works Today

| Component | Status |
|---|---|
| **Memory save/retrieve** (`mem_handler`) | ✅ Working — saves conversation exchanges, groups them semantically, retrieves relevant past exchanges for a new query |
| **Document ingestion & retrieval** (`doc_handler`) | ✅ Working — PDF text + table extraction, chunking, grouping, retrieval |
| **Web search retrieval** (`websearch`) | ✅ Working — live search results run through the same chunk/group/retrieve pipeline as docs |
| **Code RAG** (`code_rag`) | 🚧 Not working — code parsing/chunking exists, embedding+retrieval not finished |
| **Router** (intent classification) | 🚧 Not working yet |
| **Main flow** (connecting everything end-to-end) | 🚧 Not connected — each piece runs standalone right now |

Nothing is wired together into a single app yet. Each module works in isolation and can be run/tested on its own.

## How It Works

1. **Ingest** — documents (PDF text + tables via PyMuPDF/Camelot), conversation exchanges, or web search results (via DDGS + trafilatura) all get pulled in.
2. **Chunk** — text is split into sentence-aware chunks (spaCy) around an ~800-character target.
3. **Embed + keyword-score** — each chunk is embedded (`sentence-transformers`) and scored for keywords (KeyBERT).
4. **Group** — chunks are clustered into semantic groups using a similarity threshold over embeddings + keyword overlap, so retrieval pulls a coherent cluster of related information instead of disconnected fragments.
5. **Persist** — memory groups are saved to disk and *merge* with existing groups on future saves, rather than just growing a flat list.
6. **Retrieve** — a query is embedded and keyword-scored the same way, matched first against group-level summaries (fast, coarse), then reranked within the best-matching groups (fine-grained).

## Current Performance

Rough numbers from local testing (no GPU-specific tuning yet), on the 32-page FlyWire connectome paper:

**Ingesting the document:**
| Stage | Time |
|---|---|
| Reading text + extracting tables from PDF | ~10s |
| Sentence splitting | ~2.7s |
| Chunking | ~0.005s |
| Embedding + keyword extraction | ~3.5s |
| Grouping | ~0.004s |

The paper was split into **201 chunks**, which self-organized into **32 semantic groups** — a ~6x compression before retrieval even starts. Total time from raw PDF to a fully grouped, query-ready document: **~16 seconds** for 32 pages.

**Retrieving from it** (2 queries about neuron reconstruction methods):
| Stage | Time |
|---|---|
| Query embed + keyword extraction | ~0.019s |
| Embedding similarity scoring | ~0.000025s |
| Keyword scoring | ~0.00006s |

Once a document is ingested and grouped, matching a new query against it is effectively instant — the cost is almost entirely front-loaded into ingestion, not lookup. Retrieval quality/quantity is also tunable: how many chunks come back can be adjusted without sacrificing relevance, since scoring only surfaces chunks that clear the similarity + keyword threshold.

## Stack

- **LLM**: Ollama (local), currently `granite4.1:3b`
- **Embeddings**: `sentence-transformers` (`multi-qa-distilbert-cos-v1`)
- **Code embeddings**: `microsoft/unixcoder-base`
- **Keywords**: KeyBERT
- **PDF parsing**: PyMuPDF (text) + Camelot (tables)
- **NLP**: spaCy (`en_core_web_sm`)
- **Web search**: DDGS + trafilatura

## Known Limitation

Retrieval currently selects at the **group level** — a query matches a semantic group, and every chunk in that group is returned. This means loosely-related chunks that got clustered alongside genuinely relevant ones come along for the ride, which hurts precision. Fix in progress: reranking individual chunk embeddings *within* selected groups against the raw query, rather than returning whole groups wholesale.

## Roadmap

- [ ] Rerank chunk-level embeddings within selected groups (precision fix)
- [ ] Finish code RAG (AST-based chunking → embedding → retrieval)
- [ ] Finish the router (intent classification: websearch / doc / memory / preferences / tasks / events / decisions)
- [ ] Connect everything into a single main flow
- [ ] User preference, task, event, and decision tracking (beyond raw conversation memory)
- [ ] Performance tuning on chunking/grouping for larger documents

## Contributing

This is very much a work in progress and not ready for external use yet — but if you're interested in local-first AI memory systems, feel free to star/watch the repo for updates.

---

*Built for anyone tired of AI that forgets who they're talking to.*
