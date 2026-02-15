Summary: Corrective RAG (CRAG) Implementation
This code implements a Corrective Retrieval-Augmented Generation (CRAG) workflow using LangGraph.

Key Components:
1. Document Processing:

Loads two PDFs (CRAG.pdf, RNN.pdf)
Splits into chunks using RecursiveCharacterTextSplitter
Stores embeddings in FAISS vector store

2. Workflow Nodes:

Node	Description
retrieve	Gets top 4 similar docs from FAISS
evaluator	Scores each doc's relevance (0-1) using LLM
rewrite_query	Rewrites query for better web search
web_search	Searches web via Tavily API
refine	Splits docs into sentences, keeps only relevant ones
generate	Produces final answer from refined context

3. Routing Logic (Evaluator Verdicts):

Verdict	Condition	Action
Good	Any doc ≥ 0.7	→ refine → generate
Bad	All docs ≤ 0.3	→ rewrite_query → web_search → refine → generate
Ambiguous	Between thresholds	→ rewrite_query → web_search (combines with existing docs) → refine → generate

4. Graph Flow:
START → retrieve → evaluator
                      ↓
        [Good] → refine → generate → END
                      ↓
        [Bad/Ambiguous] → rewrite_query → web_search → refine → generate → END

5. Self-Correction:

If local docs are irrelevant → falls back to web search
For ambiguous cases → combines local + web docs for better context
Sentence-level filtering removes noise before generating answer

