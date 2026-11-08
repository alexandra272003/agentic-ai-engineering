"""
RAG Fusion = multi-query retrieval + Reciprocal Rank Fusion (RRF)

Flow:
  1. User question -> Groq LLM writes 3 different versions of the question
  2. Each version -> embedding -> similarity search in Qdrant (3 ranked lists)
  3. Reciprocal Rank Fusion merges the 3 ranked lists into one
  4. Top fused chunks + ORIGINAL question -> Groq LLM -> final answer

RRF formula:   score(chunk) = sum over lists of  1 / (k + rank)      (k = 60, rank starts at 1)
A chunk that appears high in several lists beats a chunk that is #1 in only one list.

Needs rag.py in the same folder, and the PDF already indexed (run `uv run rag.py` once).
Usage:  uv run rag_fusion.py
"""
import os
import sys
import hashlib
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from groq import Groq

# Reuse the setup from rag.py (same Qdrant collection, same embedding model)
from rag import (
    get_embedder, get_retriever, collection_exists,
    GROQ_MODEL, SYSTEM_PROMPT, QDRANT_URL, COLLECTION_NAME,
)

load_dotenv()

NUM_QUERIES = 3        # how many question variations the LLM creates
K_PER_QUERY = 5        # chunks fetched from Qdrant for each variation
TOP_N = 4              # chunks sent to the LLM after fusion
RRF_K = 60             # standard RRF constant


# ---------------- Step 1: LLM writes 3 questions ----------------
def generate_queries(llm, question, n=NUM_QUERIES):
    prompt = (
        f"Write {n} different search queries that look at the user's question from "
        "different angles (rephrase it, use synonyms, or ask about a related sub-topic). "
        "Return exactly one query per line. No numbering, no bullets, no extra text."
    )
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=800,
    )
    text = response.choices[0].message.content or ""

    queries = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789.) ").strip()
        if line:
            queries.append(line)
    queries = queries[:n]

    return queries or [question]  # fallback: use the original question


# ---------------- Step 3: Reciprocal Rank Fusion ----------------
def doc_key(doc):
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """ranked_lists: list of lists of Documents (best first).
    Returns [(doc, rrf_score, [(query_no, rank), ...])] sorted best first."""
    scores = defaultdict(float)
    docs = {}
    seen_in = defaultdict(list)

    for q_no, results in enumerate(ranked_lists, start=1):
        for rank, doc in enumerate(results, start=1):
            key = doc_key(doc)
            scores[key] += 1.0 / (k + rank)
            docs[key] = doc
            seen_in[key].append((q_no, rank))

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(docs[key], scores[key], seen_in[key]) for key in ordered]


# ---------------- Helpers for printing ----------------
def page_of(doc):
    return doc.metadata.get("page", 0) + 1


def snippet(doc, n=80):
    return doc.page_content[:n].replace("\n", " ")


# ---------------- Step 4: final answer ----------------
def answer(llm, question, fused_top):
    context = "\n\n---\n\n".join(
        f"[Page {page_of(doc)}]\n{doc.page_content}" for doc, _, _ in fused_top
    )
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},  # original question, not the variations
        ],
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content


def main():
    llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        if not collection_exists():
            sys.exit(f"Collection '{COLLECTION_NAME}' not found. Run `uv run rag.py` first to index your PDF.")
    except Exception:
        sys.exit(f"Cannot reach Qdrant at {QDRANT_URL}.\nStart it with:  docker compose up -d")

    store = get_retriever(get_embedder())

    print("RAG Fusion ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # 1) LLM makes the question variations
            queries = generate_queries(llm, question)
            print("\n=== Step 1: LLM generated queries ===")
            for i, q in enumerate(queries, 1):
                print(f"  Q{i}: {q}")

            # 2) similarity search in Qdrant for each variation
            ranked_lists = [store.similarity_search(q, k=K_PER_QUERY) for q in queries]
            print("\n=== Step 2: Qdrant results per query (rank: page) ===")
            for i, results in enumerate(ranked_lists, 1):
                row = ", ".join(f"#{r}: p.{page_of(d)}" for r, d in enumerate(results, 1))
                print(f"  Q{i} -> {row}")

            # 3) Reciprocal Rank Fusion
            fused = reciprocal_rank_fusion(ranked_lists)
            print(f"\n=== Step 3: After Reciprocal Rank Fusion (k={RRF_K}) ===")
            print(f"  {'Rank':<5}{'Score':<9}{'Page':<6}{'Found in (query#rank)':<26}Text")
            for rank, (doc, score, seen) in enumerate(fused[:TOP_N + 3], 1):
                found = ", ".join(f"Q{q}#{r}" for q, r in seen)
                mark = "*" if rank <= TOP_N else " "
                print(f" {mark}{rank:<5}{score:<9.4f}{page_of(doc):<6}{found:<26}{snippet(doc)}")
            print(f"  (* = top {TOP_N} chunks sent to the LLM)")

            # 4) chunks + original question -> LLM
            final = answer(llm, question, fused[:TOP_N])
            pages = sorted({page_of(doc) for doc, _, _ in fused[:TOP_N]})
            print(f"\nAssistant: {final}")
            print(f"\n(Sources: pages {', '.join(map(str, pages))})")

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
