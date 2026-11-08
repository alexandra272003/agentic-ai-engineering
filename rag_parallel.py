"""
Parallel Query Retrieval

Flow:
  1. User question -> Groq LLM writes 3 different search questions
  2. The 3 searches run AT THE SAME TIME (threads) in Qdrant
  3. Results are merged: duplicate chunks are removed, and chunks found by more
     queries (then with the higher similarity) are ranked first
  4. Top chunks + ORIGINAL question -> Groq LLM -> final answer

Difference from rag_fusion.py: this one runs the searches concurrently and does a simple
union + dedupe merge. Fusion uses the rank positions (RRF formula) to merge.

Needs rag.py in the same folder and the PDF already indexed (`uv run rag.py`).
Usage:  uv run rag_parallel.py
"""
import os
from pickle import TRUE
import sys
import time
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from groq import Groq

# Reuse the setup from rag.py (same Qdrant collection, same embedding model)
from rag import (
    get_embedder, get_retriever, collection_exists,
    GROQ_MODEL, SYSTEM_PROMPT, QDRANT_URL, COLLECTION_NAME,
)

load_dotenv()

NUM_QUERIES = 3          # questions the LLM writes
K_PER_QUERY = 4          # chunks fetched per question
TOP_N = 8                # chunks sent to the final LLM after merging
INCLUDE_ORIGINAL = True  # True = also search with the user's own question (4 searches)


# ---------------- Step 1: LLM writes 3 questions ----------------
def generate_queries(llm, question, n=NUM_QUERIES):
    prompt = (
        f"Write {n} different search queries for finding the answer to the user's question "
        "in a technical document. Each query should look at it from a different angle "
        "(different wording, a related sub-topic, or a more specific detail). "
        "Return exactly one query per line. No numbering, no bullets, no extra text."
        "The document is about Node.js, Express, Mongoose and Jest. Keep every query within these technologies. "
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

    return queries[:n] or [question]


# ---------------- Step 2: parallel search ----------------
def search_one(store, query):
    return store.similarity_search_with_score(query, k=K_PER_QUERY)


def parallel_search(store, queries):
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(search_one, store, q) for q in queries]
        return [f.result() for f in futures]  # keeps the same order as `queries`


# ---------------- Step 3: merge ----------------
def doc_key(doc):
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def merge_results(results_per_query):
    merged = {}
    for q_no, results in enumerate(results_per_query, start=1):
        for doc, score in results:
            key = doc_key(doc)
            if key not in merged:
                merged[key] = {"doc": doc, "best": score, "found_in": []}
            merged[key]["best"] = max(merged[key]["best"], score)
            merged[key]["found_in"].append(q_no)

    # more queries agreeing first, then higher similarity
    return sorted(merged.values(), key=lambda m: (-len(m["found_in"]), -m["best"]))


# ---------------- Helpers ----------------
def page_of(doc):
    return doc.metadata.get("page", 0) + 1


def snippet(doc, n=70):
    return doc.page_content[:n].replace("\n", " ")


# ---------------- Step 4: final answer ----------------
def answer(llm, question, docs):
    context = "\n\n---\n\n".join(f"[Page {page_of(d)}]\n{d.page_content}" for d in docs)
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},  # original question
        ],
        max_completion_tokens=2500,
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

    print("Parallel RAG ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # 1) LLM makes the questions
            queries = generate_queries(llm, question)
            if INCLUDE_ORIGINAL:
                queries = [question] + queries

            print("\n=== Step 1: Questions used for search ===")
            for i, q in enumerate(queries, 1):
                print(f"  Q{i}: {q}")

            # 2) searches run at the same time
            start = time.perf_counter()
            results_per_query = parallel_search(store, queries)
            elapsed = time.perf_counter() - start

            print(f"\n=== Step 2: {len(queries)} searches ran in parallel ({elapsed:.2f}s) ===")
            for i, results in enumerate(results_per_query, 1):
                row = ", ".join(f"p.{page_of(d)} ({s:.2f})" for d, s in results)
                print(f"  Q{i} -> {row}")

            # 3) merge + dedupe
            merged = merge_results(results_per_query)
            total = sum(len(r) for r in results_per_query)
            print(f"\n=== Step 3: Merged {total} results into {len(merged)} unique chunks (* = sent to LLM) ===")
            print(f"  {'#':<4}{'Page':<6}{'Best':<7}{'Found in':<14}Text")
            for rank, m in enumerate(merged[:TOP_N + 3], 1):
                found = ", ".join(f"Q{q}" for q in m["found_in"])
                mark = "*" if rank <= TOP_N else " "
                print(f" {mark}{rank:<4}{page_of(m['doc']):<6}{m['best']:<7.2f}{found:<14}{snippet(m['doc'])}")

            # 4) chunks + original question -> LLM
            top_docs = [m["doc"] for m in merged[:TOP_N]]
            final = answer(llm, question, top_docs)
            pages = sorted({page_of(d) for d in top_docs})
            print(f"\nAssistant: {final}")
            print(f"\n(Sources: pages {', '.join(map(str, pages))})")

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
