"""
HyDE = Hypothetical Document Embeddings

Idea: a question and its answer often look very different, so embedding the
QUESTION can match the wrong chunks. HyDE asks the LLM to first WRITE A FAKE
ANSWER (a hypothetical document), then embeds that fake answer. A passage that
looks like a real answer lands closer to the real answer chunks in the PDF.

Flow:
  1. User question -> Groq LLM writes a hypothetical answer passage
  2. Hypothetical passage -> embedding -> similarity search in Qdrant
  3. Real chunks found (the fake passage is thrown away after searching)
  4. Real chunks + ORIGINAL question -> Groq LLM -> final answer

The program also runs a normal search with the plain question so you can
compare plain retrieval vs HyDE retrieval side by side.

Needs rag.py in the same folder and the PDF already indexed (`uv run rag.py`).
Usage:  uv run rag_hyde.py
"""
import os
import sys
import hashlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from groq import Groq

# Reuse the setup from rag.py (same Qdrant collection, same embedding model)
from rag import (
    get_embedder, get_retriever, collection_exists,
    GROQ_MODEL, SYSTEM_PROMPT, QDRANT_URL, COLLECTION_NAME,
)

load_dotenv()

TOP_K = 4  # chunks sent to the final LLM

HYDE_PROMPT = """You are writing a short passage as it might appear in a Node.js textbook or documentation.
Write a passage of about 100-150 words that directly answers the user's question.
Use the style of a technical book: explain the idea and include a short code example if it helps.
Do not say you are guessing and do not mention the question. Output only the passage."""


# ---------------- Step 1: LLM writes the hypothetical document ----------------
def generate_hypothetical_doc(llm, question):
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=800,
    )
    text = (response.choices[0].message.content or "").strip()
    return text or question  # fallback: plain question


# ---------------- Helpers ----------------
def doc_key(doc):
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def page_of(doc):
    return doc.metadata.get("page", 0) + 1


def snippet(doc, n=70):
    return doc.page_content[:n].replace("\n", " ")


def print_results(title, results, other_keys):
    print(f"\n{title}")
    print(f"  {'Rank':<5}{'Score':<8}{'Page':<6}{'Match':<8}Text")
    for rank, (doc, score) in enumerate(results, 1):
        tag = "both" if doc_key(doc) in other_keys else "only"
        mark = "*" if rank <= TOP_K else " "
        print(f" {mark}{rank:<5}{score:<8.3f}{page_of(doc):<6}{tag:<8}{snippet(doc)}")


# ---------------- Step 4: final answer ----------------
def answer(llm, question, docs):
    context = "\n\n---\n\n".join(f"[Page {page_of(d)}]\n{d.page_content}" for d in docs)
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},  # original question
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

    print("HyDE RAG ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # 1) LLM writes a fake answer
            hypo = generate_hypothetical_doc(llm, question)
            print("\n=== Step 1: Hypothetical document (used only for searching) ===")
            print(hypo)

            # 2) search with the fake answer, and with the plain question for comparison
            hyde_results = store.similarity_search_with_score(hypo, k=TOP_K + 2)
            plain_results = store.similarity_search_with_score(question, k=TOP_K + 2)

            hyde_keys = {doc_key(d) for d, _ in hyde_results[:TOP_K]}
            plain_keys = {doc_key(d) for d, _ in plain_results[:TOP_K]}

            print("\n=== Step 2: Retrieval comparison (* = top chunks used) ===")
            print_results("Plain question search:", plain_results, hyde_keys)
            print_results("HyDE search (embedding of the hypothetical document):", hyde_results, plain_keys)

            new_chunks = hyde_keys - plain_keys
            print(f"\n  HyDE found {len(new_chunks)} chunk(s) that plain search missed in its top {TOP_K}.")

            # 3) real chunks + original question -> LLM
            top_docs = [d for d, _ in hyde_results[:TOP_K]]
            final = answer(llm, question, top_docs)
            pages = sorted({page_of(d) for d in top_docs})
            print(f"\nAssistant: {final}")
            print(f"\n(Sources: pages {', '.join(map(str, pages))})")

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
