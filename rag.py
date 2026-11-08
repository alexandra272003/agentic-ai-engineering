"""
RAG with LangChain + Qdrant (Docker) + Groq LLM

Flow (from class notes):
  INDEXING : PDF -> load page by page -> split into chunks -> embeddings -> Qdrant
  QUERYING : question -> embedding -> similarity search in Qdrant -> relevant chunks
             -> chunks + question -> Groq LLM -> answer

Embeddings default to your local Ollama (Docker) - free, no API key.

Usage:
  docker compose up -d          # start Qdrant
  docker start ollama           # make sure your Ollama container is running
  uv run rag.py                 # indexes the PDF the first time, then starts chat
  uv run rag.py --reindex       # force re-indexing (e.g. after changing the PDF)
"""
import os
import sys
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# ---------------- Config ----------------
PDF_PATH = Path(os.getenv("PDF_PATH", Path(__file__).parent / "nodejs.pdf"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
BASE_COLLECTION = "learning_langchain"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 4

# "ollama" = embedding model served by your Ollama container (free, default)
# "openai" = text-embedding-3-large (as in your notes, needs OPENAI_API_KEY)
# "local"  = all-MiniLM-L6-v2 via sentence-transformers (free, no Ollama needed)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Different embedding models produce different vector sizes, so each gets its own collection
COLLECTION_NAME = f"{BASE_COLLECTION}_{EMBEDDING_PROVIDER}"

SYSTEM_PROMPT = """You are a helpful AI assistant who responds based on the available context.
Use ONLY the context below. Do not add information from your own knowledge.
Only cite a page if the text in the context from that page supports your statement.
If the context only partly answers the question, answer that part and say what is missing.

Context:
{context}
"""


# ---------------- Embeddings ----------------
def ensure_ollama_model(name):
    """Check Ollama is reachable and pull the embedding model if it's missing."""
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json()
    except Exception:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
                 "Start Docker Desktop, then run:  docker start ollama")

    installed = [m["name"] for m in tags.get("models", [])]
    if name in installed or f"{name}:latest" in installed:
        return

    print(f"Pulling embedding model {name} (first time only) ...")
    with requests.post(f"{OLLAMA_HOST}/api/pull", json={"model": name, "stream": True},
                       stream=True, timeout=None) as r:
        if r.status_code != 200:
            sys.exit(f"Pull failed ({r.status_code}): {r.text}")
        for line in r.iter_lines():
            if not line:
                continue
            info = json.loads(line)
            if "error" in info:
                sys.exit(f"Pull failed: {info['error']}")
            status = info.get("status", "")
            if info.get("total") and info.get("completed"):
                status += f": {info['completed'] * 100 // info['total']}%"
            print("\r" + status.ljust(60), end="", flush=True)
    print("\nEmbedding model ready.\n")


def get_embedder():
    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        ensure_ollama_model(OLLAMA_EMBED_MODEL)
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_HOST)

    if EMBEDDING_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------- Indexing ----------------
def collection_exists():
    client = QdrantClient(url=QDRANT_URL)
    return client.collection_exists(COLLECTION_NAME)


def index_pdf(embedder):
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}\nPut your file there or set PDF_PATH in .env")

    print(f"Loading {PDF_PATH.name} ...")
    docs = PyPDFLoader(file_path=str(PDF_PATH)).load()  # one Document per page
    print(f"  {len(docs)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # overlapping content keeps context between chunks
    )
    split_docs = splitter.split_documents(documents=docs)
    print(f"  {len(split_docs)} chunks created")

    # Recreate the collection so re-indexing never creates duplicates
    QdrantClient(url=QDRANT_URL).delete_collection(COLLECTION_NAME)

    print("Embedding and storing in Qdrant (this can take a minute) ...")
    QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embedder,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print("Indexing done.\n")


# ---------------- Querying ----------------
def get_retriever(embedder):
    return QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
    )


def ask(llm, retriever, question):
    results = retriever.similarity_search(query=question, k=TOP_K)

    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc in results
    )

    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=1000,
    )

    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in results})
    return response.choices[0].message.content, pages, results


def main():
    llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        exists = collection_exists()
    except Exception:
        sys.exit(f"Cannot reach Qdrant at {QDRANT_URL}.\nStart it with:  docker compose up -d")

    embedder = get_embedder()

    if "--reindex" in sys.argv or not exists:
        index_pdf(embedder)
    else:
        print(f"Using existing collection '{COLLECTION_NAME}' (use --reindex to rebuild)\n")

    retriever = get_retriever(embedder)

    print("Ask questions about your PDF. Type 'exit' to quit, 'chunks' to toggle chunk display.")
    show_chunks = False

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question.lower() == "chunks":
            show_chunks = not show_chunks
            print("Show chunks:", "ON" if show_chunks else "OFF")
            continue

        try:
            answer, pages, results = ask(llm, retriever, question)
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")
            continue

        if show_chunks:
            print("\nRelevant chunks:")
            for doc in results:
                print(f"  [page {doc.metadata.get('page', 0) + 1}] {doc.page_content[:150]!r}...")

        print(f"\nAssistant: {answer}")
        print(f"\n(Sources: pages {', '.join(map(str, pages))})")


if __name__ == "__main__":
    main()