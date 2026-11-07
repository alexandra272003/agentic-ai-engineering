# Part 01 – Tokenization & Word Embeddings

Welcome to Class 01 of my **Agentic AI Engineering** learning journey.

This class focuses on two fundamental concepts in Natural Language Processing (NLP): building a custom tokenizer from scratch and understanding how machines represent words as numerical vectors.

## 📚 Topics Covered

### 1. Custom Tokenizer (Built from Scratch)

Implemented my own character-level tokenizer using Python without relying on external NLP tokenization libraries.

**Features:**
- Custom vocabulary dictionary.
- Character-to-token ID mapping.
- Encoder: Converts text into numerical token IDs.
- Decoder: Reconstructs text from token IDs.
- Reverse vocabulary for decoding.

**Example:**

```python
message = "hello world"

Encoded:
[8, 5, 12, 12, 15, 27, 23, 15, 18, 12, 4]

Decoded:
hello world
```

**Concepts learned:**
- Tokenization
- Vocabulary creation
- Token ID mapping
- Encoding and decoding
- Character-level tokenization

### 2. Word Embeddings

Implemented a Word Embeddings system using the pretrained `all-MiniLM-L6-v2` model from Sentence Transformers.

The objective is to understand how natural language is transformed into numerical vector representations.

**Features:**
- Encoder for generating numerical embeddings.
- 384-dimensional vector representations.
- Cosine similarity calculation.
- Similarity matrix generation.
- Heatmap visualization using Seaborn.
- PCA-based 2D embedding visualization.
- Semantic search to retrieve similar words.

**Example Input:**

```python
words = [
    "king", "queen", "man", "woman",
    "apple", "banana", "car", "vehicle",
    "doctor", "teacher"
]
```

**Embedding output:**

```text
Embedding Shape: (10, 384)
```

This represents 10 input words, each mapped to a 384-dimensional vector.

**Technologies used:**
- Python
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Sentence Transformers
- PCA
- Cosine Similarity

## 📂 Project Structure

```text
Class-01/
│
├── tokenizer.ipynb
├── word_embeddings.ipynb
└── README.md
```

## ⚙️ Installation

This project uses `uv` for Python dependency management.

Install the required packages:

```bash
uv add sentence-transformers numpy matplotlib seaborn scikit-learn
```

For the custom tokenizer, no external NLP library is required.

## ▶️ Execution

Open the notebooks in Jupyter Notebook or VS Code:

1. `tokenizer.ipynb`
2. `word_embeddings.ipynb`

Run the cells sequentially.

**Note:** The first execution of the Sentence Transformer model requires an internet connection to download the pretrained model.

## 🎯 Learning Outcomes

After completing this class, I learned:

- How tokenizers convert text into numerical IDs.
- How encoder and decoder functions work.
- How word embeddings represent language numerically.
- How to calculate semantic similarity using cosine similarity.
- How to visualize high-dimensional embeddings using PCA.
- How semantic search retrieves words based on vector similarity.

# Part 02 – Prompt Engineering & Prompting Techniques

Welcome to Class 02 of my **Agentic AI Engineering** learning journey.

This class focused on understanding how Large Language Models (LLMs) interpret instructions and how effective prompt design influences their responses.

I explored fundamental prompting concepts, different prompt formats, and practical prompting techniques by implementing a Chain of Thought (CoT) assistant using Python and the Groq API.

## 📚 Topics Covered

### 1. Introduction to Prompt Engineering

Learned the fundamentals of communicating with Large Language Models through prompts.

**Concepts learned:**
- Large Language Models (LLMs)
- Introduction to Prompt Engineering
- Garbage In, Garbage Out (GIGO)
- Initial prompts and input tokens
- How LLMs interpret and respond to instructions
- Importance of clear and structured prompts

### 2. Prompt Formats and Structures

Explored different prompt formats used to structure instructions and conversations for LLMs.

**Formats covered:**
- Basic Prompt Format
- Alpaca Prompt Format
- INST Format
- LLaMA 2 Prompt Format

### 3. Chain of Thought (CoT) Prompting

Created a Python file named `COT.py` to experiment with Chain of Thought prompting using the Groq API.

The objective was to make an AI assistant break down questions into smaller steps and provide structured explanations.

**Implementation features:**
- System prompt configuration
- User and assistant message management
- Conversation history using message appending
- Interactive chat using a `while` loop
- Groq API integration
- Step-by-step mathematical problem solving
- Response extraction using `result.choices[0].message.content`

**Five-step problem-solving approach:**

1. Analysis
2. Think
3. Output
4. Validate
5. Result

### 4. Prompting Techniques

Explored different prompting strategies and understood how they influence LLM responses.

**Techniques learned:**

- **Chain of Thought Prompting:** Encourages structured, step-by-step problem-solving explanations.
- **Self-Consistency Prompting:** Generates multiple reasoning attempts and compares their answers to identify a consistent result.
- **Persona-Based Prompting:** Assigns a particular identity or expertise to guide the response.
- **Role-Play Prompting:** Makes the model respond from a specified role or situation.

## ⚙️ Technologies Used

- Python
- Groq API
- Groq Python SDK
- JSON
- Python-dotenv
- Prompt Engineering
- Large Language Models (LLMs)

## ▶️ Execution

Install the required packages:

```bash
uv add groq python-dotenv
```

Configure your Groq API key in a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run the Chain of Thought assistant:

```bash
python COT.py
```

## 🎯 Learning Outcomes

After completing this class, I learned:

- How Large Language Models process prompts.
- Why prompt quality matters (Garbage In, Garbage Out).
- How different prompt formats structure LLM instructions.
- How system prompts guide model behaviour.
- How to maintain conversation history using message appending.
- How to build an interactive LLM assistant using Python.
- How Chain of Thought prompting structures problem-solving explanations.
- How Self-Consistency, Persona-Based, and Role-Play prompting work.

---

# Part 03 – RAG with LangChain, Qdrant & Groq

Welcome to Class 03 of my **Agentic AI Engineering** learning journey.

This class focused on **Retrieval-Augmented Generation (RAG)**: giving an LLM access to my own documents so it answers from real source material instead of guessing.

I built a complete RAG pipeline in `rag.py` that reads a Node.js PDF, stores it in a **Qdrant vector database running in Docker**, and answers questions using the **Groq API** as the LLM.

## 📚 Topics Covered

### 1. What is RAG?

LLMs only know what they were trained on. RAG fixes this by retrieving relevant pieces of my own data and passing them to the LLM along with the question.

**The RAG flow:**

```text
INDEXING (done once)
Data source (PDF) → load page by page → split into chunks
                  → embedding model → vectors stored in vector DB (Qdrant)

QUERYING (every question)
1. User question → embedding model → question vector
2. Search the vector DB → most relevant chunks from the data source
3. Relevant chunks + question → LLM
4. LLM → gives the answer
```

**Concepts learned:**
- Retrieval-Augmented Generation (RAG)
- Why LLMs need external knowledge
- Indexing vs querying
- Storing paragraph text with the vector, so the data source is not read again
- Grounding answers in context to reduce hallucination

### 2. Document Loading

Used LangChain's `PyPDFLoader` to read the PDF.

- Loads the PDF **page by page** (one `Document` per page).
- Each document keeps metadata such as the page number, used later for source references.

### 3. Text Splitting (Chunking)

Large pages are split into smaller chunks so retrieval is precise and fits the LLM context.

- `RecursiveCharacterTextSplitter` splits on paragraphs and sentences first.
- `chunk_size = 1000`
- `chunk_overlap = 200`: neighbouring chunks share some content so context is not lost at the boundaries.

### 4. Embeddings

Each chunk is converted into a vector so that similar meanings sit close together.

**Embedding options explored:**

| Provider | Model | Cost |
|----------|-------|------|
| OpenAI | `text-embedding-3-large` | Paid API |
| Ollama (Docker) | `nomic-embed-text` | Free, runs locally |
| Sentence Transformers | `all-MiniLM-L6-v2` | Free, runs locally |

**Important:** each embedding model produces a different vector size, so every provider gets its own Qdrant collection. Changing the model requires re-indexing.

### 5. Vector Database – Qdrant (Docker)

Qdrant is the vector database that stores the chunk vectors and performs similarity search.

Started with Docker Compose:

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

- Runs at `http://localhost:6333`.
- Dashboard available at `http://localhost:6333/dashboard`.
- A Docker volume keeps the stored vectors after restart.

### 6. Storing and Retrieving

**Store:** `QdrantVectorStore.from_documents(...)` embeds the split chunks and saves them in the `learning_langchain` collection. This is only needed once, so indexing is skipped on later runs.

**Retrieve:** `QdrantVectorStore.from_existing_collection(...)` connects to the saved collection, and `similarity_search(query=...)` returns the most relevant chunks for a question.

### 7. Generating the Answer with Groq

The retrieved chunks are inserted into the system prompt as context:

```text
You are a helpful AI assistant who responds based on the available context.
```

The Groq LLM answers using only that context. If the answer is not in the retrieved chunks, it says it does not know. The answer also shows the **source page numbers**.

## 📊 Example Run

```text
125 pages loaded
266 chunks created
Embedding and storing in Qdrant ...
Indexing done.

You: what is JSON Web Tokens
Assistant: JSON Web Tokens (JWTs) are a compact, URL-safe way to represent claims ...
(Sources: pages 19, 81, 82)
```

## ⚙️ Technologies Used

- Python
- LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-qdrant`)
- PyPDF
- Qdrant (vector database)
- Docker & Docker Compose
- Groq API (LLM)
- Ollama / Sentence Transformers / OpenAI (embeddings)
- Python-dotenv

## 📂 Project Structure

```text
Class-03/
│
├── rag.py
├── docker-compose.yml
├── nodejs.pdf
├── .env
└── README.md
```

## ⚙️ Installation

Install the required packages:

```bash
uv add langchain-community langchain-text-splitters langchain-qdrant qdrant-client pypdf langchain-ollama
```

Optional embedding providers:

```bash
uv add langchain-openai        # OpenAI embeddings (paid)
uv add langchain-huggingface   # local Sentence Transformers (free)
```

Configure `.env`:

```env
GROQ_API_KEY=your_groq_api_key

# ollama (default) | local | openai
EMBEDDING_PROVIDER=ollama

# Only if using OpenAI embeddings
# OPENAI_API_KEY=your_openai_api_key
```

## ▶️ Execution

1. Start Docker Desktop, then start Qdrant (and Ollama if used for embeddings):

```bash
docker compose up -d
docker start ollama
```

2. Place the PDF in the project folder as `nodejs.pdf`.

3. Run the RAG assistant:

```bash
uv run rag.py
```

The PDF is indexed on the first run only. To rebuild the vectors after changing the PDF or embedding model:

```bash
uv run rag.py --reindex
```

Type `chunks` while chatting to see which chunks were retrieved, and `exit` to quit.

## 🎯 Learning Outcomes

After completing this class, I learned:

- What RAG is and why it reduces hallucination.
- How to load PDFs page by page with `PyPDFLoader`.
- How chunk size and chunk overlap affect retrieval.
- How embedding models turn text into vectors, and why vector sizes differ between models.
- How to run a Qdrant vector database with Docker Compose.
- How to store vectors and run similarity search with LangChain and Qdrant.
- How to pass retrieved context and a question to an LLM through a system prompt.
- How to show source pages so answers can be verified.
- How to run embeddings and models locally with Ollama in Docker.

---
#AgenticAI #RAG #Qdrant #LangChain #VectorDatabase #Docker #Groq #Ollama #Python #LearningInPublic
