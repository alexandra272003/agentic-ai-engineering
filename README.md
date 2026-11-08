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
#AgenticAI #PromptEngineering #LLM #ChainOfThought #SelfConsistency #Groq #Python #LearningInPublic
