# Class 01 – Tokenization & Word Embeddings

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

---
#AgenticAI #NLP #Tokenization #WordEmbeddings #Python #MachineLearning #LearningInPublic
