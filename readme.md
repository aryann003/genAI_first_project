
# GenAI-Based PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive context-grounded answers.

## Features

- PDF document ingestion using PyPDFLoader.
- Recursive text splitting with configurable chunk size and overlap.
- Local text embeddings using HuggingFace.
- Vector storage and similarity retrieval using ChromaDB.
- Maximum Marginal Relevance (MMR) retrieval.
- LLM-based answer generation using Groq.
- Context-grounded prompting with an uncertainty fallback.
- Streamlit interface with retrieved-document inspection.
- Automated unit tests using Pytest.
- Retrieval evaluation using a custom keyword-based Hit Rate@4 script.
- GitHub Actions workflow for automated test execution.

## Tech Stack

- Python
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq
- Streamlit
- Pytest
- GitHub Actions

## Project Structure

```text
genAI_first_project/
├── main.py
├── create_database.py
├── rag_pipeline.py
├── evaluate_retrieval.py
├── sample_pdf.py
├── evaluation/
│   ├── evaluation_dataset.json
│   └── prompt_injection_dataset.json
├── tests/
│   └── test_rag_quality.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aryann003/genAI_first_project.git
cd genAI_first_project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

### 5. Add your PDF

Place your PDF at the path configured in `create_database.py`:

```text
document loaders/deeplearning.pdf
```

### 6. Build the vector database

```bash
python create_database.py
```

### 7. Run the application

```bash
streamlit run main.py
```

## Testing

Run unit tests:

```bash
python -m pytest -v
```

Run retrieval evaluation:

```bash
python evaluate_retrieval.py
```

The retrieval evaluation currently uses a small, manually prepared keyword-based dataset. Its Hit Rate@4 is a basic retrieval indicator, not a semantic relevance or answer-faithfulness score.

## Continuous Integration

GitHub Actions automatically runs the Pytest suite on pushes and pull requests targeting the `main` branch.

## Current Limitations

- The vector database must be generated locally from a PDF.
- The current evaluation dataset is small and document-specific.
- Prompt-injection scenarios have been prepared for manual testing; resistance has not yet been established through a security benchmark.
- Retrieval quality and generated-answer faithfulness require further evaluation.
