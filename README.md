# Mahabharata RAG

A simple local Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about a Mahabharata PDF and receive answers grounded in retrieved passages from the source document.

The project demonstrates the core components of a RAG system:

- PDF ingestion
- Text chunking
- Sentence-transformer embeddings
- FAISS vector search
- Context retrieval
- Prompt construction
- Local LLM inference using Ollama
- Source/page citations
- Interactive CLI

---

## Architecture

```text
                    Mahabharata PDF
                           |
                           v
                    PDF Ingestion
                           |
                           v
                    Text Chunking
                    1000 chars
                    200 overlap
                           |
                           v
                Sentence Transformer
                all-mpnet-base-v2
                           |
                           v
                 768-dimensional
                    embeddings
                           |
                           v
                    FAISS Index
                           |
                           |
              -------------------------
              |                       |
              |                       |
           User Query             Stored Chunks
              |                       |
              v                       |
       Query Embedding                |
              |                       |
              v                       |
          FAISS Search <--------------
              |
              v
          Top-K Chunks
          (TOP_K = 10)
              |
              v
       Prompt Construction
              |
              v
         Ollama / Llama
              |
              v
       Grounded Answer
              |
              v
        Source Page Numbers



How RAG Works in This Project



The pipeline follows these steps:

1. Document ingestion

The Mahabharata PDF is loaded and each page is represented with its page number and text.


2. Chunking

The extracted text is divided into chunks of approximately 1000 characters with an overlap of 200 characters.
The overlap helps preserve context when information spans across chunk boundaries.

Example:

Chunk 1
|--------------------1000 characters--------------------|
                         |
                         | 200 character overlap
                         v
                   Chunk 2
                   |--------------------1000--------------------|


3. Embeddings

Each text chunk is converted into a numerical vector using:
sentence-transformers
all-mpnet-base-v2
Each embedding contains:

768 dimensions

For this document:

20,006 chunks
20,006 embeddings
768 dimensions per embedding


4. Vector storage

The embeddings are indexed using FAISS.
The current index uses:

FAISS IndexFlatL2
Metric: L2 distance
Vectors: 20,006
Dimensions: 768

The retriever searches the vector index for chunks that are semantically similar to the user's question.


5. Retrieval

The query is converted into the same embedding space and FAISS retrieves the most similar chunks.
The current configuration retrieves:
TOP_K = 10

6. Prompt construction

The retrieved chunks are inserted into the prompt along with their source page numbers.

Example:

[Page 945]
Retrieved text...

[Page 1187]
Retrieved text...

The LLM is instructed to answer using only the supplied context.

7. Local LLM generation

The retrieved context and user question are sent to a locally running LLM through Ollama.
The model is instructed not to invent information that is not supported by the retrieved context.

8. Source citations

The generated answer is instructed to associate factual claims with relevant source pages.

Example:

The context states that Karna was impelled by a desire
to do good to Duryodhana. [Page 945]
Technologies
Component	Technology
Language	Python
PDF processing	PyMuPDF
Chunking	Custom Python implementation
Embeddings	Sentence Transformers
Embedding model	all-mpnet-base-v2
Vector database	FAISS
Vector index	IndexFlatL2
LLM	Llama 3.2
Local inference	Ollama
Interface	Command Line Interface
Serialization	Pickle / NumPy
Project Structure
mahabharata-rag/
│
├── data/
│   └── source PDF
│
├── src/
│   ├── chunking.py
│   ├── config.py
│   ├── embedding.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── prompt.py
│   ├── rag.py
│   ├── retriever.py
│   ├── vector_store.py
│   ├── verify_storage.py
│   └── __init__.py
│
├── storage/
│   ├── chunks.pkl
│   ├── embeddings.npy
│   └── index.faiss
│
├── .gitignore
├── requirements.txt
└── README.md

Generated files under storage/ and the source document under data/ are intentionally excluded from Git.


Setup

1. Clone the repository
git clone https://github.com/kumarankit1612/mahabharata-rag.git

cd mahabharata-rag
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

4. Add the source document

Place the Mahabharata PDF inside:

data/

The PDF is not included in this repository.

Generate the RAG Data

Run the embedding pipeline:

python -m src.embedding

This creates:

storage/
├── embeddings.npy
└── chunks.pkl

The FAISS index can then be created using the vector-store pipeline.

The generated artifacts are intentionally excluded from Git because they are derived data.

Run the Application

Make sure Ollama is installed and the required model is available locally.

The project currently uses:

llama3.2:3b

Start the RAG CLI:

python -m src.rag

The application will load the embedding model and vector store once and then allow multiple questions in the same session.

Example:

================================================================================
MAHABHARATA RAG
================================================================================

Loading embedding model...
Loading FAISS index and chunks...

Ready! Ask questions about the Mahabharata.
Type 'exit' or 'quit' to stop.

Question: Who was Arjuna's teacher?
Example
Question

Why did Karna support Duryodhana?
Retrieved context

The retriever identifies relevant passages from the document, including passages such as Page 945.

Answer

The system is instructed to distinguish between information explicitly supported by the retrieved context and information that cannot be determined from that context.

For example:

The provided context does not explicitly state why Karna
supported Duryodhana. However, it states that Karna was
impelled by a desire to do good to Duryodhana. [Page 945]
Evaluation

The system was tested using questions covering both answerable and insufficient-context scenarios.

Question	Observed behavior
Who was Arjuna's teacher?	Retrieved relevant material about Drona
Why did the Pandavas go into exile?	Retrieved relevant and partial contextual information
Who was Draupadi?	Retrieved relevant material
What happened during the dice game?	Retrieval improved after increasing TOP_K from 5 to 10
Why did the Kurukshetra war take place?	Correctly recognized insufficient retrieved context
What was Karna's favorite food?	Correctly recognized that the retrieved context did not contain the information
Retrieval experiment

The question:

What happened during the dice game?

was used to compare retrieval settings.

With:

TOP_K = 5

the system retrieved an unrelated dice-game passage involving Nala.

Increasing:

TOP_K = 10

provided additional relevant passages about the Pandava dice game and resulted in a substantially more relevant answer.

This demonstrates that retrieval configuration can materially affect downstream LLM responses.

Design Decisions
Why chunk overlap?

A 200-character overlap helps prevent important information from being lost when a sentence or related passage crosses a chunk boundary.

Why Sentence Transformers?

The project uses a pretrained sentence-transformer model to convert both document chunks and user queries into semantic vector representations.

Why FAISS?

FAISS provides a simple and efficient local vector-search implementation without requiring an external vector database.

Why a local LLM?

Using Ollama allows the entire RAG pipeline to run locally without sending the document context to an external LLM API.

Why page metadata?

Each chunk retains its original PDF page number so retrieved context can be traced back to the source document.

Why TOP_K = 10?

Testing showed that retrieving only five chunks could miss relevant passages for ambiguous questions. Increasing the candidate set to ten improved retrieval for the dice-game example.

Limitations

This is a learning and portfolio project rather than a production RAG platform.

Current limitations include:

Retrieval uses a simple FAISS IndexFlatL2 index.
Retrieval quality depends on the embedding model and chunking strategy.
Ambiguous questions can retrieve passages about different events with similar semantic meaning.
The LLM can still make incorrect inferences if the prompt/context is insufficient.
Page citations depend on the LLM correctly associating claims with retrieved evidence.
No reranking model is currently implemented.
No automated evaluation framework is currently implemented.
Future Improvements

Potential improvements include:

Cross-encoder reranking
Hybrid keyword + semantic search
Better chunking based on document structure
Automated retrieval evaluation
Faithfulness and answer-quality metrics
Streaming responses
Web UI
Conversation memory
Metadata filtering
Production vector database
API layer using FastAPI
Learning Objectives

This project was built to understand the practical components of a RAG system:

Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Search
   ↓
Retrieval
   ↓
Prompt Construction
   ↓
LLM
   ↓
Grounded Answer

The project focuses on understanding the underlying RAG architecture rather than building a large production framework.

Author

Kumar Ankit

Data Engineer

GitHub:
https://github.com/kumarankit1612

LinkedIn:
https://www.linkedin.com/in/kumarankit16


### One correction before committing

There's one sentence in the README:

```text
### Create the FAISS Index

After generating the embeddings, create the FAISS index:

```bash
python -m src.vector_store