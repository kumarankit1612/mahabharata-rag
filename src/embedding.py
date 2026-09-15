import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PDF_PATH, STORAGE_PATH
from src.ingestion import load_pdf
from src.chunking import split_text


EMBEDDING_MODEL = "all-mpnet-base-v2"


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(model, texts):
    return model.encode(
        texts,
        show_progress_bar=True
    )


def create_chunks(pages):
    chunks = []

    for page in pages:
        page_chunks = split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    return chunks


if __name__ == "__main__":
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading PDF...")
    pages = load_pdf(PDF_PATH)

    print("Creating chunks...")
    chunks = create_chunks(pages)

    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = create_embeddings(
        embedding_model,
        texts
    )

    print(f"Embedding shape: {embeddings.shape}")

    embeddings_path = STORAGE_PATH / "embeddings.npy"
    chunks_path = STORAGE_PATH / "chunks.pkl"

    np.save(embeddings_path, embeddings)

    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    print("\nEmbeddings saved successfully.")
    print(f"Embeddings: {embeddings_path}")
    print(f"Chunks: {chunks_path}")