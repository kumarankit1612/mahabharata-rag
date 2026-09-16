import pickle

import faiss

from src.config import STORAGE_PATH,TOP_K
from src.embedding import load_embedding_model


def load_vector_store():
    index_path = STORAGE_PATH / "index.faiss"
    chunks_path = STORAGE_PATH / "chunks.pkl"

    index = faiss.read_index(str(index_path))

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def retrieve(query, model, index, chunks, top_k=TOP_K):
    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for distance, index_position in zip(distances[0], indices[0]):
        results.append({
            "page_number": chunks[index_position]["page_number"],
            "text": chunks[index_position]["text"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":
    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading FAISS index and chunks...")
    index, chunks = load_vector_store()

    query = "Why did Karna support Duryodhana?"

    print(f"\nQuery: {query}")

    results = retrieve(
        query,
        embedding_model,
        index,
        chunks,
        top_k=5
    )

    print("\n--- Retrieved Results ---\n")

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Page: {result['page_number']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 80)