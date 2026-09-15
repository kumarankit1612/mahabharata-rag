import pickle

import numpy as np

from src.config import STORAGE_PATH


if __name__ == "__main__":
    embeddings_path = STORAGE_PATH / "embeddings.npy"
    chunks_path = STORAGE_PATH / "chunks.pkl"

    embeddings = np.load(embeddings_path)

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of chunks: {len(chunks)}")

    print("\n--- Sample Chunk ---")
    print(f"Page: {chunks[0]['page_number']}")
    print(f"Text: {chunks[0]['text'][:300]}")