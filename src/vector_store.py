import numpy as np
import faiss

from src.config import STORAGE_PATH


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index


if __name__ == "__main__":
    embeddings_path = STORAGE_PATH / "embeddings.npy"

    embeddings = np.load(embeddings_path)

    print(f"Loaded embeddings: {embeddings.shape}")

    index = create_faiss_index(embeddings)

    print(f"Total vectors in FAISS: {index.ntotal}")

    index_path = STORAGE_PATH / "index.faiss"

    faiss.write_index(index, str(index_path))

    print(f"FAISS index saved to: {index_path}")