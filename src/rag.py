from src.embedding import load_embedding_model
from src.retriever import load_vector_store, retrieve
from src.prompt import build_prompt
from src.llm import generate_answer


def ask_question(query, top_k=5):
    # 1. Load embedding model
    model = load_embedding_model()

    # 2. Load FAISS index and chunks
    index, chunks = load_vector_store()

    # 3. Retrieve relevant chunks
    results = retrieve(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    # 4. Build prompt using retrieved context
    prompt = build_prompt(
        query=query,
        retrieved_results=results
    )

    # 5. Generate answer using LLM
    answer = generate_answer(prompt)

    return answer, results


if __name__ == "__main__":

    query = "Why did Karna support Duryodhana?"

    answer, results = ask_question(query)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for result in results:
        print(f"Page {result['page_number']}")