from src.embedding import load_embedding_model
from src.retriever import load_vector_store, retrieve
from src.prompt import build_prompt
from src.llm import generate_answer


def ask_question(query, model, index, chunks, top_k=10):
    # 1. Retrieve relevant chunks
    results = retrieve(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    # 2. Build prompt using retrieved context
    prompt = build_prompt(
        query=query,
        retrieved_results=results
    )

    # 3. Generate answer using LLM
    answer = generate_answer(prompt)

    return answer, results


if __name__ == "__main__":

    print("=" * 80)
    print("MAHABHARATA RAG")
    print("=" * 80)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("Loading FAISS index and chunks...")
    index, chunks = load_vector_store()

    print("\nReady! Ask questions about the Mahabharata.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nExiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        answer, results = ask_question(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=10
        )

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)

        source_pages = sorted(
            set(result["page_number"] for result in results)
        )

        for page in source_pages:
            print(f"Page {page}")