from src.retriever import load_vector_store, retrieve
from src.embedding import load_embedding_model


def build_prompt(query, retrieved_results):
    context_parts = []

    for result in retrieved_results:
        context_parts.append(
            f"[Page {result['page_number']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a question-answering assistant for the provided Mahabharata document.

    Answer the user's question using only the information provided in the context below.

    If the context does not contain enough information to answer the question,
    clearly state that the answer cannot be determined from the provided context.

    Do not invent or assume information that is not present in the context.

    For every factual statement in your answer, cite the page containing
    the evidence for that statement using [Page X].

    Do not use a page citation merely because the page is present in the context.

    Do not make inferences about a character's motives, intentions, feelings,
    relationships, or reasons unless the context explicitly supports that inference.

    If the context contains evidence for only part of the question, answer only
    that supported part and clearly state what cannot be determined.

    Keep the answer concise and evidence-based.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """

    return prompt


if __name__ == "__main__":

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Loading vector store...")
    index, chunks = load_vector_store()

    query = "Why did Karna support Duryodhana?"

    print("\nRetrieving relevant chunks...")

    results = retrieve(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5
    )

    prompt = build_prompt(query, results)

    print("\n" + "=" * 80)
    print("FINAL PROMPT")
    print("=" * 80)

    print(prompt)