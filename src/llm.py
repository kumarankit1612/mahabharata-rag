from ollama import chat


MODEL_NAME = "llama3.2:3b"


def generate_answer(prompt):
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    test_prompt = "Explain what a data pipeline is in one sentence."

    answer = generate_answer(test_prompt)

    print("\n--- LLM Response ---")
    print(answer)