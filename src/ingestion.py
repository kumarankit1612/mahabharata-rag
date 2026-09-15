import fitz
from src.chunking import split_text
from src.config import PDF_PATH


def load_pdf(path):
    document = fitz.open(path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    print(f"Total pages loaded: {len(pages)}")

    return pages


if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)

    chunks = []

    for page in pages:
        page_chunks = split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    print("\n--- First Chunk ---\n")
    print(chunks[0]["text"])

    print(f"\nSource page: {chunks[0]['page_number']}")
    print(f"Chunk length: {len(chunks[0]['text'])} characters")