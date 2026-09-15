def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks

if __name__ == "__main__":
    sample_text = "A" * 2500

    chunks = split_text(
        sample_text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[2])}")
    print(f"Second chunk length: {len(chunks[3])}")