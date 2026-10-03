import os
from pypdf import PdfReader


def chunk_text(text, chunk_size=800, overlap=300):
    separators = ["\n\n", "\n", ". ", " "]

    def split_text(text, separator_index):
        if separator_index >= len(separators):
            return [text]
        separator = separators[separator_index]
        splits = text.split(separator)
        chunks = []
        for split in splits:
            if len(split) > chunk_size:
                smaller = split_text(split, separator_index + 1)
                chunks.extend(smaller)
            else:
                if split.strip():
                    chunks.append(split.strip())
        return chunks

    chunks = split_text(text, 0)

    # add overlap by prepending tail of previous chunk to next chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


def chunk_pdf(path, source=None):
    """Chunk a single PDF file and return a list of
    {"text", "source", "page", "chunk_id"} dicts."""
    source = source or os.path.basename(path)
    reader = PdfReader(path)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            for chunk_idx, chunk in enumerate(chunk_text(text)):
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"{source}_p{page_num}_c{chunk_idx}",
                })
    return chunks


def chunk_documents(data_dir="data/"):
    """Read every PDF in data_dir, chunk it, and return a list of
    {"text", "source", "page", "chunk_id"} dicts. No side effects at import time."""
    all_chunks = []

    files = sorted(os.listdir(data_dir))

    for filename in files:
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            all_chunks.extend(chunk_pdf(path, source=filename))

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_documents()
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\nSample chunk:")
        print(chunks[0])
        print("\nSample chunk 2 (check overlap with chunk 1):")
        print(chunks[1])