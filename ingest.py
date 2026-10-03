import chromadb
from dotenv import load_dotenv
from chunking import chunk_documents
from embeddings import embed_documents

load_dotenv()

# Initialize Chroma
chroma_client = chromadb.PersistentClient(path="chroma_db/")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def ingest_chunks(chunks):
    """Embed a list of {text, source, page, chunk_id} chunks and upsert them into Chroma."""
    if not chunks:
        return
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_documents(texts)
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]
    )


def ingest():
    # Step 1: chunk all PDFs
    print("Chunking documents...")
    chunks = chunk_documents()
    print(f"Total chunks: {len(chunks)}")

    # Step 2 & 3: embed and store in Chroma
    print("Embedding chunks...")
    print("Storing in Chroma...")
    ingest_chunks(chunks)

    print(f"Done! {len(chunks)} chunks stored in Chroma.")

if __name__ == "__main__":
    ingest()