import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from embeddings import embed_query

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

chroma_client = chromadb.PersistentClient(path="chroma_db/")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── Load system prompt ────────────────────────────────────────────────────────
with open("system_prompt.md", "r") as f:
    SYSTEM_PROMPT = f.read()

# ── Retrieve top-k chunks from Chroma ────────────────────────────────────────
def retrieve(query, k=8):
    query_vector = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"]
        })
    return chunks

# ── Build prompt with retrieved chunks ───────────────────────────────────────
def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[{i+1}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}\n"

    return f"""Answer the question based only on the context below.

Question: {query}

Context:
{context}
"""

# ── Generate answer ───────────────────────────────────────────────────────────
def generate(query, chunks):
    prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ── Full pipeline ─────────────────────────────────────────────────────────────
def ask(query):
    chunks = retrieve(query)
    answer = generate(query, chunks)
    return answer, chunks

# ── Run as CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RAG Document Assistant ready. Type 'quit' to exit.\n")
    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        answer, chunks = ask(query)
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for chunk in chunks:
            print(f"  - {chunk['source']}, page {chunk['page']}")
        print()