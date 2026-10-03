import os
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def embed_documents(texts: list[str], batch_size: int = 90) -> list[list[float]]:
    """Embed a list of document chunks for storage/indexing."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        text_inputs = [{"content": [{"type": "text", "text": t}]} for t in batch]
        response = co.embed(
            inputs=text_inputs,
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        all_vectors.extend(response.embeddings.float_)
        print(f"   Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
    return all_vectors


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    text_input = [{"content": [{"type": "text", "text": text}]}]
    response = co.embed(
        inputs=text_input,
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]
