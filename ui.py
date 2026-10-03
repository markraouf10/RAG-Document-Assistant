import streamlit as st
import os
from rag import ask
from chunking import chunk_pdf
from ingest import ingest_chunks

st.set_page_config(page_title="RAG Document Assistant", page_icon="🔬")
st.title("📄 RAG Document Assistant")
st.caption("Upload any PDF and ask questions. Answers are grounded and cited.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# ── Sidebar — Upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_files:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    # Save to data/ folder
                    save_path = os.path.join("data", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Extract text and chunk
                    all_chunks = chunk_pdf(save_path, source=uploaded_file.name)

                    # Embed and store in Chroma
                    if all_chunks:
                        ingest_chunks(all_chunks)
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} ingested ({len(all_chunks)} chunks)")

    if st.session_state.ingested_files:
        st.markdown("---")
        st.markdown("**📚 Loaded Papers:**")
        for f in st.session_state.ingested_files:
            st.markdown(f"- {f}")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about your papers..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer, chunks = ask(query)

        st.markdown(answer)

        with st.expander("📄 Sources"):
            seen = set()
            for chunk in chunks:
                key = f"{chunk['source']} — page {chunk['page']}"
                if key not in seen:
                    st.markdown(f"- {key}")
                    seen.add(key)

    st.session_state.messages.append({"role": "assistant", "content": answer})