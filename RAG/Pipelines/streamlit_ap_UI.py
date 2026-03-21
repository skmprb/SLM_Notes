import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("🔍 RAG Pipeline")

# Sidebar - Ingest
with st.sidebar:
    st.header("📥 Ingest Data")
    source = st.text_input("Source (URL or file path)")
    source_type = st.selectbox("Type", ["url", "pdf", "text"])
    if st.button("Ingest"):
        res = requests.post(f"{API_URL}/ingest", json={"source": source, "source_type": source_type})
        st.success(f"Ingested! {res.json()['chunks_stored']} chunks stored")
    
    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K", 1, 10, 5)
    prompt_style = st.selectbox("Prompt Style", ["simple", "sources", "strict", "conversational"])
    llm_provider = st.selectbox("LLM Provider", ["openai", "bedrock", "ollama", "huggingface"])

# Main - Query
question = st.text_input("Ask a question:")
if st.button("Search") and question:
    with st.spinner("Thinking..."):
        res = requests.post(f"{API_URL}/query", json={
            "question": question,
            "top_k": top_k,
            "prompt_style": prompt_style,
            "llm_provider": llm_provider
        }).json()
    
    st.subheader("Answer")
    st.write(res.get("answer", ""))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{res.get('latency', 0):.2f}s")
    col2.metric("Relevance", f"{res.get('evaluation', {}).get('relevance', 0):.2f}")
    col3.metric("Groundedness", f"{res.get('evaluation', {}).get('groundedness', 0):.2f}")
    
    with st.expander("Sources"):
        for s in res.get("sources", []):
            st.write(f"- {s}")

# Stats
if st.sidebar.button("📊 Show Stats"):
    stats = requests.get(f"{API_URL}/stats").json()
    st.sidebar.json(stats)
