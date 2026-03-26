import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_vectorstore, get_answer
import os

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0f0f13; color: #e8e6df; }

    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #f0ec9a;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .chat-user {
        background: #1e1e28;
        border-left: 3px solid #f0ec9a;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #e8e6df;
    }
    .chat-bot {
        background: #161620;
        border-left: 3px solid #7eb8f7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #c8d8ea;
    }
    .source-box {
        background: #1a1a24;
        border: 1px solid #2e2e40;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #888;
        font-family: 'Space Mono', monospace;
    }
    .status-ok { color: #7eb87e; font-weight: 500; }
    .status-err { color: #e07e7e; font-weight: 500; }
    div[data-testid="stSidebar"] { background: #0a0a0f; border-right: 1px solid #1e1e2e; }
    .stTextInput > div > div > input {
        background: #1a1a24 !important;
        color: #e8e6df !important;
        border: 1px solid #2e2e40 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton > button {
        background: #f0ec9a !important;
        color: #0f0f13 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover { background: #e8e070 !important; }
    .stFileUploader { background: #1a1a24; border-radius: 10px; }
    .step-tag {
        display: inline-block;
        background: #f0ec9a22;
        color: #f0ec9a;
        border: 1px solid #f0ec9a44;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-header">📄 PDF Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask anything about your document</div>', unsafe_allow_html=True)

    st.markdown('<span class="step-tag">STEP 1</span>', unsafe_allow_html=True)
    st.markdown("**Configure API**")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="AIza...",
        help="Get free key at https://console.groq.com/keys"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.markdown('<span class="status-ok">✓ API key set</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<span class="step-tag">STEP 2</span>', unsafe_allow_html=True)
    st.markdown("**Upload your PDF**")

    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and api_key:
        if st.button("⚡ Process PDF"):
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    vectorstore = build_vectorstore(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✓ Ready! Ask away.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    elif uploaded_file and not api_key:
        st.warning("Enter your API key first.")

    if st.session_state.pdf_name:
        st.divider()
        st.markdown(f"**Active doc:** `{st.session_state.pdf_name}`")
        if st.button("🗑 Clear & reset"):
            st.session_state.vectorstore = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; color:#555; font-family:'Space Mono',monospace; line-height:1.8">
    Stack<br>
    ─ LangChain<br>
    ─ HuggingFace embeddings<br>
    ─ FAISS vector store<br>
    ─ Google Gemini LLM<br>
    ─ Streamlit UI
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;gap:12px;text-align:center">
        <div style="font-size:3.5rem">📄</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#f0ec9a">Upload a PDF to begin</div>
        <div style="color:#555;font-size:0.9rem;max-width:380px">
            Enter your Gemini API key and upload any PDF in the sidebar.<br>Then ask questions — with source citations.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source passages", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f'<div class="source-box">📍 Source {i} (page {src.get("page","?")})<br>{src["text"][:300]}...</div>', unsafe_allow_html=True)

    # Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("", placeholder="Ask a question about your PDF...", label_visibility="collapsed", key="question_input")
    with col2:
        ask_btn = st.button("Ask →")

    if ask_btn and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                result = get_answer(st.session_state.vectorstore, question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                    "sources": []
                })
        st.rerun()
