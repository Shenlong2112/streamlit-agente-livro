# app.py — LangChain-first + Drive/FAISS (Drive liga na página Conexões)
import streamlit as st
from typing import Optional
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Agente de Livro", page_icon="📚", layout="wide")

def set_session_value(key: str, value: Optional[str]):
    if value and value.strip():
        st.session_state[key] = value.strip()
    else:
        st.session_state.pop(key, None)

def has_session_value(key: str) -> bool:
    return bool(st.session_state.get(key))

def test_openai_via_langchain(api_key: str) -> tuple[bool, str]:
    try:
        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.0, max_tokens=5)
        _ = llm.invoke("ping").content
        return True, "Conexão OK (key válida via LangChain)."
    except Exception as e:
        return False, f"Falha ao conectar: {e}"

with st.sidebar:
    st.markdown("## Conexões")
    st.caption("A OpenAI API key fica **somente nesta sessão** (RAM).")
    openai_key_input = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("user_openai_key", ""))
    set_session_value("user_openai_key", openai_key_input)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Testar (LangChain)", use_container_width=True):
            if not has_session_value("user_openai_key"):
                st.error("Cole sua OpenAI API key primeiro.")
            else:
                ok, msg = test_openai_via_langchain(st.session_state["user_openai_key"])
                st.success("✅ " + msg) if ok else st.error("❌ " + msg)
    with c2:
        if st.button("Limpar chave", use_container_width=True):
            set_session_value("user_openai_key", None)
            st.success("Chave removida da sessão.")

st.title("📚 Agente/Editor para Livro — (Drive + FAISS)")
if has_session_value("user_openai_key"):
    st.info("🔐 OpenAI key ativa nesta sessão.")
else:
    st.warning("Cole sua OpenAI key na **sidebar**.")

st.markdown("""
Use as páginas no menu:
- **🔌 Conexões**: conectar **Google Drive** (OAuth). Nenhum arquivo fica no seu PC.
- **📝 Editor de Livro**: listar **transcrições (.md) no seu Drive**, gerar **versões** e fixar **FINAL**.
- **💬 Chatbot Knowledge**: chat que vai usar **somente** seu *knowledge* (quando ligarmos o RAG/FAISS).
""")

