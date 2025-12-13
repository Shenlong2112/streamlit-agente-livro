# src/embeddings/vectorstore_faiss.py
from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Use sempre o mesmo modelo para evitar conflitos de dimensão ao mesclar índices
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

# Conjunto de caracteres invisíveis comuns em copy/paste
_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

def _clean_ascii(s: str) -> str:
    """Remove quebras, espaços invisíveis e qualquer caractere não-ASCII."""
    if not isinstance(s, str):
        s = str(s or "")
    s = s.replace("\r", "").replace("\n", "").strip()
    for ch in _ZERO_WIDTH:
        s = s.replace(ch, "")
    return s.encode("ascii", "ignore").decode("ascii")

def _ensure_api_key() -> str:
    """
    Garante que OPENAI_API_KEY está disponível e sanitizada.
    Também sanitiza OPENAI_ORG_ID/OPENAI_PROJECT se existirem.
    """
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    key = _clean_ascii(key or "")
    if not key.startswith("sk-") or len(key) < 20:
        raise RuntimeError(
            "OPENAI_API_KEY ausente ou inválida. Vá em **Conexões** e cole sua chave (começa com `sk-`)."
        )
    os.environ["OPENAI_API_KEY"] = key

    # Se estiverem definidos, higieniza também (evita headers com não-ASCII)
    for var in ("OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"):
        val = os.getenv(var) or st.secrets.get(var, None)
        if val:
            os.environ[var] = _clean_ascii(str(val))

    return key

def create_faiss_index(texts: List[str], metadata: Optional[List[dict]] = None) -> FAISS:
    """Cria um índice FAISS a partir de textos (com split e metadados opcionais)."""
    key = _ensure_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=key)

    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    all_chunks: List[str] = []
    all_metas: List[dict] = []

    for i, t in enumerate(texts or []):
        chunks = splitter.split_text(t or "")
        all_chunks.extend(chunks)
        if metadata and i < len(metadata):
            all_metas.extend([metadata[i]] * len(chunks))
        else:
            all_metas.extend([{}] * len(chunks))

    if not all_chunks:
        # Evita índice vazio, que pode quebrar alguns ambientes
        all_chunks = [" "]
        all_metas = [{}]

    try:
        return FAISS.from_texts(all_chunks, embeddings, metadatas=all_metas)
    except Exception as e:
        # Erros típicos aqui: headers com não-ASCII (UnicodeEncodeError via httpx),
        # chave inválida, quota, etc.
        raise RuntimeError(
            "Falha ao gerar embeddings. Verifique se sua OPENAI_API_KEY está correta, sem "
            "caracteres invisíveis, e se sua conta tem acesso ao modelo de embeddings. "
            f"Detalhe técnico: {type(e).__name__}: {e}"
        ) from e

def save_faiss_index(index: FAISS, path: str):
    index.save_local(path)

def load_faiss_index(path: str) -> FAISS:
    """Carrega um índice FAISS salvo em disco (precisa de embeddings para buscas)."""
    key = _ensure_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=key)
    return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)

