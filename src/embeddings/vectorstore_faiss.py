# src/embeddings/vectorstore_faiss.py
from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Defina um modelo e mantenha SEMPRE o mesmo para evitar conflito de dimensões ao mesclar índices
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

def _ensure_api_key() -> str:
    """Garante que a OPENAI_API_KEY esteja disponível neste processo."""
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY ausente. Vá em **Conexões** e cole sua chave.")
    os.environ["OPENAI_API_KEY"] = key  # garante disponibilidade para libs internas
    return key

def create_faiss_index(texts: List[str], metadata: Optional[List[dict]] = None) -> FAISS:
    """Cria um índice FAISS a partir de uma lista de textos (com split)."""
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
        # evita criar índice vazio (que pode quebrar em alguns ambientes)
        all_chunks = [" "]
        all_metas = [{}]

    return FAISS.from_texts(all_chunks, embeddings, metadatas=all_metas)

def save_faiss_index(index: FAISS, path: str):
    index.save_local(path)

def load_faiss_index(path: str) -> FAISS:
    """Carrega um índice FAISS salvo em disco (precisa de embeddings para buscas)."""
    key = _ensure_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=key)
    # allow_dangerous_deserialization é necessário em alguns ambientes/versões
    return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)

