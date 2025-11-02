# src/embeddings/vectorstore_faiss.py
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def create_faiss_index(texts: List[str], metadata: Optional[List[dict]] = None) -> FAISS:
    """
    Gera um índice FAISS a partir dos textos fornecidos.
    """
    embeddings = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    # Divide cada texto individualmente para preservar fronteiras e evitar juntar tudo
    split_texts: List[str] = []
    for t in texts:
        split_texts.extend(splitter.split_text(t))

    # Cria o índice (se tiver metadados alinhados aos chunks, passe em metadatas=...)
    store = FAISS.from_texts(split_texts, embedding=embeddings)
    return store

def save_faiss_index(index: FAISS, path: str) -> None:
    """Salva o índice FAISS em um diretório local"""
    index.save_local(path)

def load_faiss_index(path: str) -> FAISS:
    """Carrega um índice FAISS de um diretório local"""
    return FAISS.load_local(path, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)

