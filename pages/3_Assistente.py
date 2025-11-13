# pages/3_Assistente.py
from __future__ import annotations

import os
import io
import zipfile
import tempfile
from typing import List, Optional, TYPE_CHECKING, Any

import streamlit as st

from src.storage.drive import (
    drive_service_from_token,
    list_files_md,
    download_binary,
)
from src.knowledge.repo import ensure_user_tree, VECSTORE_DIR

# Embeddings / FAISS
from src.embeddings.vectorstore_faiss import (
    load_faiss_index,
    create_faiss_index,
)

# ===== LLM (BYOK) =====
# Em runtime usamos _ChatOpenAI (pode ser None se a lib não estiver instalada).
# Para type hints, usamos ChatOpenAI apenas dentro de TYPE_CHECKING, evitando o alerta do Pylance.
try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI  # runtime
except Exception:
    _ChatOpenAI = None  # type: ignore

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI  # só para type hints
else:
    ChatOpenAI = Any  # type: ignore[misc,assignment]

# ===== Pesquisa web opcional (DuckDuckGo) =====
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None  # type: ignore


# ==========================
# Helpers
# ==========================
def _unzip_bytes_to_tempdir(data: bytes) -> str:
    td = tempfile.mkdtemp(prefix="faiss_")
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(td)
    return td


def _load_global_index_from_drive(service) -> Optional[object]:
    """
    Baixa todos os pacotes *.faiss.zip da pasta Vecstore e mescla num único índice.
    Retorna o VectorStore FAISS do LangChain ou None se não houver pacotes.
    """
    ids = ensure_user_tree(service)
    vec_id = ids["vec"]

    zips = [f for f in list_files_md(service, vec_id, extensions=[".zip"]) if f["name"].lower().endswith(".faiss.zip")]
    if not zips:
        return None

    base_store = None
    for fmeta in zips:
        try:
            data = download_binary(service, fmeta["id"])
            td = _unzip_bytes_to_tempdir(data)
            store = load_faiss_index(td)  # carrega FAISS salvo localmente
            if base_store is None:
                base_store = store
            else:
                try:
                    base_store.merge_from(store)
                except Exception:
                    # Se embeddings divergirem, ignora esse pacote
                    pass
        except Exception:
            continue
    return base_store


def _search_web_duckduckgo(q: str, k: int = 3) -> str:
    if DDGS is None:
        return "⚠️ Módulo duckduckgo_search não está instalado no servidor."
    out = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(q, max_results=k)):
            title = r.get("title") or ""
            href = r.get("href") or ""
            body = r.get("body") or ""
            out.append(f"{i+1}. {title}\n{href}\n{body}\n")
    return "\n".join(out) if out else "Nenhum resultado encontrado."


def _get_llm() -> Optional[ChatOpenAI]:
    """
    Instancia o LLM para streaming.
    Retorna None se não houver OPENAI_API_KEY ou se langchain_openai não estiver instalado.
    """
    if not st.session_state.get("OPENAI_API_KEY"):
        return None
    if _ChatOpenAI is None:
        return None
    # Garanta que libs usem sua chave BYOK
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
    return _ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True, openai_api_key=st.session_state["OPENAI_API_KEY"])


def _rag_answer(llm: ChatOpenAI, question: str, store, k: int) -> str:
    """
    Faz RAG simples: recupera k trechos do acervo FAISS e pede ao modelo para responder
    priorizando essas fontes. Se nada encontrado, responde normalmente.
    """
    context_blocks: List[str] = []
    if store is not None and k > 0:
        try:
            docs = store.similarity_search(question, k=k)
            for d in docs:
                context_blocks.append(d.page_content)
        except Exception:
            pass

    sys = (
        "Você é um assistente que deve **priorizar o acervo** do usuário.\n"
        "Responda de forma direta e útil. Se o acervo não contiver a resposta e o usuário "
        "não pedir web explicitamente, diga que não encontrou no acervo e peça mais contexto.\n"
        "Nunca invente fatos. Se usar trechos, sintetize, não cole bruto.\n"
    )
    ctx = "\n\n".join(context_blocks) if context_blocks else "NENHUM TRECHO ENCONTRADO"
    usr = f"PERGUNTA:\n{question}\n\nACERVO (top-{k}):\n{ctx}"

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]

    # Streaming manual
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        for chunk in llm.stream(messages):
            acc += (chunk.content or "")
            placeholder.markdown(acc)
        return acc


# ==========================
# Página
# ==========================
st.set_page_config(page_title="Assistente", page_icon="🤖", layout="wide")
st.title("🤖 Assistente")

# Requisitos
if not st.session_state.get("OPENAI_API_KEY"):
    st.warning("Cole sua **OPENAI_API_KEY** em **Conexões** para usar o Assistente.")
    st.stop()

if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte o **Google Drive** em **Conexões** para usar o acervo.")
    st.stop()

# LLM
llm = _get_llm()
if llm is None:
    st.error("Não foi possível iniciar o modelo (verifique a instalação de `langchain-openai`).")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])

# Estado do chat
st.session_state.setdefault("messages", [])
st.session_state.setdefault("faiss_loaded", False)
st.session_state.setdefault("faiss_store", None)

# Painel lateral de controle
with st.sidebar:
    st.header("🔧 Controles")
    k_acervo = st.slider("Trechos do acervo (top-k)", 1, 12, 8, 1)
    do_web = st.toggle(
        "Permitir pesquisa web quando a pergunta começar com **web:**",
        value=True,
        help="Se habilitado, mensagens que começarem com 'web:' acionam busca DuckDuckGo.",
    )
    if st.button("Recarregar índice global (Vecstore)"):
        with st.spinner("Carregando índice global a partir do Drive..."):
            store = _load_global_index_from_drive(service)
            st.session_state["faiss_store"] = store
            st.session_state["faiss_loaded"] = store is not None
            if store is None:
                st.warning(f"Nenhum pacote **.faiss.zip** encontrado em **{VECSTORE_DIR}**.")
            else:
                st.success("Índice global recarregado.")

# Carrega automaticamente a primeira vez
if not st.session_state["faiss_loaded"]:
    with st.spinner("Carregando índice global a partir do Drive..."):
        store = _load_global_index_from_drive(service)
        st.session_state["faiss_store"] = store
        st.session_state["faiss_loaded"] = store is not None
        if store is None:
            st.info(
                f"A pasta **{VECSTORE_DIR}** ainda não tem pacotes de embeddings (.faiss.zip). "
                "Salve versões no **Editor** para populá-la."
            )

# UI de chat
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pergunte algo. Para web, comece com: web: sua busca")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Caminho 1: pesquisa web explícita
    if do_web and prompt.strip().lower().startswith("web:"):
        query = prompt.split(":", 1)[1].strip() or prompt
        if DDGS is None:
            with st.chat_message("assistant"):
                st.markdown("⚠️ Para usar a busca web, instale `duckduckgo_search` no servidor.")
        else:
            with st.spinner("Buscando na web (DuckDuckGo)..."):
                web_text = _search_web_duckduckgo(query, k=5)
            # Usa LLM para sintetizar os resultados web
            sys = "Você é um assistente que sintetiza resultados de busca em respostas claras e objetivas."
            usr = f"Consulta: {query}\n\nResultados:\n{web_text}\n\nResuma e responda de forma útil."
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""
                for chunk in llm.stream(messages):
                    acc += (chunk.content or "")
                    placeholder.markdown(acc)
                st.session_state["messages"].append({"role": "assistant", "content": acc})

    # Caminho 2: RAG com acervo (prioridade máxima)
    else:
        answer = _rag_answer(llm, prompt, st.session_state.get("faiss_store"), k_acervo)
        st.session_state["messages"].append({"role": "assistant", "content": answer})



