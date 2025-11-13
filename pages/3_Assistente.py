# pages/3_Assistente.py
from __future__ import annotations

import os
import io
import time
import json
import zipfile
import tempfile
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING, Any

import streamlit as st
import streamlit.components.v1 as components

from src.storage.drive import (
    drive_service_from_token,
    list_files_md,
    list_files_in_folder,
    download_binary,
    download_text,
    upload_text,
    upload_binary,
    update_file_contents,
    find_or_create_folder,
    ensure_subfolder,
)
from src.knowledge.repo import ensure_user_tree, VECSTORE_DIR

# Embeddings / FAISS
from src.embeddings.vectorstore_faiss import (
    load_faiss_index,
    create_faiss_index,
    save_faiss_index,
)

# ===== LLM (BYOK) =====
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


CHAT_DIR = "Chats"  # pasta dedicada para memória de chat no Drive


# ==========================
# Helpers de Drive / FAISS
# ==========================
def _unzip_bytes_to_tempdir(data: bytes) -> str:
    td = tempfile.mkdtemp(prefix="faiss_")
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(td)
    return td


def _zip_dir_to_bytes(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path)
                zf.write(full, rel)
    return buf.getvalue()


def _load_global_index_from_drive(service) -> Optional[object]:
    """
    Baixa todos os pacotes *.faiss.zip da pasta Vecstore e mescla num único índice.
    Inclui os pacotes gerados pelo Editor e também os pacotes de chat.
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


# ==========================
# Memória de Chat (Drive)
# ==========================
def _ensure_chat_folder(service, root_id: str) -> str:
    """Garante a pasta 'Chats' dentro do root do usuário."""
    return ensure_subfolder(service, root_id, CHAT_DIR)


def _new_chat_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{ts}.txt"


def _append_to_chat_file(service, chat_file_id: str, role: str, text: str) -> None:
    """Acrescenta uma linha ao arquivo de chat no Drive (timestamp + role + texto)."""
    try:
        current = download_text(service, chat_file_id)
    except Exception:
        current = ""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"[{stamp}] {role.upper()}:\n{text.rstrip()}\n\n"
    new_bytes = (current + block).encode("utf-8")
    update_file_contents(service, chat_file_id, new_bytes, mimetype="text/plain")


def _load_last_chat_file(service, chats_folder_id: str) -> Optional[dict]:
    """Retorna metadados do último arquivo de chat (mais recente) ou None."""
    files = list_files_md(service, chats_folder_id, extensions=[".txt"])
    return files[0] if files else None


def _download_chat_text(service, chat_file_id: str) -> str:
    try:
        return download_text(service, chat_file_id)
    except Exception:
        return ""


def _update_chat_embeddings(service, vec_folder_id: str, chat_file_id: str, chat_text: str) -> None:
    """
    Gera embeddings FAISS do texto completo do chat e salva/atualiza um pacote
    'chat-<fileid>.faiss.zip' no Vecstore.
    """
    if not chat_text.strip():
        return
    # cria índice a partir do histórico inteiro (robusto contra falta de state)
    index = create_faiss_index([chat_text])
    with tempfile.TemporaryDirectory() as td:
        save_faiss_index(index, td)
        data = _zip_dir_to_bytes(td)

    pkg_name = f"chat-{chat_file_id}.faiss.zip"
    # se já existir, atualiza; senão, cria
    existing = list_files_in_folder(service, vec_folder_id, name_equals=pkg_name)
    if existing:
        update_file_contents(service, existing[0]["id"], data, mimetype="application/zip")
    else:
        upload_binary(service, vec_folder_id, pkg_name, data, mimetype="application/zip")


# ==========================
# LLM e Web
# ==========================
def _search_web_duckduckgo(q: str, k: int = 3) -> str:
    """
    Busca robusta no DuckDuckGo (api/html/lite + retries).
    """
    if DDGS is None:
        return "⚠️ Módulo `duckduckgo_search` não está instalado no servidor."

    backends = ("api", "html", "lite")
    last_err = None

    for backend in backends:
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=k, backend=backend))
                if results:
                    out = []
                    for i, r in enumerate(results, start=1):
                        title = (r.get("title") or "").strip()
                        href = (r.get("href") or "").strip()
                        body = (r.get("body") or "").strip()
                        out.append(f"{i}. {title}\n{href}\n{body}\n")
                    return "\n".join(out)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue

    if last_err:
        return f"⚠️ Falha na busca web (rate limit/erro do DDG). Tente novamente em instantes.\nDetalhe: {type(last_err).__name__}"
    return "Nenhum resultado encontrado."


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


def _copy_button(text: str, key: str):
    """
    Botão 'Copiar' (client-side) usando clipboard do navegador.
    """
    safe = json.dumps(text)  # escapa para JS
    components.html(
        f"""
        <button
          onclick="navigator.clipboard.writeText({safe}); this.innerText='Copiado!'; setTimeout(()=>this.innerText='Copiar', 1300);"
          style="margin-top:6px;padding:6px 10px;border-radius:8px;border:1px solid #555;background:#111;color:#eee;cursor:pointer;"
        >Copiar</button>
        """,
        height=40,
    )


def _rag_answer(llm: ChatOpenAI, question: str, store, k: int, copy_key: str) -> str:
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
        _copy_button(acc, key=f"copy_{copy_key}")
        return acc


def _summarize_chat_if_any(llm: Optional[ChatOpenAI], text: str) -> str:
    """Gera um resumo curto do histórico, se houver LLM."""
    if not text.strip() or llm is None:
        return ""
    sys = "Resuma a conversa a seguir em no máximo 6 bullets claros e específicos. Não invente conteúdo."
    usr = text[-8000:]  # limite de segurança
    out = llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    return getattr(out, "content", "") or ""


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
    st.warning("Conecte o **Google Drive** em **Conexões** para usar o acervo e a memória.")
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
st.session_state.setdefault("chat_file_id", None)       # arquivo .txt no Drive para esta conversa
st.session_state.setdefault("chat_folder_id", None)     # pasta Chats
st.session_state.setdefault("chat_loaded_once", False)  # já carregou/retomou histórico nesta sessão?

# Painel lateral de controle
with st.sidebar:
    st.header("🔧 Controles")
    k_acervo = st.slider("Trechos do acervo (top-k)", 1, 12, 8, 1)
    do_web = st.toggle(
        "Permitir pesquisa web quando a pergunta começar com **web:**",
        value=True,
        help="Se habilitado, mensagens que começarem com 'web:' acionam busca DuckDuckGo.",
    )
    # Índice global (inclui editor e chat)
    if st.button("Recarregar índice global (Vecstore)"):
        with st.spinner("Carregando índice global a partir do Drive..."):
            store = _load_global_index_from_drive(service)
            st.session_state["faiss_store"] = store
            st.session_state["faiss_loaded"] = store is not None
            if store is None:
                st.warning(f"Nenhum pacote **.faiss.zip** encontrado em **{VECSTORE_DIR}**.")
            else:
                st.success("Índice global recarregado.")

    # Novo chat (começar do zero)
    if st.button("🆕 Iniciar novo chat"):
        # cria novo arquivo na pasta Chats e zera a conversa da sessão
        ids = ensure_user_tree(service)
        root_id = ids["root"]
        chats_id = _ensure_chat_folder(service, root_id)
        st.session_state["chat_folder_id"] = chats_id
        fname = _new_chat_filename()
        new_id = upload_text(service, chats_id, fname, f"Início do chat: {datetime.now().isoformat()}\n\n")
        st.session_state["chat_file_id"] = new_id
        st.session_state["messages"] = []
        st.session_state["chat_loaded_once"] = True
        st.success("Novo chat iniciado.")

# Garante árvore e carrega índice global (1ª vez)
ids = ensure_user_tree(service)
root_id = ids["root"]
vec_id = ids["vec"]

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

# Garante pasta de chats e tenta retomar último histórico (só 1ª vez na sessão)
if not st.session_state["chat_loaded_once"]:
    chats_id = _ensure_chat_folder(service, root_id)
    st.session_state["chat_folder_id"] = chats_id

    last = _load_last_chat_file(service, chats_id)
    if last:
        # Retoma: usa o último arquivo como chat_file atual e mostra um resumo
        st.session_state["chat_file_id"] = last["id"]
        chat_text = _download_chat_text(service, last["id"])
        summary = _summarize_chat_if_any(llm, chat_text)
        if summary:
            st.session_state["messages"].append({"role": "assistant", "content": f"**Resumo da sua última conversa:**\n\n{summary}"})
            st.toast("Última conversa retomada (resumo).", icon="🧠")
    else:
        # Se não houver, cria um novo arquivo de chat
        fname = _new_chat_filename()
        new_id = upload_text(service, chats_id, fname, f"Início do chat: {datetime.now().isoformat()}\n\n")
        st.session_state["chat_file_id"] = new_id

    st.session_state["chat_loaded_once"] = True

# UI de chat (renderiza histórico da sessão)
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pergunte algo. Para web, comece com: web: sua busca")
if prompt:
    # 1) Renderiza prompt
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # 1a) Grava no arquivo de chat
    if st.session_state.get("chat_file_id"):
        _append_to_chat_file(service, st.session_state["chat_file_id"], "user", prompt)

    # 2) Caminho 1: pesquisa web explícita
    if do_web and prompt.strip().lower().startswith("web:"):
        query = prompt.split(":", 1)[1].strip() or prompt
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
            # botão de copiar
            _copy_button(acc, key="copy_web")
        answer = acc

    # 3) Caminho 2: RAG com acervo (prioridade máxima)
    else:
        answer = _rag_answer(llm, prompt, st.session_state.get("faiss_store"), k_acervo, copy_key="rag")

    # 4) Acrescenta resposta ao histórico da sessão
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # 5) Grava a resposta no arquivo de chat
    if st.session_state.get("chat_file_id"):
        _append_to_chat_file(service, st.session_state["chat_file_id"], "assistant", answer)
        # Atualiza embeddings do chat no Vecstore
        try:
            full_chat_text = _download_chat_text(service, st.session_state["chat_file_id"])
            _update_chat_embeddings(service, vec_id, st.session_state["chat_file_id"], full_chat_text)
        except Exception as e:
            st.warning(f"Memória do chat salva, mas houve falha ao indexar no Vecstore: {e}")



