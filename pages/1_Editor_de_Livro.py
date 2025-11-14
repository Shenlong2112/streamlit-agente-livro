# pages/1_Editor_de_Livro.py
from __future__ import annotations

import os
import io
import zipfile
import tempfile
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING, Any

import streamlit as st
from unidecode import unidecode

from src.storage.drive import (
    drive_service_from_token,
    list_files_md,
    download_text,
    upload_text,
    upload_binary,
)
from src.knowledge.repo import (
    ensure_user_tree,
    TRANSCRICAO_DIR,
    VERSOES_DIR,
    VECSTORE_DIR,
    build_version_filename,
)
from src.embeddings.vectorstore_faiss import (
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


# =============== Utils ===============
def _zip_dir_to_bytes(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path)
                zf.write(full, rel)
    return buf.getvalue()


def _first_line_slug(text: str, fallback: str = "versao") -> str:
    base = (text or "").strip().split("\n", 1)[0]
    base = unidecode(base).lower()
    keep = []
    for ch in base:
        if ch.isalnum() or ch in (" ", "-", "_", "."):
            keep.append(ch)
    slug = "".join(keep).strip().replace(" ", "_")
    slug = slug or fallback
    return slug[:60]


def _get_llm() -> Optional[ChatOpenAI]:
    if not st.session_state.get("OPENAI_API_KEY"):
        return None
    if _ChatOpenAI is None:
        return None
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
    return _ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        openai_api_key=st.session_state["OPENAI_API_KEY"],
    )


def _format_for_book(llm: ChatOpenAI, raw_text: str, style_prompt: str) -> str:
    sys = (
        "Você é um **editor de livros tradicional**. Seu trabalho é **corrigir gramática, clareza, coesão** "
        "e **formatar** o texto para um livro, mantendo a **voz do autor** e **sem inventar fatos**. "
        "Quando houver ambiguidade, prefira a versão mais natural em Português do Brasil."
    )
    usr = f"Estilo/audiência/instruções do autor:\n{style_prompt}\n\nTexto original:\n{raw_text}"
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    acc = ""
    for chunk in llm.stream(messages):
        acc += (chunk.content or "")
    return acc


# =============== Página ===============
st.set_page_config(page_title="Editor de Livro", page_icon="📝", layout="wide")
st.title("📝 Editor de Livro")

# Requisitos
if not st.session_state.get("OPENAI_API_KEY"):
    st.warning("Cole sua **OPENAI_API_KEY** em **Conexões** para usar o Editor.")
    st.stop()

if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte o **Google Drive** em **Conexões** para usar o Editor.")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])
ids = ensure_user_tree(service)
trans_id = ids["trans"]
versions_id = ids["versions"]
vec_id = ids["vec"]

# Aplica novo texto (se veio de geração) **antes** de renderizar widgets
st.session_state.setdefault("texto_atual_editor", "")
if st.session_state.get("_pending_new_text") is not None:
    st.session_state["texto_atual_editor"] = st.session_state.pop("_pending_new_text")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("📂 Selecionar texto do Drive")
    origem = st.radio("Origem", ["Transcrições", "Versões"], horizontal=True)
    folder_id = trans_id if origem == "Transcrições" else versions_id

    files = list_files_md(service, folder_id, extensions=[".txt"])
    options = [f["name"] for f in files] if files else []
    sel = st.selectbox("Arquivo", options, index=0 if options else None, placeholder="Escolha...")
    if sel and st.button("Carregar no editor", use_container_width=True):
        file_id = next(f["id"] for f in files if f["name"] == sel)
        content = download_text(service, file_id)
        st.session_state["_pending_new_text"] = content
        st.rerun()

with colB:
    st.subheader("🎯 Instruções do editor")
    instr = st.text_area(
        "Diga o estilo/audiência/observações",
        value=st.session_state.get("editor_instrucoes", ""),
        key="editor_instrucoes",
        height=140,
        placeholder="Ex.: Tom acessível, público leigo, evitar jargão; ritmo mais direto…",
    )

st.markdown("---")

# Bloco principal de edição
st.subheader("🖊️ Texto atual")
texto_atual = st.text_area(
    "Edite livremente abaixo. Este é o texto que será salvo como nova versão.",
    key="texto_atual_editor",
    height=420,
)

col1, col2 = st.columns(2)
with col1:
    if st.button("✨ Gerar nova versão a partir do texto atual", use_container_width=True):
        llm = _get_llm()
        if llm is None:
            st.error("LLM indisponível. Verifique sua OPENAI_API_KEY.")
            st.stop()
        with st.spinner("Editando com o agente…"):
            novo = _format_for_book(llm, texto_atual, st.session_state.get("editor_instrucoes", ""))
        st.session_state["_pending_new_text"] = novo
        st.success("Nova versão gerada. Atualizando editor…")
        st.rerun()

with col2:
    if st.button("💾 Salvar edição como **nova versão** (Drive + Vecstore)", use_container_width=True):
        if not texto_atual.strip():
            st.warning("Não há texto para salvar.")
            st.stop()

        base_title = _first_line_slug(texto_atual, "versao")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname_text = build_version_filename(base_title, suffix=None)
        existing = [f["name"] for f in list_files_md(service, versions_id, extensions=[".txt"])]
        same_base = [n for n in existing if n.startswith(base_title)]
        if same_base:
            fname_text = f"{base_title}_v{len(same_base)+1}_{ts}.txt"

        with st.spinner("Salvando versão em texto…"):
            _ = upload_text(service, versions_id, fname_text, texto_atual)

        with st.spinner("Indexando esta versão no Vecstore…"):
            index = create_faiss_index([texto_atual])
            with tempfile.TemporaryDirectory() as td:
                save_faiss_index(index, td)
                data = _zip_dir_to_bytes(td)
            faiss_name = f"{os.path.splitext(fname_text)[0]}.faiss.zip"
            upload_binary(service, vec_id, faiss_name, data, mimetype="application/zip")

        st.success(f"Versão salva como **{fname_text}** e indexada no **Vecstore**.")

st.caption(
    "Dica: a cada clique em **Salvar**, um arquivo `.txt` é criado em **Versoes** e um pacote `.faiss.zip` correspondente é salvo em **Vecstore**."
)







