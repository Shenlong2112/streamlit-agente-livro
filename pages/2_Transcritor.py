# pages/2_Transcritor.py
from __future__ import annotations

import os
import io
import zipfile
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
from unidecode import unidecode
from pypdf import PdfReader

from src.storage.drive import (
    drive_service_from_token,
    upload_text,
    upload_binary,
    list_files_md,
)
from src.knowledge.repo import (
    ensure_user_tree,
    TRANSCRICAO_DIR,
    VERSOES_DIR,      # mantido (áudio bruto não muda o fluxo)
    VECSTORE_DIR,
    REFERENCIAS_DIR,  # <<< NOVO
    build_version_filename,
)
from src.embeddings.vectorstore_faiss import (
    create_faiss_index,
    save_faiss_index,
)

# ---------- Utils ----------
def _zip_dir_to_bytes(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path)
                zf.write(full, rel)
    return buf.getvalue()

def _first_line_slug(text: str, fallback: str = "documento") -> str:
    base = (text or "").strip().split("\n", 1)[0] or fallback
    base = unidecode(base).lower()
    keep = []
    for ch in base:
        if ch.isalnum() or ch in (" ", "-", "_", "."):
            keep.append(ch)
    slug = "".join(keep).strip().replace(" ", "_")
    return slug[:60] or fallback

def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extrai texto de um PDF. (Para PDFs escaneados sem OCR, pode retornar vazio.)"""
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        parts.append(txt)
    return "\n\n".join(parts).strip()

# ---------- Página ----------
st.set_page_config(page_title="Transcritor", page_icon="🎙️", layout="wide")
st.title("🎙️ Transcritor / Ingestão")

# Conexões
if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte o **Google Drive** em **Conexões** para usar esta página.")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])
ids = ensure_user_tree(service)
trans_id = ids["trans"]       # Transcrições brutas (áudio -> texto)
versions_id = ids["versions"] # (mantido, mas não usaremos para PDFs)
vec_id = ids["vec"]           # Vecstore (.faiss.zip)
refs_id = ids["refs"]         # <<< NOVO: pasta Referencias

# ===============================
# Seção 1 — Transcrição de Áudio
# ===============================
with st.expander("🎧 Transcrever áudio (salva **só** em Transcrições)", expanded=True):
    st.caption(
        "Áudios serão transcritos e salvos em **Transcrições** (sem embeddings). "
        "Para indexar no acervo/vecstore, leve o texto ao **Editor** e salve como nova versão."
    )
    audio = st.file_uploader(
        "Envie um arquivo de áudio (mp3, m4a, wav, webm)",
        type=["mp3", "m4a", "wav", "webm"],
        accept_multiple_files=False,
    )
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        audio_title = st.text_input("Título (opcional, para nome do .txt)", placeholder="ex.: entrevista_cap1")
    with col_a2:
        st.selectbox(
            "Motor de transcrição",
            ["OpenAI Whisper (BYOK)"],
            index=0,
            help="Mantendo o comportamento atual (BYOK)."
        )

    if st.button("Transcrever", use_container_width=True, type="primary", disabled=audio is None):
        if audio is None:
            st.warning("Envie um arquivo de áudio.")
            st.stop()

        # Checagem de tamanho para o endpoint (≈25 MB)
        size_mb = len(audio.getvalue()) / (1024 * 1024)
        if size_mb > 25:
            st.error(
                f"Arquivo com {size_mb:.1f} MB. O endpoint de transcrição aceita até 25 MB por arquivo. "
                "Comprima ou divida em partes menores."
            )
            st.stop()

        # >>> Substitua pelo seu fluxo real de transcrição (BYOK) <<<
        with st.spinner("Transcrevendo áudio..."):
            transcricao = f"[Transcrição simulada de {audio.name} — substitua pela chamada real]"
        # -----------------------------------------------------------

        # Nome e salvamento em Transcrições (sem embeddings)
        if audio_title.strip():
            base = _first_line_slug(audio_title)
        else:
            base = os.path.splitext(os.path.basename(audio.name))[0]
            base = _first_line_slug(base or "transcricao")

        fname = build_version_filename(base, suffix=None).replace(".txt", "_transcricao.txt")
        upload_text(service, trans_id, fname, transcricao)
        st.success(f"Transcrição salva em **{TRANSCRICAO_DIR}** como **{fname}**.")
        st.info("Para indexar no acervo/vecstore, use o **Editor** e salve como nova versão.")

# =======================================
# Seção 2 — Ingestão de PDFs (REFERENCIAS)
# =======================================
st.markdown("---")
st.subheader("📄 Ingestão de PDFs (salva em **Referencias** + indexa no **Vecstore**)")

pdfs = st.file_uploader(
    "Envie um ou mais PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Os PDFs serão convertidos em texto, salvos como .txt em **Referencias** e indexados no **Vecstore**."
)

if pdfs:
    st.caption("Dica: para PDFs escaneados (imagem), use OCR; sem OCR, o texto pode sair vazio.")
    if st.button("Processar PDFs", use_container_width=True, type="primary"):
        for pdf in pdfs:
            with st.spinner(f"Extraindo texto de **{pdf.name}**..."):
                data = pdf.getvalue()
                text = _extract_pdf_text(data)

            if not text.strip():
                st.warning(f"Não foi possível extrair texto de **{pdf.name}** (PDF pode ser escaneado sem OCR). Pulando.")
                continue

            # Nome base pelo 1º título (ou nome do PDF)
            base_title = _first_line_slug(text, fallback=os.path.splitext(pdf.name)[0])
            existing = [f["name"] for f in list_files_md(service, refs_id, extensions=[".txt"])]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname_txt = build_version_filename(base_title, suffix=None)
            if fname_txt in existing:
                # Evita colisão de nomes
                fname_txt = f"{base_title}_v{len([n for n in existing if n.startswith(base_title)])+1}_{ts}.txt"

            # 1) Salva o texto extraído em **Referencias**
            upload_text(service, refs_id, fname_txt, text)

            # 2) Gera embeddings e salva pacote no **Vecstore**
            with st.spinner("Indexando no Vecstore…"):
                index = create_faiss_index([text])
                with tempfile.TemporaryDirectory() as td:
                    save_faiss_index(index, td)
                    data_zip = _zip_dir_to_bytes(td)
                faiss_name = f"{os.path.splitext(fname_txt)[0]}.faiss.zip"
                upload_binary(service, vec_id, faiss_name, data_zip, mimetype="application/zip")

            st.success(
                f"**{pdf.name}** → salvo como **{fname_txt}** em **{REFERENCIAS_DIR}** "
                f"e indexado como **{faiss_name}** em **{VECSTORE_DIR}**."
            )


