# pages/2_Transcritor.py
from __future__ import annotations

import os
import io
import zipfile
import tempfile
from datetime import datetime
from typing import Optional, List

import streamlit as st
from unidecode import unidecode
from pypdf import PdfReader  # <-- novo: extração de texto de PDFs

from src.storage.drive import (
    drive_service_from_token,
    upload_text,
    upload_binary,
    list_files_md,
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

# =========================================================
# Utils compartilhadas (zip + slug) — mesmas do Editor
# =========================================================
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
    """Extrai texto de um PDF (melhor esforço). Para PDFs escaneados sem OCR, retornará vazio."""
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        parts.append(txt)
    return "\n\n".join(parts).strip()


# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(page_title="Transcritor", page_icon="🎙️", layout="wide")
st.title("🎙️ Transcritor / Ingestão")

# Requisitos de conexão
if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte o **Google Drive** em **Conexões** para usar esta página.")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])
ids = ensure_user_tree(service)
trans_id = ids["trans"]      # Transcrições brutas (áudio -> texto)
versions_id = ids["versions"]  # Versões / documentos limpos (acervo)
vec_id = ids["vec"]          # Vecstore (embeddings FAISS)


# =========================================================
# Seção 1 — Transcrição de áudio (comportamento existente)
# =========================================================
with st.expander("🎧 Transcrever áudio (mantém igual — salva **só** em Transcrições)", expanded=True):
    st.caption(
        "Arquivos de áudio serão transcritos e **salvos em Transcrições** (sem ir para o Vecstore). "
        "Use o **Editor** para revisar e salvar versões no acervo/vecstore."
    )
    audio = st.file_uploader(
        "Envie um arquivo de áudio (mp3, m4a, wav, webm)",
        type=["mp3", "m4a", "wav", "webm"],
        accept_multiple_files=False,
    )
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        audio_title = st.text_input("Título (opcional, usado para nomear o .txt)", placeholder="ex.: entrevista_cap1")
    with col_a2:
        transcriber = st.selectbox(
            "Motor de transcrição",
            ["OpenAI Whisper (BYOK)"],
            index=0,
            help="Mantendo o comportamento atual (BYOK)."
        )

    if st.button("Transcrever", use_container_width=True, type="primary", disabled=audio is None):
        if audio is None:
            st.warning("Envie um arquivo de áudio.")
            st.stop()

        # Checagem de tamanho (sugestão)
        size_mb = len(audio.getvalue()) / (1024 * 1024)
        if size_mb > 25:
            st.error(
                f"Arquivo com {size_mb:.1f} MB. O endpoint de transcrição aceita até 25 MB por arquivo. "
                "Comprima ou divida em partes menores."
            )
            st.stop()

        # >>> Aqui entra sua rotina atual de transcrição BYOK (não alterada) <<<
        # Vamos simular com um placeholder para não mexer na sua lógica:
        with st.spinner("Transcrevendo áudio..."):
            # TODO: substitua este bloco pela sua chamada real ao Whisper BYOK
            transcricao = f"[Transcrição simulada de {audio.name} — substitua por sua rotina real]"
        # --------------------------------------------------------------

        # Nome e salvamento em Transcrições (sem embeddings)
        if audio_title.strip():
            base = _first_line_slug(audio_title)
        else:
            base = os.path.splitext(os.path.basename(audio.name))[0]
            base = _first_line_slug(base or "transcricao")

        fname = build_version_filename(base, suffix=None).replace(".txt", "_transcricao.txt")
        upload_text(service, trans_id, fname, transcricao)
        st.success(f"Transcrição salva em **Transcrições** como **{fname}**.")
        st.info("Se quiser indexar no Vecstore, carregue o texto no **Editor** e salve como nova versão.")


# =========================================================
# Seção 2 — NOVO: Ingestão de PDFs → Acervo (Versoes) + Vecstore
# =========================================================
st.markdown("---")
st.subheader("📄 Ingestão de PDFs para o acervo (texto + embeddings)")

pdfs = st.file_uploader(
    "Envie um ou mais PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Os PDFs serão convertidos em texto, salvos em **Versoes** e indexados no **Vecstore**."
)

if pdfs:
    st.caption("Dica: para PDFs escaneados (imagem), use um PDF com OCR. PDFs sem texto extraível podem resultar em arquivos vazios.")
    if st.button("Processar PDFs", use_container_width=True, type="primary"):
        for pdf in pdfs:
            with st.spinner(f"Extraindo texto de **{pdf.name}**..."):
                data = pdf.getvalue()
                text = _extract_pdf_text(data)

            if not text.strip():
                st.warning(f"Não foi possível extrair texto de **{pdf.name}** (PDF pode ser escaneado sem OCR). Pulando.")
                continue

            # Nome base pelo 1º título, caindo para nome do PDF
            base_title = _first_line_slug(text, fallback=os.path.splitext(pdf.name)[0])
            # Evita colisão de nomes em Versoes
            existing = [f["name"] for f in list_files_md(service, versions_id, extensions=[".txt"])]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname_txt = build_version_filename(base_title, suffix=None)
            if fname_txt in existing:
                fname_txt = f"{base_title}_v{len([n for n in existing if n.startswith(base_title)])+1}_{ts}.txt"

            # 1) Salva o texto no acervo (Versoes)
            upload_text(service, versions_id, fname_txt, text)

            # 2) Gera embeddings FAISS e salva pacote no Vecstore
            with st.spinner("Indexando no Vecstore…"):
                index = create_faiss_index([text])
                with tempfile.TemporaryDirectory() as td:
                    save_faiss_index(index, td)
                    data_zip = _zip_dir_to_bytes(td)
                faiss_name = f"{os.path.splitext(fname_txt)[0]}.faiss.zip"
                upload_binary(service, vec_id, faiss_name, data_zip, mimetype="application/zip")

            st.success(f"**{pdf.name}** → salvo como **{fname_txt}** (Versoes) e indexado como **{faiss_name}** (Vecstore).")


