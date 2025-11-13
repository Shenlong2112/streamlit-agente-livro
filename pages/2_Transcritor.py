# pages/2_Transcritor.py
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from unidecode import unidecode

from src.storage.drive import (
    drive_service_from_token,
    upload_text,
)
from src.knowledge.repo import ensure_user_tree, TRANSCRICAO_DIR


# ============== Helpers ==============
def _slug_from_text(txt: str, max_len: int = 60) -> str:
    base = unidecode((txt or "").strip())
    if "\n" in base:
        base = base.split("\n", 1)[0].strip()
    keep = []
    for ch in base.lower():
        if ch.isalnum() or ch in (" ", "-", "_", "."):
            keep.append(ch)
    slug = "".join(keep).strip().replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug[:max_len] or "transcricao").strip("_")


def _openai_whisper_transcribe(file_name: str, file_bytes: bytes, api_key: str) -> str:
    """
    Transcreve áudio usando OpenAI Whisper (sem depender do pacote openai).
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {
        "file": (file_name, io.BytesIO(file_bytes), "application/octet-stream"),
    }
    data = {
        "model": "whisper-1",
        "response_format": "text",
        "temperature": "0",
    }
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Falha na transcrição ({resp.status_code}): {resp.text[:300]}")
    return resp.text


# ============== Página ==============
st.set_page_config(page_title="Transcritor", page_icon="🎙️", layout="centered")
st.title("🎙️ Transcritor (Whisper BYOK)")

# Requisitos
if not st.session_state.get("OPENAI_API_KEY"):
    st.warning("Cole sua **OPENAI_API_KEY** em **Conexões** para transcrever.")
    st.stop()

if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte o **Google Drive** em **Conexões** para salvar as transcrições.")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])
ids = ensure_user_tree(service)
trans_folder_id = ids["trans"]  # pasta "Transcricoes"

st.caption(f"As transcrições brutas serão salvas em **{TRANSCRICAO_DIR}** no seu Google Drive.")

audio = st.file_uploader(
    "Envie um arquivo de áudio (mp3, wav, m4a, ogg, webm)",
    type=["mp3", "wav", "m4a", "ogg", "webm"],
    accept_multiple_files=False,
)

if audio is not None:
    st.audio(audio)
    if st.button("Transcrever e salvar no Drive", use_container_width=True):
        with st.spinner("Transcrevendo com Whisper..."):
            try:
                content = audio.read()
                text = _openai_whisper_transcribe(audio.name, content, st.session_state["OPENAI_API_KEY"])
            except Exception as e:
                st.error(f"Falha na transcrição: {e}")
                st.stop()

        # Nome auto que remete ao conteúdo
        snippet = _slug_from_text(text, max_len=60)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{snippet}_{timestamp}.txt"

        with st.spinner("Salvando no Google Drive..."):
            try:
                _ = upload_text(service, trans_folder_id, filename, text)
                st.success(f"Transcrição salva como **{filename}** em **{TRANSCRICAO_DIR}**.")
                with st.expander("Pré-visualização da transcrição"):
                    st.text_area("Transcrição", value=text, height=320)
            except Exception as e:
                st.error(f"Falha ao salvar no Drive: {e}")


