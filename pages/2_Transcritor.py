# pages/2_Transcritor.py
from __future__ import annotations

import io
import os
import re
import tempfile
from typing import Optional, Dict, Any

import streamlit as st
from unidecode import unidecode

from openai import OpenAI
from langchain_openai import ChatOpenAI

from src.knowledge.repo import append_version, TRANSCRICAO_DIR
from src.storage.drive import ensure_app_folder

st.set_page_config(page_title="Transcritor (Whisper)", page_icon="🎙️", layout="wide")
st.title("🎙️ Transcritor (Whisper) → Google Drive /transcricao")

# ------------------ GUARDAS ------------------
openai_key: Optional[str] = st.session_state.get("openai_key")
drive_token: Optional[Dict[str, Any]] = st.session_state.get("drive_token")

col1, col2 = st.columns(2)
with col1:
    st.caption("OpenAI (BYOK)")
    if openai_key:
        st.success("OPENAI_API_KEY carregada da sessão.")
    else:
        st.warning("Informe sua OPENAI_API_KEY na página **Conexões**.")

with col2:
    st.caption("Google Drive")
    if drive_token:
        try:
            ensure_app_folder(drive_token)
            st.success("Google Drive conectado.")
        except Exception as e:
            st.error(f"Drive não operacional: {e}")
            drive_token = None
    else:
        st.warning("Conecte o Google Drive na página **Conexões**.")

if not openai_key or not drive_token:
    st.stop()

st.divider()

# ------------------ HELPERS ------------------
_slug_ok = re.compile(r"[^a-z0-9\-]+")
def slugify(s: str) -> str:
    if not s:
        return "x"
    s = unidecode(s).lower().strip()
    s = s.replace(" ", "-").replace("_", "-")
    s = _slug_ok.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"

def suggest_title_from_text(text: str, api_key: str) -> str:
    """Pede um título curto e descritivo (5–8 palavras), sem pontuação. Fallback para começo do texto."""
    try:
        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)
        system = "Gere um título curto (5–8 palavras), sem pontuação nem aspas, descrevendo o conteúdo."
        prompt = f"Texto:\n{text[:3000]}"
        resp = llm.invoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompt}]
        )
        title = resp.content.strip()
        # higiene rápida
        title = re.sub(r"[^\w\s\-]", " ", title)
        title = re.sub(r"\s{2,}", " ", title).strip()
        return title[:120] if title else text[:60]
    except Exception:
        # fallback: primeiras palavras do texto
        preview = " ".join(text.strip().split()[:8])
        return preview or "transcricao"

# ------------------ UI: Upload & Transcrever ------------------
st.subheader("1) Envie um áudio e transcreva")

audio = st.file_uploader(
    "Selecione um arquivo de áudio (mp3, wav, m4a, mp4, webm)",
    type=["mp3", "wav", "m4a", "mp4", "webm"],
    accept_multiple_files=False,
)

col_a, col_b = st.columns([1, 1])
with col_a:
    lang_hint = st.text_input("Idioma (opcional, ex.: pt, en, es)", value="pt")
with col_b:
    temperature = st.slider("Temperature (Whisper, opcional)", 0.0, 1.0, 0.0, 0.1)

if st.button("📝 Transcrever com Whisper", type="primary", disabled=not audio):
    if not audio:
        st.warning("Envie um arquivo de áudio primeiro.")
        st.stop()

    with st.spinner("Transcrevendo áudio com Whisper…"):
        client = OpenAI(api_key=openai_key)

        # Grava temporário para compatibilidade com SDK
        suffix = os.path.splitext(audio.name)[1] or ".mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio.read())
        tmp.flush()
        tmp.close()

        try:
            with open(tmp.name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    # dicas (opcionais)
                    language=(lang_hint or None),
                    temperature=temperature,
                    response_format="text",  # retorna só o texto puro
                )
            transcribed_text = transcript  # já é string quando response_format="text"
        except Exception as e:
            st.error(f"Falha na transcrição: {e}")
            transcribed_text = ""
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    if transcribed_text:
        # Sugere um doc_id baseado no conteúdo
        title = suggest_title_from_text(transcribed_text, openai_key)
        doc_id_suggest = slugify(title)
        st.session_state["transcription_text"] = transcribed_text
        st.session_state["transcription_doc_id"] = doc_id_suggest
        st.success("Transcrição concluída! Revise os campos abaixo e salve no Drive.")
    else:
        st.stop()

st.divider()

# ------------------ UI: Revisar, nomear e salvar ------------------
st.subheader("2) Revisar e salvar no Drive (/transcricao)")

doc_id = st.text_input(
    "Nome (doc_id) para salvar no Drive — será usado em `/transcricao/{doc_id}.json` e `{doc_id}__whisper__NNN.txt`",
    value=st.session_state.get("transcription_doc_id", ""),
    placeholder="ex.: entrevista-cap-03-visao-estrategica",
)
transcribed_text_current = st.text_area(
    "Transcrição bruta (sem ajustes do editor)",
    value=st.session_state.get("transcription_text", ""),
    height=350,
)

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("💾 Salvar transcrição bruta no Drive (.txt)", type="primary", disabled=not (doc_id and transcribed_text_current)):
        try:
            # Salva como VERSÃO em /transcricao → gera .txt e atualiza manifesto .json
            append_version(
                st.session_state["drive_token"],
                doc_id=doc_id.strip(),
                text=transcribed_text_current,
                meta={"source": "whisper"},
                subfolder=TRANSCRICAO_DIR,
            )
            st.success(f"Transcrição salva em **/transcricao/** como `{doc_id}__whisper__NNN.txt` (+ manifesto `{doc_id}.json`).")
            # Atualiza sugestão para próxima ação
            st.session_state["transcription_doc_id"] = doc_id.strip()
        except Exception as e:
            st.error(f"Falha ao salvar no Drive: {e}")

with c2:
    if st.button("➡️ Enviar este texto para o Editor (sem salvar)"):
        # Prepara o Editor para fazer bootstrap a partir de transcrição
        st.session_state["current_doc_id"] = doc_id.strip()
        st.session_state["bootstrap_text"] = transcribed_text_current
        st.success("Texto enviado para o Editor. Abra a página **Editor de Livro** para continuar a revisão e salvar versões.")


st.caption(
    "• Este fluxo **NÃO** envia embeddings para o vecstore — a transcrição bruta fica somente em `/transcricao/`.\n"
    "• No **Editor de Livro**, ao salvar uma nova versão, o texto é gravado em `/versoes/` (.txt + manifesto) e indexado no vecstore.\n"
    "• O nome (doc_id) acima será usado para relacionar transcrições e versões do mesmo conteúdo."
)

