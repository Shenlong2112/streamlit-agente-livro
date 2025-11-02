# src/pipelines/transcribe.py
from __future__ import annotations

import re
from io import BytesIO
from typing import Optional, Tuple
from unidecode import unidecode

# OpenAI SDK v1
try:
    from openai import OpenAI
except Exception:  # fallback para pacotes antigos nomeados como openai>=1
    from openai import OpenAI  # type: ignore


def transcribe_audio(
    file_bytes: bytes,
    filename: str,
    openai_key: str,
    language: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Envia áudio para o Whisper (OpenAI) e retorna (texto, idioma_detectado).
    `language`: se None, o Whisper tenta detectar.
    """
    client = OpenAI(api_key=openai_key)

    # O SDK atual aceita file-like; usamos BytesIO
    file_obj = BytesIO(file_bytes)
    file_obj.name = filename  # ajuda o servidor a inferir tipo

    # Modelos comuns: "whisper-1" (clássico) ou "gpt-4o-transcribe" (mais novo)
    # Mantemos "whisper-1" por compatibilidade ampla.
    params = {
        "model": "whisper-1",
        "file": file_obj,
    }
    if language:
        params["language"] = language

    resp = client.audio.transcriptions.create(**params)

    # SDK retorna .text e .language dependendo do model/SDK
    text = getattr(resp, "text", None) or getattr(resp, "text", "")
    detected_lang = getattr(resp, "language", None)

    # limpeza leve do texto
    text = normalize_text(text)
    return text, detected_lang


def normalize_text(text: str) -> str:
    """Limpeza simples: espaços, linhas duplicadas, normalização básica."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove espaços repetidos
    text = re.sub(r"[ \t]+", " ", text)
    # normaliza quebras de linha múltiplas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_slug_from_text(text: str, max_words: int = 8) -> str:
    """
    Gera um slug curto a partir de um texto/título:
    - remove acentos
    - mantém letras/números
    - separa por hífens
    """
    # Pega as primeiras N palavras significativas
    words = re.findall(r"\b\w+\b", unidecode(text))
    if not words:
        return ""

    words = words[:max_words]
    slug = "-".join(w.lower() for w in words)
    # remove traços repetidos e bordas
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    # fallback se ficou muito curto
    if len(slug) < 3:
        slug = "documento"
    return slug
