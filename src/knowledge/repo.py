# src/knowledge/repo.py
from __future__ import annotations

import json
import time
import re
from typing import Dict, Any, Optional

from unidecode import unidecode

from src.storage.drive import (
    ensure_app_folder,
    ensure_subfolder,
    find_file,
    upload_bytes,
    download_file,
    update_file_bytes,
)

# ===== Pastas lógicas dentro do APP_FOLDER no Drive =====
TRANSCRICAO_DIR = "transcricao"   # ← transcrições brutas (Whisper)
VERSOES_DIR     = "versoes"       # ← versões revisadas (Editor)

# ----------------- helpers internos -----------------

def _folder_id(token: Dict[str, Any], subfolder: str) -> str:
    """Resolve o ID da subpasta (transcricao/versoes) dentro do APP_FOLDER."""
    root = ensure_app_folder(token)
    return ensure_subfolder(token, root, subfolder)

def _json_name(doc_id: str) -> str:
    return f"{doc_id}.json"

_slug_ok = re.compile(r"[^a-z0-9\-]+")
def _slugify(s: str) -> str:
    if not s:
        return "x"
    s = unidecode(s).lower().strip()
    s = s.replace(" ", "-").replace("_", "-")
    s = _slug_ok.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"

def _write_version_txt(
    token: Dict[str, Any],
    doc_id: str,
    text: str,
    subfolder: str,
    version_index: int,
    source: str = "versao",
) -> str:
    """Cria um arquivo .txt individual para a versão."""
    folder_id = _folder_id(token, subfolder)
    name = f"{_slugify(doc_id)}__{_slugify(source)}__{version_index:03d}.txt"
    return upload_bytes(
        token,
        name,
        text.encode("utf-8"),
        parent_id=folder_id,
        mime="text/plain; charset=UTF-8",
    )

# ----------------- API pública -----------------

def save_doc(
    token: Dict[str, Any],
    doc_id: str,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
    subfolder: str = VERSOES_DIR,
) -> str:
    """
    Cria um novo manifesto JSON (histórico) com a primeira versão (se texto vier).
    Retorna file_id do JSON no Drive. Também grava .txt da v1 se 'text' não for vazio.
    """
    ts = int(time.time())
    payload = {
        "id": doc_id,
        "created_at": ts,
        "versions": [],
    }
    # Se já vier uma primeira versão
    if text:
        payload["versions"].append({"ts": ts, "text": text, "meta": (meta or {})})

    folder_id = _folder_id(token, subfolder)
    name = _json_name(doc_id)
    fid = upload_bytes(
        token,
        name,
        # Sem escapes unicode feios
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        parent_id=folder_id,
        mime="application/json; charset=UTF-8",
    )

    # Se já tinha texto, gravar .txt v1
    if text:
        source = (meta or {}).get("source", "versao")
        _write_version_txt(token, doc_id, text, subfolder, version_index=1, source=source)

    return fid

def get_doc(
    token: Dict[str, Any],
    doc_id: str,
    subfolder: str = VERSOES_DIR,
) -> Dict[str, Any]:
    """
    Carrega o manifesto JSON (lança FileNotFoundError se não existir).
    """
    folder_id = _folder_id(token, subfolder)
    name = _json_name(doc_id)
    fid = find_file(token, name, parent_id=folder_id)
    if not fid:
        raise FileNotFoundError(f"{name} não encontrado em {subfolder}")
    data = download_file(token, fid)
    return json.loads(data.decode("utf-8"))

def append_version(
    token: Dict[str, Any],
    doc_id: str,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
    subfolder: str = VERSOES_DIR,
) -> str:
    """
    Acrescenta uma nova versão ao manifesto JSON **e** salva um .txt próprio.
    Retorna o file_id atualizado (JSON) no Drive.
    """
    folder_id = _folder_id(token, subfolder)
    name = _json_name(doc_id)
    fid = find_file(token, name, parent_id=folder_id)
    if not fid:
        # se não existe, cria como primeira versão
        return save_doc(token, doc_id, text, meta, subfolder=subfolder)

    current = json.loads(download_file(token, fid).decode("utf-8"))
    current["versions"].append({
        "ts": int(time.time()),
        "text": text,
        "meta": (meta or {}),
    })

    # Atualiza manifesto JSON (sem \uXXXX)
    fid = update_file_bytes(
        token,
        fid,
        name,
        json.dumps(current, ensure_ascii=False).encode("utf-8"),
        mime="application/json; charset=UTF-8",
    )

    # Grava .txt para ESTA versão
    version_index = len(current["versions"])  # 1-based
    source = (meta or {}).get("source", "versao")
    _write_version_txt(token, doc_id, text, subfolder, version_index=version_index, source=source)

    return fid

