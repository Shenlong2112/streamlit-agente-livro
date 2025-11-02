# src/storage/drive.py
from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any, List
import requests
import streamlit as st

# ---- Config (lê do secrets.toml) ----
GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
APP_FOLDER = st.secrets.get("APP_FOLDER", "Apps/AgenteLivro")

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DRIVE_FILES = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
DRIVE_ABOUT = "https://www.googleapis.com/drive/v3/about"

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Em DEV local, a rota da sua página de conexões:
REDIRECT_URI = "http://localhost:8501/Conexoes"

# ------------------ OAuth helpers ------------------

def _assert_secrets():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise RuntimeError("Configure GOOGLE_CLIENT_ID/SECRET em .streamlit/secrets.toml")

def get_auth_url() -> str:
    _assert_secrets()
    from urllib.parse import urlencode
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"

def handle_oauth_callback(code: str) -> Dict[str, Any]:
    _assert_secrets()
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    r = requests.post(TOKEN_ENDPOINT, data=data, timeout=30)
    r.raise_for_status()
    token = r.json()
    token["created_at"] = int(time.time())
    return token

def _auth_header(token: Dict[str,Any]) -> Dict[str,str]:
    return {"Authorization": f"Bearer {token['access_token']}"}

def _refresh_token(token: Dict[str,Any]) -> Dict[str,Any]:
    _assert_secrets()
    if "refresh_token" not in token:
        raise RuntimeError("Sem refresh_token. Refaça a conexão ao Google Drive.")
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": token["refresh_token"],
        "grant_type": "refresh_token",
    }
    r = requests.post(TOKEN_ENDPOINT, data=data, timeout=30)
    r.raise_for_status()
    newtok = r.json()
    token.update(newtok)
    token["created_at"] = int(time.time())
    return token

def _ensure_fresh_token(token: Dict[str,Any]) -> Dict[str,Any]:
    created = token.get("created_at", 0)
    # margem antes dos 3600s
    if int(time.time()) - created > 3300:
        token = _refresh_token(token)
    return token

# ------------------ Drive basic ops ------------------

def drive_me(token: Dict[str,Any]) -> Dict[str,Any]:
    token = _ensure_fresh_token(token)
    params = {"fields": "user"}
    r = requests.get(DRIVE_ABOUT, headers=_auth_header(token), params=params, timeout=30)
    r.raise_for_status()
    return r.json()["user"]

def ensure_app_folder(token: Dict[str,Any]) -> str:
    token = _ensure_fresh_token(token)
    q = f"name = '{APP_FOLDER}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    params = {"q": q, "fields": "files(id,name)"}
    r = requests.get(DRIVE_FILES, headers=_auth_header(token), params=params, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": APP_FOLDER, "mimeType": "application/vnd.google-apps.folder"}
    r = requests.post(DRIVE_FILES, headers=_auth_header(token), json=meta, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def upload_bytes(token: Dict[str,Any], name: str, data: bytes, parent_id: Optional[str]=None, mime: str="application/octet-stream") -> str:
    token = _ensure_fresh_token(token)
    metadata = {"name": name}
    if parent_id:
        metadata["parents"] = [parent_id]
    mmeta = ("metadata", ("metadata.json", json.dumps(metadata), "application/json; charset=UTF-8"))
    mfile = ("file", (name, data, mime))
    r = requests.post(
        f"{DRIVE_UPLOAD}?uploadType=multipart",
        headers=_auth_header(token),
        files=[mmeta, mfile],
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["id"]

def download_file(token: Dict[str,Any], file_id: str) -> bytes:
    token = _ensure_fresh_token(token)
    r = requests.get(f"{DRIVE_FILES}/{file_id}?alt=media", headers=_auth_header(token), timeout=120)
    r.raise_for_status()
    return r.content

def find_file(token: Dict[str,Any], name: str, parent_id: Optional[str]=None) -> Optional[str]:
    token = _ensure_fresh_token(token)
    if parent_id:
        q = f"name = '{name}' and '{parent_id}' in parents and trashed = false"
    else:
        q = f"name = '{name}' and trashed = false"
    params = {"q": q, "fields": "files(id,name)"}
    r = requests.get(DRIVE_FILES, headers=_auth_header(token), params=params, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    return files[0]["id"] if files else None

def ensure_subfolder(token: Dict[str,Any], parent_id: str, name: str) -> str:
    """Garante uma subpasta e retorna o id."""
    token = _ensure_fresh_token(token)
    q = f"name = '{name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    params = {"q": q, "fields": "files(id,name)"}
    r = requests.get(DRIVE_FILES, headers=_auth_header(token), params=params, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    r = requests.post(DRIVE_FILES, headers=_auth_header(token), json=meta, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def update_file_bytes(token: Dict[str,Any], file_id: str, name: str, data: bytes, mime: str="application/octet-stream") -> str:
    """Atualiza conteúdo de um arquivo existente (mantém o mesmo file_id)."""
    token = _ensure_fresh_token(token)
    metadata = {"name": name}
    mmeta = ("metadata", ("metadata.json", json.dumps(metadata), "application/json; charset=UTF-8"))
    mfile = ("file", (name, data, mime))
    r = requests.patch(
        f"{DRIVE_UPLOAD}/{file_id}?uploadType=multipart",
        headers=_auth_header(token),
        files=[mmeta, mfile],
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["id"]

def delete_file(token: Dict[str,Any], file_id: str) -> None:
    token = _ensure_fresh_token(token)
    r = requests.delete(f"{DRIVE_FILES}/{file_id}", headers=_auth_header(token), timeout=30)
    if r.status_code not in (200, 204):
        r.raise_for_status()

# --------------- NOVO: listar arquivos de uma pasta ---------------

def list_files_in_folder(
    token: Dict[str,Any],
    parent_id: str,
    mime_contains: Optional[str] = None,
    name_suffix: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Lista arquivos em uma pasta do Drive.
    - mime_contains: filtra por substring do mimeType (ex.: 'json', 'zip')
    - name_suffix: filtra por sufixo do nome (ex.: '.json')
    Retorna: lista de dicts {id, name, mimeType, modifiedTime}
    """
    token = _ensure_fresh_token(token)
    q = f"'{parent_id}' in parents and trashed = false"
    if mime_contains:
        q += f" and mimeType contains '{mime_contains}'"
    params = {
        "q": q,
        "pageSize": min(limit, 1000),
        "fields": "files(id,name,mimeType,modifiedTime)",
        "orderBy": "modifiedTime desc",
    }
    r = requests.get(DRIVE_FILES, headers=_auth_header(token), params=params, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    if name_suffix:
        files = [f for f in files if f.get("name","").endswith(name_suffix)]
    return files
