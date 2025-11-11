# src/storage/drive.py
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from urllib.parse import urlencode, quote as urlquote

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_OAUTH_SCOPE = "https://www.googleapis.com/auth/drive.file openid email profile"


# ---------- helpers ----------
def _assert_secrets() -> None:
    missing = [k for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI") if k not in st.secrets]
    if missing:
        raise RuntimeError(f"Configure {', '.join(missing)} em .streamlit/secrets.toml")


def _esc_drive_str(s: str) -> str:
    """Escapa aspas e barras para a query do Drive (Apostrophe precisa de \\')."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


# ---------- OAuth ----------
def get_auth_url() -> str:
    """Gera a URL de autorização (força seletor de conta)."""
    _assert_secrets()
    params = {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
        "response_type": "code",
        "scope": GOOGLE_OAUTH_SCOPE,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent select_account",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params, quote_via=urlquote)}"


def exchange_code_for_token(code: str) -> Dict[str, Any]:
    _assert_secrets()
    data = {
        "code": code,
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
        "grant_type": "authorization_code",
    }
    resp = requests.post(TOKEN_ENDPOINT, data=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Falha ao obter token ({resp.status_code}): {resp.text[:300]}")
    token = resp.json()
    if "expires_in" in token:
        token["expires_at"] = int(time.time()) + int(token["expires_in"])
    return token


def refresh_token_if_needed(token: Dict[str, Any]) -> Dict[str, Any]:
    if not token:
        raise RuntimeError("Token ausente.")
    # ainda válido?
    if token.get("expires_at", 0) - int(time.time()) > 60:
        return token
    if "refresh_token" not in token:
        return token

    data = {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "refresh_token": token["refresh_token"],
        "grant_type": "refresh_token",
    }
    resp = requests.post(TOKEN_ENDPOINT, data=data, timeout=30)
    if resp.status_code != 200:
        return token
    newt = resp.json()
    token["access_token"] = newt.get("access_token", token.get("access_token"))
    if "expires_in" in newt:
        token["expires_at"] = int(time.time()) + int(newt["expires_in"])
    return token


def drive_service_from_token(token: Dict[str, Any]):
    token = refresh_token_if_needed(token)
    creds = Credentials(
        token=token.get("access_token"),
        refresh_token=token.get("refresh_token"),
        token_uri=TOKEN_ENDPOINT,
        client_id=st.secrets["GOOGLE_CLIENT_ID"],
        client_secret=st.secrets["GOOGLE_CLIENT_SECRET"],
        scopes=GOOGLE_OAUTH_SCOPE.split(),
    )
    # cache_discovery=False evita tentativa de cache em disco
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------- Drive utils ----------
def _query_and_list(service, q: str, page_size: int = 100) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
            pageToken=page_token,
            pageSize=page_size,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def find_or_create_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    q = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    q += f" and name = '{_esc_drive_str(name)}'"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = _query_and_list(service, q, page_size=50)
    if res:
        return res[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    folder = service.files().create(body=body, fields="id").execute()
    return folder["id"]


def ensure_subfolder(service, parent_id: str, subfolder_name: str) -> str:
    return find_or_create_folder(service, subfolder_name, parent_id=parent_id)


def list_files_in_folder(
    service,
    folder_id: str,
    mime_type: Optional[str] = None,
    name_equals: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = f"trashed = false and '{folder_id}' in parents"
    if mime_type:
        q += f" and mimeType = '{_esc_drive_str(mime_type)}'"
    if name_equals:
        q += f" and name = '{_esc_drive_str(name_equals)}'"
    return _query_and_list(service, q, page_size=100)


def list_files_md(service, folder_id: str, extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    files = list_files_in_folder(service, folder_id)
    if extensions:
        exts = {e.lower() for e in extensions}
        files = [f for f in files if any(f["name"].lower().endswith(x) for x in exts)]
    files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    return files


def upload_text(service, folder_id: str, filename: str, text: str) -> str:
    media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/plain", resumable=False)
    body = {"name": filename, "parents": [folder_id]}
    file = service.files().create(body=body, media_body=media, fields="id").execute()
    return file["id"]


def download_text(service, file_id: str) -> str:
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", errors="replace")


def upload_binary(service, folder_id: str, filename: str, data: bytes, mimetype: str = "application/octet-stream") -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    body = {"name": filename, "parents": [folder_id]}
    file = service.files().create(body=body, media_body=media, fields="id").execute()
    return file["id"]


def download_binary(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def update_file_contents(service, file_id: str, data: bytes, mimetype: str = "application/octet-stream") -> None:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    service.files().update(fileId=file_id, media_body=media).execute()


def safe_delete(service, file_id: str) -> None:
    try:
        service.files().delete(fileId=file_id).execute()
    except HttpError as e:
        # ignora 404
        if getattr(e, "status_code", None) == 404:
            return
        raise

