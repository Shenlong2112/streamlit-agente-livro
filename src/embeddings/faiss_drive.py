# src/embeddings/faiss_drive.py
from __future__ import annotations

import io
import os
import time
import zipfile
import tempfile
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.storage.drive import (
    ensure_app_folder,
    ensure_subfolder,
    list_files_in_folder,
    find_file,
    download_file,
    upload_bytes,
    update_file_bytes,
)

FAISS_DIRNAME = "faiss"
GLOBAL_INDEX_BASENAME = "_global.faiss.zip"


# ---------- Helpers de pasta/Drive ----------

def _faiss_folder_id(token: Dict[str, Any]) -> str:
    root_id = ensure_app_folder(token)
    vec_id = ensure_subfolder(token, root_id, "vecstore")
    return ensure_subfolder(token, vec_id, FAISS_DIRNAME)


def _name_for_doc(doc_id: str) -> str:
    return f"{doc_id}.faiss.zip"


def _get_file_bytes(token: Dict[str, Any], parent_id: str, name: str) -> Optional[bytes]:
    fid = find_file(token, name, parent_id=parent_id)
    if not fid:
        return None
    return download_file(token, fid)


def _put_file_bytes(token: Dict[str, Any], parent_id: str, name: str, data: bytes) -> str:
    fid = find_file(token, name, parent_id=parent_id)
    if fid:
        return update_file_bytes(token, fid, name, data, mime="application/zip")
    return upload_bytes(token, name, data, parent_id=parent_id, mime="application/zip")


# ---------- Helpers de zip/FAISS ----------

def _save_vectorstore_to_zip_bytes(vs: FAISS) -> bytes:
    """Salva FAISS em pasta temp e retorna um zip daquela pasta como bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "faiss_index")
        vs.save_local(index_dir)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(index_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    # mantém a subpasta 'faiss_index' dentro do zip
                    arcname = os.path.relpath(full, start=tmpdir)
                    z.write(full, arcname)
        return buf.getvalue()


def _load_vectorstore_from_zip_bytes(zip_bytes: bytes, embeddings: OpenAIEmbeddings) -> FAISS:
    """Carrega FAISS de bytes zipados — tolerante a zips antigos (com/sem pasta)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            z.extractall(tmpdir)

        # 1) Preferir diretórios (p.ex. faiss_index/)
        dirs = [p for p in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, p))]
        if dirs:
            # escolha a primeira pasta que contenha arquivos do índice
            for cand in dirs:
                cand_dir = os.path.join(tmpdir, cand)
                files = os.listdir(cand_dir)
                if any(fname.endswith((".faiss", ".pkl", ".json")) for fname in files):
                    return FAISS.load_local(cand_dir, embeddings, allow_dangerous_deserialization=True)

        # 2) Caso não haja diretório, tente raiz (zip antigo com arquivos na raiz)
        if any(fname.endswith((".faiss", ".pkl", ".json")) for fname in os.listdir(tmpdir)):
            return FAISS.load_local(tmpdir, embeddings, allow_dangerous_deserialization=True)

        # 3) Se nada bater, considere estrutura inesperada
        raise RuntimeError("Estrutura inesperada dentro do zip do FAISS.")


def _empty_index(embeddings: OpenAIEmbeddings) -> FAISS:
    return FAISS.from_texts(["__bootstrap__"], embeddings, metadatas=[{"bootstrap": True}])


def _load_or_create_index(token: Dict[str, Any], name: str, embeddings: OpenAIEmbeddings) -> Tuple[FAISS, bool]:
    """Carrega índice (.faiss.zip) ou cria vazio. Robusto contra zips antigos/corrompidos."""
    parent_id = _faiss_folder_id(token)
    bin_data = _get_file_bytes(token, parent_id, name)
    if bin_data is None:
        return _empty_index(embeddings), False
    try:
        return _load_vectorstore_from_zip_bytes(bin_data, embeddings), True
    except Exception:
        # não aborta: volta com índice vazio; próximo save sobrescreve
        return _empty_index(embeddings), False


def _save_index(token: Dict[str, Any], name: str, vs: FAISS) -> str:
    parent_id = _faiss_folder_id(token)
    data = _save_vectorstore_to_zip_bytes(vs)
    return _put_file_bytes(token, parent_id, name, data)


# ---------- API: upsert por doc + global ----------

def upsert_texts_to_drive_index(
    token: Dict[str, Any],
    doc_id: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upsert em **dois** índices: {doc_id}.faiss.zip e _global.faiss.zip.
    """
    if not texts:
        return {"added": 0}

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # enriquecer metadados
    now = int(time.time())
    metas = metadatas or [{} for _ in texts]
    enriched = []
    for m in metas:
        mm = dict(m or {})
        mm.setdefault("doc_id", doc_id)
        mm.setdefault("created_at", now)
        enriched.append(mm)

    # 1) índice do doc
    doc_name = _name_for_doc(doc_id)
    vs_doc, _ = _load_or_create_index(token, doc_name, embeddings)
    vs_doc.add_texts(texts=texts, metadatas=enriched)
    fid_doc = _save_index(token, doc_name, vs_doc)

    # 2) índice global
    vs_global, _ = _load_or_create_index(token, GLOBAL_INDEX_BASENAME, embeddings)
    vs_global.add_texts(texts=texts, metadatas=enriched)
    fid_glob = _save_index(token, GLOBAL_INDEX_BASENAME, vs_global)

    return {"doc_index_saved": fid_doc, "global_index_saved": fid_glob, "added": len(texts)}


# ---------- Reconstrução do global a partir dos docs ----------

def rebuild_global_from_all_docs(
    token: Dict[str, Any],
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reconstrói o índice global agregando TODOS os {doc_id}.faiss.zip (ignora _global).
    """
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    parent_id = _faiss_folder_id(token)

    all_doc_zips = list_files_in_folder(token, parent_id, name_suffix=".faiss.zip", limit=1000)
    all_doc_zips = [f for f in all_doc_zips if f["name"] != GLOBAL_INDEX_BASENAME]

    if not all_doc_zips:
        vs_global = _empty_index(embeddings)
        fid = _save_index(token, GLOBAL_INDEX_BASENAME, vs_global)
        return {"global_index_saved": fid, "sources": 0}

    merged: Optional[FAISS] = None
    sources = 0
    for f in all_doc_zips:
        data = download_file(token, f["id"])
        try:
            vs_doc = _load_vectorstore_from_zip_bytes(data, embeddings)
        except Exception:
            continue  # pula zips antigos/corrompidos
        if merged is None:
            merged = vs_doc
        else:
            merged.merge_from(vs_doc)
        sources += 1

    if merged is None:
        merged = _empty_index(embeddings)

    fid_glob = _save_index(token, GLOBAL_INDEX_BASENAME, merged)
    return {"global_index_saved": fid_glob, "sources": sources}

