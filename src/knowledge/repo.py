# src/knowledge/repo.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from unidecode import unidecode

# Funções utilitárias do Drive (já existentes no seu projeto)
from src.storage.drive import (
    drive_service_from_token,
    find_or_create_folder,
    ensure_subfolder,
    list_files_in_folder,
    list_files_md,
    upload_text,
    download_text,
    upload_binary,
    download_binary,
    update_file_contents,
    safe_delete,
)

# ============================
# Constantes de estrutura
# ============================
ROOT_DIR_NAME: str = "AgenteLivro"
TRANSCRICAO_DIR: str = "Transcricoes"   # sem acentos para evitar surpresas
VERSOES_DIR: str = "Versoes"
VECSTORE_DIR: str = "Vecstore"          # pasta separada para os índices/embeddings


# ============================
# Helpers de nome/título/arquivo
# ============================
def _slugify(title: str) -> str:
    """Gera um 'slug' seguro para nomes de arquivos"""
    t = unidecode(title or "").strip()
    # primeira linha como título, se for um texto longo
    if "\n" in t:
        t = t.split("\n", 1)[0].strip()
    t = t.lower()
    t = re.sub(r"[^a-z0-9\-_. ]+", "", t)
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"_+", "_", t)
    return t[:80] or "sem_titulo"


def build_version_filename(base_title: str, suffix: Optional[str] = None) -> str:
    """Cria um nome de arquivo único para versões, com timestamp opcional."""
    slug = _slugify(base_title)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{slug}_{suffix}_{ts}.txt"
    return f"{slug}_{ts}.txt"


# ============================
# Árvore de pastas do usuário
# ============================
def ensure_user_tree(service) -> Dict[str, str]:
    """
    Garante a árvore padrão no Google Drive do usuário e retorna os IDs.
    Retorno: {"root": ..., "trans": ..., "versions": ..., "vec": ...}
    """
    root_id = find_or_create_folder(service, ROOT_DIR_NAME)
    trans_id = ensure_subfolder(service, root_id, TRANSCRICAO_DIR)
    versions_id = ensure_subfolder(service, root_id, VERSOES_DIR)
    vec_id = ensure_subfolder(service, root_id, VECSTORE_DIR)
    return {"root": root_id, "trans": trans_id, "versions": versions_id, "vec": vec_id}


# ============================
# Ações comuns do editor/transcritor
# ============================
def list_texts_in_folder(service, folder_id: str) -> List[Dict[str, str]]:
    """Lista arquivos .txt (ou .md) em uma pasta, ordenados por modificação decrescente."""
    return list_files_md(service, folder_id, extensions=[".txt", ".md"])


def download_text_file(service, file_id: str) -> str:
    """Baixa conteúdo de texto de um file_id."""
    return download_text(service, file_id)


def save_new_version_text(service, versions_folder_id: str, base_title: str, text: str, add_suffix_version: bool = True) -> Tuple[str, str]:
    """
    Salva um novo arquivo de versão de texto e retorna (file_id, filename).
    Se add_suffix_version=True, acrescenta algo como '_v' no nome antes do timestamp.
    """
    suffix = "v" if add_suffix_version else None
    filename = build_version_filename(base_title, suffix=suffix)
    file_id = upload_text(service, versions_folder_id, filename, text)
    return file_id, filename


