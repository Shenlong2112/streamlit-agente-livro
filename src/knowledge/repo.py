# src/knowledge/repo.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from src.storage.drive import find_or_create_folder, ensure_subfolder

# Nome do diretório raiz do app no Drive
ROOT_DIR_NAME = "Agente_Livro"

# Subpastas
TRANSCRICAO_DIR = "Transcricoes"   # áudios transcritos (bruto)
VERSOES_DIR = "Versoes"            # versões revisadas/salvas pelo Editor
VECSTORE_DIR = "Vecstore"          # pacotes .faiss.zip (embeddings)
REFERENCIAS_DIR = "Referencias"    # <<< NOVO: PDFs e textos derivados de PDFs

def ensure_user_tree(service) -> Dict[str, str]:
    """
    Garante a árvore de pastas do usuário no Drive e retorna os IDs.
    """
    root_id = find_or_create_folder(service, ROOT_DIR_NAME)
    trans_id = ensure_subfolder(service, root_id, TRANSCRICAO_DIR)
    versions_id = ensure_subfolder(service, root_id, VERSOES_DIR)
    vec_id = ensure_subfolder(service, root_id, VECSTORE_DIR)
    refs_id = ensure_subfolder(service, root_id, REFERENCIAS_DIR)  # novo

    return {
        "root": root_id,
        "trans": trans_id,
        "versions": versions_id,
        "vec": vec_id,
        "refs": refs_id,  # novo
    }

def build_version_filename(base_title: str, suffix: Optional[str] = None) -> str:
    """
    Gera um nome de arquivo com timestamp. Ex.: "capitulo_1_20250101_121314.txt"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{base_title}_{suffix}_{ts}.txt"
    return f"{base_title}_{ts}.txt"


