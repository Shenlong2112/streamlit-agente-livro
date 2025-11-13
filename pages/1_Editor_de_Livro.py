# pages/1_Editor_de_Livro.py
from __future__ import annotations

import io
import os
import zipfile
import tempfile
from datetime import datetime
from typing import List, Dict

import streamlit as st

# Drive utils
from src.storage.drive import (
    drive_service_from_token,
    list_files_md,
    download_text,
    upload_text,
    find_or_create_folder,
    ensure_subfolder,
    upload_binary,
)

# Vetor/embeddings (suas funções já existentes)
from src.embeddings.vectorstore_faiss import (
    create_faiss_index,
    save_faiss_index,
)

# Repo helpers/constantes (novos)
from src.knowledge.repo import (
    ensure_user_tree,
    list_texts_in_folder,
    download_text_file,
    save_new_version_text,
    TRANSCRICAO_DIR,
    VERSOES_DIR,
    VECSTORE_DIR,
)

# Opcional: LLM para "gerar nova versão" (usa sua chave BYOK)
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # evita quebrar se faltar dependência; você pode instalar depois


# ==========================
# Helpers locais
# ==========================
def _pack_dir_to_zip_bytes(path: str) -> bytes:
    """Compacta o diretório 'path' para bytes .zip (na memória)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path)
                zf.write(full, rel)
    return buf.getvalue()


def _ensure_folders_and_ids(service) -> Dict[str, str]:
    """
    Garante a árvore de pastas e devolve IDs.
    {'root', 'trans', 'versions', 'vec'}
    """
    return ensure_user_tree(service)


def _load_text_from_choice(service, folder_id: str) -> str:
    """UI para escolher um arquivo e retornar o texto dele."""
    files = list_texts_in_folder(service, folder_id)
    if not files:
        st.info("Nenhum arquivo encontrado nesta pasta.")
        return ""

    names = [f["name"] for f in files]
    idx = st.selectbox("Escolha um arquivo", range(len(names)), format_func=lambda i: names[i], key=f"pick_{folder_id}")
    file_id = files[idx]["id"]
    return download_text_file(service, file_id)


def _get_llm():
    """Instancia um LLM para gerar nova versão (se langchain_openai estiver disponível)."""
    if not st.session_state.get("OPENAI_API_KEY"):
        return None
    if ChatOpenAI is None:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=st.session_state["OPENAI_API_KEY"])


def _rewrite_text_with_instructions(llm, raw_text: str, estilo: str, audiencia: str, instrucoes: str) -> str:
    """Gera uma nova versão a partir do texto base e instruções do usuário."""
    if llm is None:
        # fallback: não gera, apenas devolve o próprio texto
        return raw_text

    sys = (
        "Você é um editor de livros tradicional. Sua tarefa:\n"
        "- Corrigir gramática e ortografia.\n"
        "- Organizar parágrafos e tornar o texto claro e fluente.\n"
        "- Não invente conteúdo que não exista no texto original; mantenha fidelidade.\n"
        "- Respeite estilo/audiência se fornecidos.\n"
    )
    usr = (
        f"ESTILO: {estilo or 'padrão'}\n"
        f"AUDIÊNCIA: {audiencia or 'geral'}\n"
        f"INSTRUÇÕES EXTRAS: {instrucoes or 'nenhuma'}\n\n"
        "TEXTO BASE:\n"
        f"{raw_text}"
    )
    # prompt simple
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    out = llm.invoke(msgs)  # langchain_openai API
    return getattr(out, "content", "") or ""


# ==========================
# Página
# ==========================
st.set_page_config(page_title="Editor de Livro", page_icon="📝", layout="wide")
st.title("📝 Editor de Livro")

# Requisitos de conexão
if not st.session_state.get("google_connected") or not st.session_state.get("google_token"):
    st.warning("Conecte primeiro o Google Drive em **Conexões**.")
    st.stop()

service = drive_service_from_token(st.session_state["google_token"])

# Garante/obtém IDs de pastas
ids = _ensure_folders_and_ids(service)
root_id = ids["root"]
trans_id = ids["trans"]
vers_id = ids["versions"]
vec_id = ids["vec"]

# Estado UI
st.session_state.setdefault("texto_atual_editor", "")
st.session_state.setdefault("titulo_base", "")
st.session_state.setdefault("versao_gerada", "")

# Colunas: escolha de fonte (transcrição, versões, outro) + blocos de edição/ações
col_left, col_right = st.columns([0.4, 0.6])

with col_left:
    st.subheader("📂 Selecionar fonte")
    fonte = st.radio(
        "De onde buscar o texto?",
        options=["Transcrições", "Versões"],
        horizontal=True,
    )

    if fonte == "Transcrições":
        st.caption(f"Pasta: {TRANSCRICAO_DIR}")
        txt = _load_text_from_choice(service, trans_id)
    else:
        st.caption(f"Pasta: {VERSOES_DIR}")
        txt = _load_text_from_choice(service, vers_id)

    if txt:
        # Define o texto atual e título base (primeira linha)
        st.session_state["texto_atual_editor"] = txt
        primeira_linha = txt.split("\n", 1)[0].strip()
        st.session_state["titulo_base"] = primeira_linha or f"texto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.success("Texto carregado para edição.")

with col_right:
    st.subheader("✏️ Texto atual")
    st.caption("Edite livremente. Ao salvar, o arquivo irá para **Versões** e também gerará embeddings.")

    # Campo de edição (mantém chave fixa!)
    edited = st.text_area(
        "Conteúdo",
        value=st.session_state["texto_atual_editor"],
        height=420,
        key="texto_atual_editor",
    )

    # Seção: gerar nova versão com instruções
    st.markdown("---")
    st.subheader("🧠 Gerar nova versão (opcional)")
    colA, colB = st.columns(2)
    with colA:
        estilo = st.text_input("Estilo desejado (opcional)", value="")
    with colB:
        audiencia = st.text_input("Audiência (opcional)", value="")

    instrucoes = st.text_area("Instruções adicionais (opcional)", value="", height=120)

    if st.button("Gerar nova versão a partir do texto atual"):
        llm = _get_llm()
        novo = _rewrite_text_with_instructions(llm, edited, estilo, audiencia, instrucoes)
        if not novo.strip():
            st.error("Não foi possível gerar nova versão (verifique a chave OpenAI ou tente novamente).")
        else:
            # Atualiza o bloco "Texto atual" imediatamente
            # Importante: use st.session_state.update em vez de reatribuir após o widget existir
            st.session_state.update({"texto_atual_editor": novo})
            st.toast("Nova versão gerada e aplicada ao Texto atual.", icon="✍️")
            st.rerun()

    st.markdown("---")
    # Título base (o nome deriva daqui)
    titulo_sugerido = st.text_input("Título base para salvar (opcional)", value=st.session_state.get("titulo_base") or "")

    if st.button("💾 Salvar edição como nova versão (+ embeddings)"):
        texto_para_salvar = st.session_state.get("texto_atual_editor", "").strip()
        if not texto_para_salvar:
            st.error("Nada para salvar.")
        else:
            base_title = titulo_sugerido or "versao"
            # 1) Salva .txt como nova versão
            file_id, filename = save_new_version_text(service, vers_id, base_title, texto_para_salvar, add_suffix_version=True)

            # 2) Gera embeddings/FAISS Apenas desta versão e envia um ZIP para pasta Vecstore
            try:
                index = create_faiss_index([texto_para_salvar])
                with tempfile.TemporaryDirectory() as tmpdir:
                    save_faiss_index(index, tmpdir)  # salva a estrutura local do FAISS
                    data = _pack_dir_to_zip_bytes(tmpdir)
                # Nome do pacote de embeddings correspondente a esta versão
                emb_name = os.path.splitext(filename)[0] + ".faiss.zip"
                _ = upload_binary(service, vec_id, emb_name, data, mimetype="application/zip")
            except Exception as e:
                st.warning(f"Versão salva, mas houve falha ao gerar/enviar embeddings: {e}")

            st.success(f"Versão salva como **{filename}** em {VERSOES_DIR}.")
            st.toast("Embeddings enviados para Vecstore (pacote .zip).", icon="🧩")
            # atualiza título base para o próximo save
            st.session_state["titulo_base"] = os.path.splitext(filename)[0]






