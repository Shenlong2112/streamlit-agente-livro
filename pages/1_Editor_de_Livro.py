# pages/1_Editor_de_Livro.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st

from src.knowledge.repo import (
    get_doc,
    save_doc,
    append_version,
    TRANSCRICAO_DIR,
    VERSOES_DIR,
)
from src.storage.drive import (
    ensure_app_folder,
    ensure_subfolder,
    list_files_in_folder,
    download_file,   # ← novo: para baixar .txt
)
from langchain_openai import ChatOpenAI
from src.embeddings.faiss_drive import upsert_texts_to_drive_index

st.set_page_config(page_title="Editor de Livro", page_icon="📖", layout="wide")
st.title("📖 Editor de Livro")

# ---------- GUARDAS ----------
openai_key: Optional[str] = st.session_state.get("openai_key")
drive_token: Optional[Dict[str, Any]] = st.session_state.get("drive_token")

col_a, col_b = st.columns(2)
with col_a:
    st.caption("OpenAI (BYOK)")
    if openai_key:
        st.success("OPENAI_API_KEY carregada da sessão.")
    else:
        st.warning("Cole sua OPENAI_API_KEY na página **Conexões** para usar o editor.")

with col_b:
    st.caption("Google Drive")
    drive_ok = False
    if drive_token:
        try:
            _ = ensure_app_folder(drive_token)
            drive_ok = True
            st.success("Google Drive conectado.")
        except Exception as e:
            st.error(f"Drive não operacional: {e}")
    else:
        st.warning("Conecte o Google Drive na página **Conexões**.")

if not openai_key or not drive_token or not drive_ok:
    st.stop()

st.divider()

# ---------- APLICA PENDENTES ANTES DE CRIAR O WIDGET ----------
if "pending_texto_atual" in st.session_state:
    st.session_state["texto_atual_editor"] = st.session_state.pop("pending_texto_atual")

# ---------- SELEÇÃO / CRIAÇÃO DO DOCUMENTO (Manual) ----------
st.subheader("Documento")
with st.form("doc_selector", clear_on_submit=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        doc_id = st.text_input(
            "ID/nome do documento (ex.: capitulo-01-ou-titulo-do-audio)",
            value=st.session_state.get("current_doc_id", ""),
            placeholder="meu-livro-capitulo-01",
        )
    with c2:
        action = st.selectbox("Ação", ["Abrir", "Criar novo"], index=0)
    submitted = st.form_submit_button("Carregar")

bootstrap_from_transcricao = False

def _queue_current_text(text: str):
    """Enfileira para virar 'Texto atual' no próximo ciclo (antes do widget)."""
    st.session_state["pending_texto_atual"] = text

def _load_from_versoes(doc_id_str: str) -> bool:
    try:
        current_doc = get_doc(drive_token, doc_id_str, subfolder=VERSOES_DIR)
        st.session_state["current_doc"] = current_doc
        st.session_state["current_doc_subfolder"] = VERSOES_DIR
        st.session_state["current_doc_id"] = doc_id_str
        versions = current_doc.get("versions", [])
        _queue_current_text(versions[-1]["text"] if versions else "")
        st.success(f"Documento **{doc_id_str}** carregado de **versoes/**.")
        return True
    except FileNotFoundError:
        return False

def _bootstrap_from_transcricao(doc_id_str: str) -> bool:
    global bootstrap_from_transcricao
    try:
        trans_doc = get_doc(drive_token, doc_id_str, subfolder=TRANSCRICAO_DIR)
        text = trans_doc["versions"][-1]["text"] if trans_doc.get("versions") else ""
        st.session_state["current_doc"] = {"id": doc_id_str, "versions": []}
        st.session_state["current_doc_subfolder"] = VERSOES_DIR
        st.session_state["current_doc_id"] = doc_id_str
        st.session_state["bootstrap_text"] = text
        _queue_current_text(text)
        bootstrap_from_transcricao = True
        st.info("Documento não existe em **versoes/**. Carregado texto da **transcricao/** para iniciar a edição.")
        return True
    except FileNotFoundError:
        return False

if submitted and doc_id.strip():
    doc_id = doc_id.strip()
    st.session_state["current_doc_id"] = doc_id
    if not _load_from_versoes(doc_id):
        if action == "Criar novo":
            save_doc(drive_token, doc_id, "", {"created_at": int(datetime.now().timestamp())}, subfolder=VERSOES_DIR)
            st.session_state["current_doc"] = get_doc(drive_token, doc_id, subfolder=VERSOES_DIR)
            st.session_state["current_doc_subfolder"] = VERSOES_DIR
            _queue_current_text("")
            st.success(f"Documento **{doc_id}** criado em **versoes/**.")
        else:
            if not _bootstrap_from_transcricao(doc_id):
                st.error("Documento não encontrado em **versoes/** nem em **transcricao/**. "
                         "Se deseja criar, escolha **Criar novo** e envie novamente.")
    st.rerun()

# ---------- Picker do Google Drive ----------
with st.expander("📂 Carregar do Google Drive (picker)", expanded=False):
    root_id = ensure_app_folder(drive_token)
    id_versoes = ensure_subfolder(drive_token, root_id, "versoes")
    id_trans = ensure_subfolder(drive_token, root_id, "transcricao")

    fonte = st.radio("Escolha a fonte", ["versoes (revisado)", "transcricao (bruto)"], horizontal=True)

    if fonte.startswith("versoes"):
        tipo = st.radio("Tipo de arquivo", ["Manifesto (.json)", "Arquivos de versão (.txt)"], horizontal=True)
        if tipo == "Manifesto (.json)":
            lista = list_files_in_folder(drive_token, id_versoes, name_suffix=".json", limit=500)
            if not lista:
                st.caption("Nenhum manifesto (.json) em versoes/.")
            else:
                nomes = [f["name"] for f in lista]
                escolha = st.selectbox("Selecione um manifesto", nomes)
                if st.button("Carregar manifesto"):
                    doc_id_pick = escolha[:-5] if escolha.endswith(".json") else escolha
                    if not _load_from_versoes(doc_id_pick):
                        st.error("Manifesto listado não pôde ser carregado de **versoes/**.")
                    st.rerun()
        else:
            lista = list_files_in_folder(drive_token, id_versoes, name_suffix=".txt", limit=500)
            if not lista:
                st.caption("Nenhum arquivo de versão (.txt) em versoes/.")
            else:
                nomes = [f["name"] for f in lista]  # ex.: my-doc__editor-llm__003.txt
                escolha = st.selectbox("Selecione um arquivo de versão (.txt)", nomes)
                if st.button("Carregar .txt selecionado"):
                    # Deduz doc_id do padrão {doc_id}__{source}__{nnn}.txt
                    base = escolha[:-4] if escolha.endswith(".txt") else escolha
                    parts = base.split("__")
                    doc_id_pick = parts[0] if parts else base

                    # Baixa conteúdo e injeta como Texto atual
                    # Precisamos do file_id correspondente:
                    file_obj = next((f for f in lista if f["name"] == escolha), None)
                    if not file_obj:
                        st.error("Não foi possível localizar o arquivo selecionado.")
                    else:
                        file_id = file_obj["id"]
                        content = download_file(drive_token, file_id).decode("utf-8")

                        # Garantir que há um manifesto para esse doc_id (cria vazio se necessário)
                        try:
                            _ = get_doc(drive_token, doc_id_pick, subfolder=VERSOES_DIR)
                        except FileNotFoundError:
                            save_doc(drive_token, doc_id_pick, "", {"created_at": int(datetime.now().timestamp())}, subfolder=VERSOES_DIR)

                        # Posiciona contexto atual
                        st.session_state["current_doc_id"] = doc_id_pick
                        st.session_state["current_doc"] = get_doc(drive_token, doc_id_pick, subfolder=VERSOES_DIR)
                        st.session_state["current_doc_subfolder"] = VERSOES_DIR
                        _queue_current_text(content)
                        st.success(f"Carregado **{escolha}** como Texto atual (doc_id: `{doc_id_pick}`).")
                        st.rerun()
    else:
        lista = list_files_in_folder(drive_token, id_trans, name_suffix=".json", limit=500)
        if not lista:
            st.caption("Nenhum arquivo encontrado em transcricao/.")
        else:
            nomes = [f["name"] for f in lista]
            escolha = st.selectbox("Selecione um arquivo de transcrição (.json)", nomes)
            if st.button("Carregar transcrição"):
                doc_id_pick = escolha[:-5] if escolha.endswith(".json") else escolha
                ok = _bootstrap_from_transcricao(doc_id_pick)
                if not ok:
                    st.error("Arquivo listado não pôde ser carregado de **transcricao/**.")
                st.rerun()

# ---------- Se ainda não há doc selecionado, pare aqui ----------
if "current_doc" not in st.session_state:
    st.info("Selecione/crie um documento ou carregue algo pelo picker acima.")
    st.stop()

doc: Dict[str, Any] = st.session_state["current_doc"]
doc_id: str = st.session_state["current_doc_id"]
versions: List[Dict[str, Any]] = doc.get("versions", [])

# Inicializa o valor do textarea se necessário
if "texto_atual_editor" not in st.session_state:
    if st.session_state.get("bootstrap_text") and not versions:
        st.session_state["texto_atual_editor"] = st.session_state["bootstrap_text"]
    else:
        st.session_state["texto_atual_editor"] = versions[-1]["text"] if versions else ""

# ---------- HELPERS: salvar + indexar + version_tag (não toca no widget diretamente) ----------
def _next_version_meta(meta_base: Dict[str, Any]) -> Dict[str, Any]:
    try:
        existing = get_doc(drive_token, doc_id, subfolder=VERSOES_DIR)
        count = len(existing.get("versions", []))
    except FileNotFoundError:
        count = 0
    version_index = count + 1
    version_tag = f"{doc_id}_v{version_index}"
    out = dict(meta_base or {})
    out["version_index"] = version_index
    out["version_tag"] = version_tag
    return out

def _ensure_doc_in_versoes_with(text_for_first_save: str, meta: Dict[str, Any]):
    try:
        _ = get_doc(drive_token, doc_id, subfolder=VERSOES_DIR)
        return
    except FileNotFoundError:
        save_doc(drive_token, doc_id, text_for_first_save, meta, subfolder=VERSOES_DIR)

def _save_and_index(text: str, meta: Dict[str, Any]):
    meta = _next_version_meta(meta)
    _ensure_doc_in_versoes_with(text, meta)
    try:
        append_version(drive_token, doc_id, text, meta, subfolder=VERSOES_DIR)
    except FileNotFoundError:
        save_doc(drive_token, doc_id, text, meta, subfolder=VERSOES_DIR)

    # Atualiza índice agregado (doc-level)
    _ = upsert_texts_to_drive_index(
        drive_token, doc_id, [text], metadatas=[meta], openai_api_key=openai_key,
    )

    st.session_state["current_doc"] = get_doc(drive_token, doc_id, subfolder=VERSOES_DIR)
    st.session_state["pending_texto_atual"] = text
    st.session_state.pop("bootstrap_text", None)

# ---------- PAINEL PRINCIPAL ----------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Texto atual — `{doc_id}`")
    st.text_area(
        "Conteúdo (edição manual opcional)",
        value=st.session_state["texto_atual_editor"],
        height=420,
        key="texto_atual_editor",
    )
    if st.button("💾 Salvar edição como nova versão", type="primary", use_container_width=True):
        try:
            _save_and_index(st.session_state["texto_atual_editor"], {"source": "manual-edit"})
            st.success("Nova versão salva (arquivo .txt + manifesto JSON) e indexada no vecstore.")
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao salvar/indexar: {e}")

with right:
    st.subheader("Gerar nova versão (LLM)")

    with st.expander("Opções de estilo", expanded=True):
        estilo = st.text_input("Estilo (ex.: conciso, envolvente, técnico)", value="editorial claro, coeso e envolvente")
        audiencia = st.text_input("Audiência", value="leitores de não-ficção")
        tom = st.text_input("Tom", value="profissional e acessível")
        instrucoes = st.text_area(
            "Instruções específicas (opcional)",
            placeholder="Ex.: enfatizar exemplos; manter terminologia consistente; evitar jargões…",
        )

    if st.button("✍️ Gerar nova versão a partir do texto atual", use_container_width=True):
        base_text = st.session_state.get("texto_atual_editor", "")
        if not base_text:
            st.warning("Não há texto base. Edite à esquerda ou transcreva/importe um texto.")
        else:
            with st.spinner("Gerando versão com o LLM…"):
                try:
                    llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0.3)
                    system = (
                        "Você é um editor de livro tradicional. Seu trabalho é REVISAR e FORMATAR o texto do autor "
                        "para publicação. Preserve conteúdo factual, melhore clareza, coesão, ortografia e estilo. "
                        "Não invente informações externas. Use formatação mínima (títulos e subtítulos)."
                    )
                    user_prompt = f"""
Estilo: {estilo}
Audiência: {audiencia}
Tom: {tom}
Instruções: {instrucoes or "(nenhuma)"}

TAREFA: Reescreva o texto abaixo para publicação em livro, mantendo fidelidade ao conteúdo.

TEXTO:
\"\"\"{base_text}\"\"\"
"""
                    resp = llm.invoke(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user_prompt}]
                    )
                    new_text = resp.content.strip()
                except Exception as e:
                    st.error(f"Falha na geração com LLM: {e}")
                    new_text = ""

            if new_text:
                st.session_state["pending_texto_atual"] = new_text
                st.success("Versão gerada e aplicada ao **Texto atual**.")
                st.rerun()
            else:
                st.warning("Nada foi gerado.")

st.divider()

# ---------- HISTÓRICO DE VERSÕES (do manifesto JSON) ----------
doc_for_history: Optional[Dict[str, Any]] = None
try:
    doc_for_history = get_doc(drive_token, doc_id, subfolder=VERSOES_DIR)
except FileNotFoundError:
    doc_for_history = None

st.subheader("Histórico de versões")
if not doc_for_history or not doc_for_history.get("versions"):
    st.info("Ainda não há versões salvas em **versoes/** para este documento.")
else:
    versions = doc_for_history["versions"]
    for i, v in enumerate(reversed(versions), start=1):
        ts = v.get("ts")
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "s/ data"
        meta = v.get("meta", {})
        label = f"v{len(versions) - (i-1)} — {when} — {meta.get('source','?')} — {meta.get('version_tag','')}"
        with st.expander(label, expanded=False):
            texto_v = v.get("text", "")
            st.write(texto_v[:1000] + ("..." if len(texto_v) > 1000 else ""))
            cols = st.columns(2)

            with cols[0]:
                if st.button(f"💬 usar no editor (salvar+indexar) — {label}", key=f"use_{i}"):
                    try:
                        _save_and_index(texto_v, {"source": "picked"})
                        st.success("Versão promovida (txt + manifesto) e indexada (vecstore).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha: {e}")

            with cols[1]:
                if st.button(f"⭐ canônica (salvar+indexar) — {label}", key=f"canon_{i}"):
                    try:
                        _save_and_index(texto_v, {"source": "picked", "canonical": True})
                        st.success("Versão canônica salva (txt + manifesto) e indexada (vecstore).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha: {e}")

st.caption(
    "• Agora **cada versão** gera um **.txt próprio** em `versoes/` e também atualiza o **manifesto JSON** (com acentos legíveis). "
    "• O vecstore segue **agregado por documento** para consultas RAG sobre múltiplas versões. "
    "• Você também pode carregar `.txt` diretamente pelo picker de `versoes/`."
)





