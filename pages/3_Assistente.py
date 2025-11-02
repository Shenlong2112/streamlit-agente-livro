# pages/3_Assistente.py
from __future__ import annotations

import io
import os
import re
import json
import time
import tempfile
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

from src.storage.drive import (
    ensure_app_folder,
    ensure_subfolder,
    list_files_in_folder,
    find_file,
    download_file,
    upload_bytes,
    update_file_bytes,
)

from src.embeddings.faiss_drive import rebuild_global_from_all_docs
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI  # streaming/resumo
from duckduckgo_search import DDGS  # web opcional

# ---------------- UI & Const ----------------
st.set_page_config(page_title="Assistente (Memória + RAG + Web opcional)", page_icon="💬", layout="wide")
st.title("💬 Assistente (memória infinita + seu acervo | web só se você pedir)")

GLOBAL_ZIP_NAME = "_global.faiss.zip"
VERSOES_DIRNAME = "versoes"
VECSTORE_DIRNAME = "vecstore"
FAISS_DIRNAME = "faiss"
MEMORY_DIRNAME = "memory"

CHATS_DIRNAME = "chats"
CHATS_INDEX = "index.json"
HISTORY_FILE = "history.jsonl"
SUMMARY_FILE = "summary.txt"

# ---------------- Guards ----------------
openai_key: Optional[str] = st.session_state.get("openai_key")
drive_token: Optional[Dict[str, Any]] = st.session_state.get("drive_token")

c1, c2 = st.columns(2)
with c1:
    st.caption("OpenAI (BYOK)")
    if openai_key:
        st.success("OPENAI_API_KEY carregada da sessão.")
    else:
        st.warning("Cole sua OPENAI_API_KEY em **Conexões** para usar o assistente.")
with c2:
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
        st.warning("Conecte o Google Drive em **Conexões**.")
if not openai_key or not drive_token or not drive_ok:
    st.stop()

# ---------------- Helpers base ----------------
_slug_ok = re.compile(r"[^a-z0-9\-]+")
def slugify(s: str) -> str:
    from unidecode import unidecode
    if not s:
        return "x"
    s = unidecode(s).lower().strip()
    s = s.replace(" ", "-").replace("_", "-")
    s = _slug_ok.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"

def _drive_ids() -> Dict[str, str]:
    root = ensure_app_folder(drive_token)
    # knowledge vecstore
    vec_id = ensure_subfolder(drive_token, root, VECSTORE_DIRNAME)
    faiss_id = ensure_subfolder(drive_token, vec_id, FAISS_DIRNAME)
    mem_id = ensure_subfolder(drive_token, vec_id, MEMORY_DIRNAME)
    # versões (txt) e chats
    versoes_id = ensure_subfolder(drive_token, root, VERSOES_DIRNAME)
    chats_id = ensure_subfolder(drive_token, root, CHATS_DIRNAME)
    return {
        "root": root, "vec": vec_id, "faiss": faiss_id, "memory": mem_id,
        "versoes": versoes_id, "chats": chats_id
    }

def _download_text(parent_id: str, name: str) -> Optional[str]:
    fid = find_file(drive_token, name, parent_id=parent_id)
    if not fid:
        return None
    data = download_file(drive_token, fid)
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")

def _upload_or_update_text(parent_id: str, name: str, content: str, mime: str = "text/plain") -> str:
    fid = find_file(drive_token, name, parent_id=parent_id)
    data = content.encode("utf-8")
    if fid:
        return update_file_bytes(drive_token, fid, name, data, mime=mime)
    return upload_bytes(drive_token, name, data, parent_id=parent_id, mime=mime)

def _append_jsonl(parent_id: str, name: str, record: Dict[str, Any]) -> str:
    fid = find_file(drive_token, name, parent_id=parent_id)
    line = json.dumps(record, ensure_ascii=False)
    if fid:
        # baixa, anexa e atualiza
        raw = download_file(drive_token, fid).decode("utf-8", errors="ignore")
        raw2 = raw + ("\n" if (raw and not raw.endswith("\n")) else "") + line + "\n"
        return update_file_bytes(drive_token, fid, name, raw2.encode("utf-8"), mime="application/json")
    else:
        # cria
        body = line + "\n"
        return upload_bytes(drive_token, name, body.encode("utf-8"), parent_id=parent_id, mime="application/json")

def _read_history(parent_id: str) -> List[Dict[str, Any]]:
    txt = _download_text(parent_id, HISTORY_FILE)
    if not txt:
        return []
    out: List[Dict[str, Any]] = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

# ---------------- FAISS zip helpers (global & memory) ----------------
def _load_faiss_from_zip_bytes(zip_bytes: bytes, embeddings: OpenAIEmbeddings) -> FAISS:
    # tolerante a zip com/sem subpasta
    with tempfile.TemporaryDirectory() as tmpdir:
        import zipfile, os
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            z.extractall(tmpdir)
        candidates = [p for p in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, p))]
        index_dir = os.path.join(tmpdir, candidates[0]) if candidates else tmpdir
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

def _save_faiss_to_zip_bytes(vs: FAISS) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "faiss_index")
        vs.save_local(index_dir)
        buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(buf, "w") as z:
            for root, _, files in os.walk(index_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, start=tmpdir)
                    z.write(full, arc)
        return buf.getvalue()

@st.cache_data(show_spinner=True, ttl=600)
def _get_global_zip_bytes() -> bytes:
    """Garante que o global exista: tenta reconstruir; se nada houver, cria vazio."""
    ids = _drive_ids()
    fid = find_file(drive_token, GLOBAL_ZIP_NAME, parent_id=ids["faiss"])
    if not fid:
        try:
            rebuild_global_from_all_docs(drive_token, openai_api_key=openai_key)
            fid = find_file(drive_token, GLOBAL_ZIP_NAME, parent_id=ids["faiss"])
        except Exception:
            fid = None
    if not fid:
        # cria vazio bootstrap
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        empty_vs = FAISS.from_texts(["__bootstrap__"], embeddings, metadatas=[{"bootstrap": True}])
        data = _save_faiss_to_zip_bytes(empty_vs)
        upload_bytes(drive_token, GLOBAL_ZIP_NAME, data, parent_id=ids["faiss"], mime="application/zip")
        fid = find_file(drive_token, GLOBAL_ZIP_NAME, parent_id=ids["faiss"])
        if not fid:
            raise FileNotFoundError("Falha ao criar o índice global.")
    return download_file(drive_token, fid)

def _memory_zip_name(chat_id: str) -> str:
    return f"{chat_id}.faiss.zip"

def _load_or_create_memory_vs(chat_id: str) -> FAISS:
    ids = _drive_ids()
    fname = _memory_zip_name(chat_id)
    fid = find_file(drive_token, fname, parent_id=ids["memory"])
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    if not fid:
        return FAISS.from_texts(["__mem_bootstrap__"], embeddings, metadatas=[{"bootstrap": True, "chat_id": chat_id}])
    data = download_file(drive_token, fid)
    return _load_faiss_from_zip_bytes(data, embeddings)

def _save_memory_vs(chat_id: str, vs: FAISS) -> None:
    ids = _drive_ids()
    fname = _memory_zip_name(chat_id)
    data = _save_faiss_to_zip_bytes(vs)
    fid = find_file(drive_token, fname, parent_id=ids["memory"])
    if fid:
        update_file_bytes(drive_token, fid, fname, data, mime="application/zip")
    else:
        upload_bytes(drive_token, fname, data, parent_id=ids["memory"], mime="application/zip")

# ---------------- Web & LLM helpers ----------------
def _is_smalltalk(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.match(r"^(oi|ol[áa]|bom dia|boa tarde|boa noite|tudo bem|e a[ií])\b", t)) or len(t.split()) <= 2

def _wants_web(q: str) -> tuple[bool, str]:
    if not q:
        return False, q
    s = q.strip()
    low = s.lower()
    if low.startswith("/web ") or low == "/web":
        return True, s[5:].strip()
    if low.startswith("@web "):
        return True, s[5:].strip()
    triggers = [
        "pesquise na internet", "busque na internet", "pesquisa na internet",
        "na internet", "na web", "busque na web", "pesquise na web", "procure na web",
        "faça uma busca", "faz uma busca", "pesquise no google", "busque no google"
    ]
    if any(t in low for t in triggers):
        return True, s
    return False, s

def _web_search_ddg(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not query:
        return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "") or "(sem título)",
                    "href": r.get("href", "") or "",
                    "body": r.get("body", "") or "",
                })
    except Exception as e:
        st.warning(f"Falha na busca DuckDuckGo: {e}")
    return results

def _stream_llm(messages: List[Dict[str, str]], model: str, api_key: str):
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.2)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

def _summarize(previous_summary: str, new_turns: List[Dict[str, Any]]) -> str:
    """Atualiza resumo rolante com LLM (barato)."""
    client = OpenAI(api_key=openai_key)
    msgs = [
        {"role": "system", "content": "Você resume conversas para manter contexto longo. Seja conciso e factual."},
        {"role": "user", "content":
         f"RESUMO ATUAL:\n{previous_summary or '(vazio)'}\n\n"
         f"MENSAGENS RECENTES:\n" +
         "\n".join([f"- {t['role']}: {t['content']}" for t in new_turns]) +
         "\n\nAtualize o resumo de modo a reter decisões, preferências, glossário e objetivos. Máx ~200-250 palavras."}
    ]
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2)
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # fallback simples
        tail = " ".join(t.get("content","") for t in new_turns)[-800:]
        return (previous_summary + "\n" + tail)[-2000:]

# ---------------- Chats: index/list/create/load ----------------
def _ensure_chats_index() -> Tuple[str, List[Dict[str, Any]]]:
    ids = _drive_ids()
    chats_root = ids["chats"]
    fid = find_file(drive_token, CHATS_INDEX, parent_id=chats_root)
    if not fid:
        upload_bytes(drive_token, CHATS_INDEX, "[]".encode("utf-8"), parent_id=chats_root, mime="application/json")
        return chats_root, []
    data = download_file(drive_token, fid).decode("utf-8", errors="ignore")
    try:
        return chats_root, json.loads(data)
    except Exception:
        return chats_root, []

def _save_chats_index(chats_root: str, items: List[Dict[str, Any]]) -> None:
    body = json.dumps(items, ensure_ascii=False, indent=2)
    _upload_or_update_text(chats_root, CHATS_INDEX, body, mime="application/json")

def _create_chat(title: str) -> Dict[str, Any]:
    ids = _drive_ids()
    chats_root = ids["chats"]
    _, items = _ensure_chats_index()
    ts = int(time.time())
    chat_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{slugify(title) or 'chat'}"
    # cria pasta do chat
    chat_dir_id = ensure_subfolder(drive_token, chats_root, chat_id)
    # cria arquivos vazios
    _upload_or_update_text(chat_dir_id, SUMMARY_FILE, "", mime="text/plain")
    _upload_or_update_text(chat_dir_id, HISTORY_FILE, "", mime="application/json")
    # registra no índice
    rec = {"id": chat_id, "title": title, "created_at": ts, "updated_at": ts}
    # se já existe, atualiza
    items = [it for it in items if it.get("id") != chat_id] + [rec]
    _save_chats_index(chats_root, items)
    return rec

def _list_chats() -> List[Dict[str, Any]]:
    _, items = _ensure_chats_index()
    # ordenar por updated_at desc
    return sorted(items, key=lambda x: x.get("updated_at", 0), reverse=True)

def _get_chat_dir_id(chat_id: str) -> str:
    ids = _drive_ids()
    return ensure_subfolder(drive_token, ids["chats"], chat_id)

def _touch_chat_updated(chat_id: str) -> None:
    chats_root, items = _ensure_chats_index()
    now = int(time.time())
    changed = False
    for it in items:
        if it.get("id") == chat_id:
            it["updated_at"] = now
            changed = True
            break
    if changed:
        _save_chats_index(chats_root, items)

# ---------------- Barra de ferramentas geral ----------------
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    k_mem = st.slider("Top-K memória", 1, 8, 4, 1, help="Quantos trechos da memória do chat recuperar")
with colB:
    k_global = st.slider("Top-K acervo", 2, 12, 6, 1, help="Quantos trechos do knowledge global recuperar")
with colC:
    if st.button("♻️ Recarregar índice global"):
        _get_global_zip_bytes.clear()
        st.success("Cache do índice global limpo.")
        st.rerun()

st.caption("• O assistente usa **memória do chat** + **seu acervo global**. Para web, comece com `/web ...` (DuckDuckGo).")
st.divider()

# ---------------- Seletor de chat ----------------
existing = _list_chats()
labels = [f"{it['title']} — {time.strftime('%Y-%m-%d %H:%M', time.localtime(it.get('updated_at', 0)))}" for it in existing]
choices = ["➕ Novo chat..."] + labels

default_idx = 0
if "chat_id" in st.session_state and st.session_state["chat_id"]:
    # tenta posicionar no chat atual
    for idx, it in enumerate(existing):
        if it.get("id") == st.session_state.get("chat_id"):
            default_idx = idx + 1
            break

sel = st.selectbox("Conversa", choices, index=default_idx)
creating_new = (sel == "➕ Novo chat...")

if creating_new:
    tcol1, tcol2 = st.columns([3,1])
    with tcol1:
        new_title = st.text_input("Título do novo chat", value="Meu chat")
    with tcol2:
        if st.button("Criar"):
            rec = _create_chat(new_title.strip() or "Meu chat")
            st.session_state["chat_id"] = rec["id"]
            st.session_state["chat_title"] = rec["title"]
            # zera histórico em sessão
            st.session_state["global_chat_messages"] = []
            st.session_state["memory_vs"] = None
            st.rerun()
else:
    # map index -> chat_id
    idx = labels.index(sel)
    chosen = existing[idx]
    st.session_state["chat_id"] = chosen["id"]
    st.session_state["chat_title"] = chosen["title"]

chat_id: Optional[str] = st.session_state.get("chat_id")
if not chat_id:
    st.stop()

# ---------------- Carregar memória do chat e histórico ----------------
chat_dir_id = _get_chat_dir_id(chat_id)

# carrega summary
summary_text = _download_text(chat_dir_id, SUMMARY_FILE) or ""

# carrega histórico do Drive -> render na UI
history = _read_history(chat_dir_id)

if "global_chat_messages" not in st.session_state or not st.session_state["global_chat_messages"]:
    # popular sessão a partir do Drive
    st.session_state["global_chat_messages"] = [{"role": h["role"], "content": h["content"]} for h in history]

# render histórico
for msg in st.session_state["global_chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# carrega vecstore de memória (em memória de sessão)
if "memory_vs" not in st.session_state or st.session_state["memory_vs"] is None:
    try:
        st.session_state["memory_vs"] = _load_or_create_memory_vs(chat_id)
    except Exception as e:
        st.warning(f"Falha ao carregar memória vetorial: {e}")
        st.session_state["memory_vs"] = _load_or_create_memory_vs(chat_id)  # tenta vazio

# carrega vecstore global
try:
    zipped = _get_global_zip_bytes()
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    global_vs = _load_faiss_from_zip_bytes(zipped, embeddings)
except Exception as e:
    st.error(f"Falha ao carregar o índice global: {e}")
    st.stop()

mem_retriever = st.session_state["memory_vs"].as_retriever(
    search_type="mmr", search_kwargs={"k": k_mem, "fetch_k": max(8, 2 * k_mem), "lambda_mult": 0.5}
)
global_retriever = global_vs.as_retriever(
    search_type="mmr", search_kwargs={"k": k_global, "fetch_k": max(12, 2 * k_global), "lambda_mult": 0.5}
)

# ---------------- Entrada do usuário ----------------
user_input = st.chat_input("Fale comigo — uso sua memória + acervo; para web, use /web ...")
if user_input:
    # 1) Anexa turno do usuário em Drive (histórico)
    user_turn = {"ts": int(time.time()), "role": "user", "content": user_input}
    _append_jsonl(chat_dir_id, HISTORY_FILE, user_turn)

    # 2) Exibe no chat
    st.session_state["global_chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3) Decide uso de web
    use_web, clean_query = _wants_web(user_input)

    # 4) Recuperação: memória do chat + acervo global (a menos que smalltalk pura)
    mem_docs: List[Any] = []
    acervo_docs: List[Any] = []
    used_mem = False
    used_acervo = False
    web_results: List[Dict[str, str]] = []

    if not _is_smalltalk(clean_query):
        try:
            mem_docs = mem_retriever.get_relevant_documents(clean_query)
            used_mem = len(mem_docs) > 0
        except Exception as e:
            st.warning(f"Não consegui recuperar da memória: {e}")
            mem_docs = []
            used_mem = False

        try:
            acervo_docs = global_retriever.get_relevant_documents(clean_query)
            used_acervo = len(acervo_docs) > 0
        except Exception as e:
            st.warning(f"Não consegui recuperar do acervo: {e}")
            acervo_docs = []
            used_acervo = False

    # 5) Web se explicitamente pedido
    if use_web:
        with st.spinner("Buscando na web…"):
            web_results = _web_search_ddg(clean_query, max_results=6)

    # 6) Montar contextos (M# para memória, A# para acervo, W# para web)
    mem_blocks, mem_refs = [], []
    for i, d in enumerate(mem_docs, start=1):
        meta = d.metadata or {}
        turn_idx = meta.get("turn_index")
        ts = meta.get("ts")
        mem_blocks.append(f"[M{i}] {d.page_content}")
        mem_refs.append(f"[M{i}] turn={turn_idx if turn_idx is not None else '?'} • ts={ts if ts else ''}")

    acervo_blocks, acervo_refs = [], []
    for i, d in enumerate(acervo_docs, start=1):
        meta = d.metadata or {}
        doc_id = meta.get("doc_id") or meta.get("source") or "doc"
        vtag = meta.get("version_tag") or ""
        src = meta.get("source") or ""
        acervo_blocks.append(f"[A{i}] {d.page_content}")
        tail = f"{' • ' + vtag if vtag else ''}"
        acervo_refs.append(f"[A{i}] {doc_id}{tail}")

    web_blocks, web_citations = [], []
    if web_results:
        for j, r in enumerate(web_results, start=1):
            title = r["title"]
            url = r["href"]
            snippet = r["body"]
            web_blocks.append(f"[W{j}] {title}\nURL: {url}\nResumo: {snippet}")
            web_citations.append(f"[W{j}] [{title}]({url})" if url else f"[W{j}] {title}")

    # 7) Mensagens para o LLM (inclui SUMMARY)
    system_parts = [
        "Você é um assistente editorial.",
        "Use primeiro a MEMÓRIA do chat ([M#]) para manter coerência de longo prazo.",
        "Use o ACERVO global ([A#]) para conteúdo factual do conhecimento do usuário.",
        "Use WEB ([W#]) somente quando o usuário pedir explicitamente.",
        "Cite as fontes com [M#], [A#] e [W#]. Seja objetivo.",
    ]
    system = "\n".join(system_parts)

    blocks: List[str] = []
    if summary_text:
        blocks.append("RESUMO DO CHAT:\n" + summary_text)
    if used_mem:
        blocks.append("MEMÓRIA (trechos relevantes):\n" + "\n\n".join(mem_blocks))
    if used_acervo:
        blocks.append("ACERVO (trechos relevantes):\n" + "\n\n".join(acervo_blocks))
    if web_blocks:
        blocks.append("WEB (pedido explícito):\n" + "\n\n".join(web_blocks))
    blocks_text = "\n\n".join(blocks) if blocks else ""

    if used_mem or used_acervo or web_blocks:
        user_for_llm = (
            f"PERGUNTA: {clean_query}\n\n"
            f"{blocks_text}\n\n"
            "INSTRUÇÕES:\n"
            "- Priorize a coerência com a memória [M#].\n"
            "- Use o acervo [A#] para fundamentar fatos do conhecimento do usuário.\n"
            "- Use a web [W#] apenas porque foi solicitada explicitamente.\n"
            "- Cite as fontes com [M#], [A#], [W#]."
        )
    else:
        # smalltalk/sem contexto
        user_for_llm = clean_query

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_for_llm}]

    # 8) Resposta com streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            for token in _stream_llm(messages, model="gpt-4o-mini", api_key=openai_key):
                acc += token
                placeholder.markdown(acc)
        except Exception as e:
            acc = f"Falha ao gerar resposta: {e}"
            placeholder.markdown(acc)

        # rodapé
        footers = []
        if mem_refs:
            footers.append("Memória: " + "  •  ".join(mem_refs))
        if acervo_refs:
            footers.append("Acervo: " + "  •  ".join(acervo_refs))
        if web_citations:
            footers.append("Web: " + "  •  ".join(web_citations))
        if footers:
            st.caption("\n\n".join(footers))
        elif not (used_mem or used_acervo or web_blocks):
            st.caption("Sem memória/acervo/web usados nesta resposta.")

    # 9) Persistência do turno do assistente
    asst_turn = {"ts": int(time.time()), "role": "assistant", "content": acc}
    _append_jsonl(chat_dir_id, HISTORY_FILE, asst_turn)
    _touch_chat_updated(chat_id)

    # 10) Atualiza sessão
    st.session_state["global_chat_messages"].append({"role": "assistant", "content": acc})

    # 11) Atualiza resumo rolante (usa últimas N msgs + resumo anterior)
    # carrega últimas 6-10 do histórico para resumo
    latest_history = _read_history(chat_dir_id)[-8:]
    try:
        new_summary = _summarize(summary_text, latest_history)
        summary_text = new_summary
        _upload_or_update_text(chat_dir_id, SUMMARY_FILE, summary_text, mime="text/plain")
    except Exception as e:
        st.warning(f"Falha ao atualizar resumo: {e}")

    # 12) Upsert na memória vetorial deste chat (apenas os 2 últimos turnos)
    try:
        mem_vs: FAISS = st.session_state["memory_vs"]
        turn_index_base = len(_read_history(chat_dir_id))  # depois de gravar 2 linhas; usa como hint
        texts = [user_turn["content"], asst_turn["content"]]
        metas = [
            {"chat_id": chat_id, "role": "user", "turn_index": turn_index_base - 1, "ts": user_turn["ts"]},
            {"chat_id": chat_id, "role": "assistant", "turn_index": turn_index_base, "ts": asst_turn["ts"]},
        ]
        mem_vs.add_texts(texts=texts, metadatas=metas)
        _save_memory_vs(chat_id, mem_vs)
    except Exception as e:
        st.warning(f"Falha ao atualizar memória vetorial: {e}")


