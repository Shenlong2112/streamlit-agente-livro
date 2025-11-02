# pages/0_Conexoes.py
from __future__ import annotations

import time
import streamlit as st

from src.storage.drive import (
    get_auth_url,
    handle_oauth_callback,
    drive_me,
    ensure_app_folder,
)

st.set_page_config(page_title="Conexões", page_icon="🔌")

# ---- sessão: chaves padrão (evita perder entre reruns) ----
st.session_state.setdefault("openai_key", "")
st.session_state.setdefault("drive_token", None)

st.title("🔌 Conexões")

# ----------------------------
# OPENAI BYOK (chave do usuário)
# ----------------------------
st.subheader("OpenAI – sua própria chave (BYOK)")
st.caption("A chave é mantida apenas na sessão do app (não salvamos em disco).")

openai_key_input = st.text_input(
    "OPENAI_API_KEY",
    value=st.session_state.get("openai_key", ""),
    type="password",
    placeholder="cole sua chave aqui",
)
# Atualiza sessão sempre que o valor mudar
if openai_key_input != st.session_state["openai_key"]:
    st.session_state["openai_key"] = openai_key_input

if st.session_state["openai_key"]:
    st.success("Chave armazenada na sessão.")

st.divider()

# ----------------------------
# GOOGLE DRIVE (OAuth)
# ----------------------------
st.subheader("Google Drive")

# Trata retorno do OAuth (code nos query params) ANTES de mostrar o botão
query_params = st.query_params
if "code" in query_params and not st.session_state.get("drive_token"):
    try:
        token = handle_oauth_callback(query_params.get("code"))
        st.session_state["drive_token"] = token
        st.success("Google Drive conectado com sucesso!")

        # Mostra quem é o usuário e valida pasta do app
        me = drive_me(st.session_state["drive_token"])
        st.info(f"Conectado como: **{me.get('emailAddress', 'desconhecido')}**")
        folder_id = ensure_app_folder(st.session_state["drive_token"])
        st.caption(f"Pasta do app pronta (id: {folder_id})")

        # Redireciona para o Editor, mantendo a MESMA aba
        st.info("Redirecionando para o Editor…")
        time.sleep(0.6)
        try:
            st.switch_page("pages/1_Editor_de_Livro.py")
        except Exception:
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Falha ao concluir a conexão com o Google Drive: {e}")

# Se ainda não há token, mostra o botão para iniciar o OAuth
if not st.session_state.get("drive_token"):
    auth_url = get_auth_url()
    # Força abrir/voltar NA MESMA ABA (evita perder a session_state)
    st.markdown(
        f'''
        <a href="{auth_url}" target="_self">
            <button style="padding:0.6rem 1rem; font-size:1rem;">Conectar Google Drive</button>
        </a>
        ''',
        unsafe_allow_html=True,
    )
    st.caption(
        "Se aparecer erro 400 de redirect, confira no Google Cloud Console os "
        "Authorized redirect URIs e inclua: "
        "`http://localhost:8501/Conexoes`, `http://localhost:8501/Conexoes/`, "
        "`http://127.0.0.1:8501/Conexoes`, `http://127.0.0.1:8501/Conexoes/`."
    )
else:
    # Já conectado: exibe status
    try:
        me = drive_me(st.session_state["drive_token"])
        st.success(f"Google Drive já conectado: **{me.get('emailAddress', 'desconhecido')}**")
        folder_id = ensure_app_folder(st.session_state["drive_token"])
        st.caption(f"Pasta do app pronta (id: {folder_id})")
    except Exception as e:
        st.error(f"Token inválido/expirado. Reconecte. Detalhe: {e}")
        st.session_state["drive_token"] = None

st.divider()

st.caption(
    "Após conectar o Google Drive e informar sua OPENAI_API_KEY, acesse o **Editor de Livro**. "
    "Observação: este ambiente local usa `http://localhost:8501/Conexoes` como Redirect URI."
)

