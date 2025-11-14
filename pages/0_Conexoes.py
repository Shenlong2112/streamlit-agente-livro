# pages/0_Conexoes.py
from __future__ import annotations

import os
import streamlit as st

from src.storage.drive import (
    get_auth_url,
    exchange_code_for_token,
    drive_service_from_token,
)

st.set_page_config(page_title="Conexões", page_icon="🔌", layout="wide")
st.title("🔌 Conexões")

# =========================
# OPENAI (BYOK) — Somente UI
# =========================
st.subheader("OpenAI — **sua** própria chave (BYOK)")
st.caption("Sua chave é usada apenas no seu navegador/sessão (não é salva no servidor).")

openai_key_input = st.text_input(
    "OPENAI_API_KEY",
    type="password",
    placeholder="cole aqui sua chave da OpenAI",
    value=st.session_state.get("OPENAI_API_KEY", ""),
    help="Acesse https://platform.openai.com/ para criar sua chave.",
)
col_save, col_clear = st.columns([1, 1])
with col_save:
    if st.button("Salvar chave na sessão", use_container_width=True):
        if openai_key_input.strip():
            st.session_state["OPENAI_API_KEY"] = openai_key_input.strip()
            os.environ["OPENAI_API_KEY"] = openai_key_input.strip()
            st.success("Chave salva para esta sessão.")
        else:
            st.warning("Cole uma chave válida antes de salvar.")
with col_clear:
    if st.button("Limpar chave da sessão", use_container_width=True):
        st.session_state.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        st.info("Chave removida desta sessão.")

st.markdown("---")

# =========================
# Google Drive (OAuth)
# =========================
st.subheader("Google Drive")
st.caption("Conecte seu Drive para armazenar transcrições, versões e o vecstore.")

# Trata retorno do OAuth (código na URL)
qs = st.query_params
if "code" in qs:
    code = qs.get("code")
    try:
        token = exchange_code_for_token(code)
        st.session_state["google_token"] = token
        st.session_state["google_connected"] = True
        # limpa querystring e recarrega a própria página
        st.query_params.clear()
        st.success("Google Drive conectado com sucesso.")
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao finalizar OAuth: {e}")

# Estado atual
connected = st.session_state.get("google_connected") and st.session_state.get("google_token")
if connected:
    st.success("Google Drive **conectado**.")
    # Teste leve do serviço (não exibe nada sensível)
    try:
        _ = drive_service_from_token(st.session_state["google_token"])
    except Exception as e:
        st.warning(f"Conectado, mas houve um aviso ao inicializar o serviço: {e}")

    if st.button("Desconectar Google Drive", use_container_width=True):
        st.session_state.pop("google_token", None)
        st.session_state["google_connected"] = False
        st.info("Conexão removida desta sessão.")
else:
    st.info("Você ainda não conectou o Google Drive.")
    try:
        auth_url = get_auth_url()
        # Link abre na mesma aba usando HTML simples
        st.markdown(
            f'<a href="{auth_url}" target="_self" class="stButton"><button style="width:100%">Conectar Google Drive</button></a>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(
            "Não foi possível gerar a URL de autenticação. "
            "Verifique **GOOGLE_CLIENT_ID**, **GOOGLE_CLIENT_SECRET** e **GOOGLE_REDIRECT_URI** em *App settings → Secrets*."
        )
        st.exception(e)




