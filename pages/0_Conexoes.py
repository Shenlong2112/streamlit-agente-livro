# pages/0_Conexoes.py
import logging
from urllib.parse import urlparse, parse_qs

import streamlit as st
from src.storage.drive import get_auth_url, exchange_code_for_token, drive_service_from_token

st.set_page_config(page_title="Conexões", page_icon="🔌", layout="centered")

# Estado inicial
st.session_state.setdefault("OPENAI_API_KEY", "")
st.session_state.setdefault("google_token", None)
st.session_state.setdefault("google_connected", False)

# Também tratamos ?code aqui (se o usuário já estiver nesta página)
try:
    q = st.query_params
    code = q.get("code")
    error = q.get("error")
except Exception:
    q = st.experimental_get_query_params()
    code = (q.get("code") or [None])[0]
    error = (q.get("error") or [None])[0]

if error:
    st.error(f"Erro do Google OAuth: {error}")

if code and not st.session_state.get("google_connected"):
    try:
        token = exchange_code_for_token(code)
        st.session_state["google_token"] = token
        st.session_state["google_connected"] = True
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
        st.success("Google Drive conectado com sucesso.")
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao concluir o OAuth: {e}")

st.title("🔌 Conexões")

# --- OpenAI BYOK ---
st.subheader("OpenAI (sua chave)")
st.caption("A chave fica apenas nesta sessão do navegador.")
openai_key = st.text_input("OPENAI_API_KEY", type="password", value=st.session_state["OPENAI_API_KEY"])
st.session_state["OPENAI_API_KEY"] = (openai_key or "").strip()
if st.session_state["OPENAI_API_KEY"]:
    st.success("Chave da OpenAI inserida.")
else:
    st.info("Cole sua chave da OpenAI para habilitar o agente.")

st.divider()

# --- Google Drive OAuth ---
st.subheader("Google Drive")
st.caption("Conecte sua conta para criar/ler seus arquivos (escopo `drive.file`).")

if st.session_state.get("google_connected") and st.session_state.get("google_token"):
    st.success("Google Drive conectado.")
    # Teste opcional do service
    try:
        _ = drive_service_from_token(st.session_state["google_token"])
    except Exception as e:
        st.warning(f"Conectado, mas houve um alerta ao criar o service: {e}")
else:
    try:
        auth_url = get_auth_url()
    except Exception as e:
        st.error(f"Não foi possível gerar o link de conexão. Verifique os Secrets do app. Detalhe: {e}")
        auth_url = None

    c1, c2 = st.columns([1, 1])
    with c1:
        if auth_url:
            # Abre NA MESMA ABA
            st.markdown(f"[Conectar Google Drive]({auth_url})", unsafe_allow_html=True)
        else:
            st.button("Conectar Google Drive", disabled=True, use_container_width=True)

    with c2:
        dbg = st.toggle("🔧 Debug do OAuth", value=False, help="Mostra a URL de autorização e parâmetros (remova depois).")

    if dbg and auth_url:
        st.divider()
        st.caption("🔍 auth_url")
        st.code(auth_url, language="text")
        try:
            qs = parse_qs(urlparse(auth_url).query)
            show = {
                "client_id": qs.get("client_id", []),
                "redirect_uri": qs.get("redirect_uri", []),
                "scope": qs.get("scope", []),
                "response_type": qs.get("response_type", []),
                "access_type": qs.get("access_type", []),
                "prompt": qs.get("prompt", []),
                "state": "<presente>" if "state" in qs else "<ausente>",
            }
            st.json(show)
        except Exception as e:
            st.warning(f"Não foi possível parsear o auth_url: {e}")
        logging.info("AUTH_URL -> %s", auth_url)

# Verificação (sem expor valores)
with st.expander("Verificação rápida dos Secrets (nomes apenas)"):
    try:
        keys = list(st.secrets.keys())
        redir = st.secrets.get("GOOGLE_REDIRECT_URI", "")
        cid = st.secrets.get("GOOGLE_CLIENT_ID", "")
        st.write({"secrets_keys": keys})
        st.write({"client_id_suffix": cid[-20:] if isinstance(cid, str) else ""})
        st.write({"redirect_uri": redir})
    except Exception as e:
        st.warning(f"Secrets não disponíveis: {e}")



