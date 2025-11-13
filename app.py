# app.py
import streamlit as st

# Precisamos só desta função aqui; o resto do fluxo fica na página Conexões
from src.storage.drive import exchange_code_for_token

st.set_page_config(page_title="Agente do Livro", page_icon="📚", layout="wide")

# Estado base da sessão
st.session_state.setdefault("google_connected", False)
st.session_state.setdefault("google_token", None)
st.session_state.setdefault("OPENAI_API_KEY", "")

# ---- Captura retorno do Google quando ele volta para a RAIZ do app ----
try:
    q = st.query_params  # Streamlit recente
    code = q.get("code")
    error = q.get("error")
except Exception:
    q = st.experimental_get_query_params()  # fallback versões antigas
    code = (q.get("code") or [None])[0]
    error = (q.get("error") or [None])[0]

if error:
    st.warning(f"Google OAuth erro: {error}")

if code and not st.session_state.get("google_connected"):
    try:
        token = exchange_code_for_token(code)
        st.session_state["google_token"] = token
        st.session_state["google_connected"] = True
        # Limpa parâmetros para não repetir
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
        st.toast("Google Drive conectado com sucesso.", icon="✅")
        # Volta para a página de Conexões (se suportado)
        try:
            st.switch_page("pages/0_Conexoes.py")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Falha ao concluir o OAuth: {e}")

# ---- HOME (conteúdo simples) ----
st.title("📚 Agente do Livro")
st.write(
    "Use o menu lateral para acessar **Conexões**, **Editor**, **Transcritor** e **Assistente**."
)

# Status rápido
col1, col2 = st.columns(2)
with col1:
    st.metric("Google Drive", "Conectado" if st.session_state.get("google_connected") else "Desconectado")
with col2:
    st.metric("OpenAI Key", "Definida" if bool(st.session_state.get("OPENAI_API_KEY")) else "Vazia")


