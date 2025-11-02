# src/llm/editor.py
from __future__ import annotations
from langchain_openai import ChatOpenAI
from typing import Optional

SYSTEM_EDITOR = """Você é um editor de livros tradicional.
TAREFA: revisar o texto fornecido, mantendo o conteúdo factual e a organização original na medida do possível.
- NÃO invente fatos, personagens, eventos ou conteúdos novos.
- Corrija gramática, concordância, ortografia, pontuação e fluidez.
- Padronize aspas, travessões e formatação; quebre parágrafos muito longos quando fizer sentido.
- Preserve citações e nomes próprios. Mantenha o sentido.
- Se as instruções do usuário pedirem estilo/voz/audiência, aplique sem acrescentar novas ideias.
- Saída: somente o texto final revisado, sem explicações, sem markdown extra.
"""

def build_user_prompt(
    original_text: str,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
    language: Optional[str] = "PT-BR",
    notes: Optional[str] = None,
    instructions: Optional[str] = None,
) -> str:
    parts = []
    parts.append("Configurações desejadas para a edição:")
    parts.append(f"- Idioma/norma: {language or 'PT-BR'}")
    if audience: parts.append(f"- Público-alvo: {audience}")
    if tone: parts.append(f"- Tom/voz: {tone}")
    if notes: parts.append(f"- Observações: {notes}")
    if instructions: parts.append(f"- Instruções específicas: {instructions}")
    parts.append("\n--- TEXTO ORIGINAL ---\n")
    parts.append(original_text)
    return "\n".join(parts)

def edit_as_book_editor(
    api_key: str,
    original_text: str,
    audience: Optional[str],
    tone: Optional[str],
    language: Optional[str],
    notes: Optional[str],
    instructions: Optional[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    user_prompt = build_user_prompt(
        original_text=original_text,
        audience=audience,
        tone=tone,
        language=language,
        notes=notes,
        instructions=instructions,
    )
    msg = llm.invoke([("system", SYSTEM_EDITOR), ("user", user_prompt)])
    return msg.content.strip()
