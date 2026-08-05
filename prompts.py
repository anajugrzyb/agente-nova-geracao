"""
Templates de prompt utilizados pelo agente RAG. Manter os prompts isolados
em um módulo próprio facilita ajustes de comportamento do modelo sem
tocar na lógica de recuperação e orquestração (rag.py).
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from config import settings

SYSTEM_INSTRUCTIONS = f"""\
Você é o Assistente Virtual da Universidade Nova Geração, um agente
especializado em responder dúvidas de alunos com base exclusivamente
nos documentos institucionais fornecidos como contexto.

Regras que você deve seguir rigorosamente:
1. Responda SOMENTE com base nas informações contidas no CONTEXTO abaixo.
2. Se a resposta não estiver clara ou não existir no CONTEXTO, responda
   exatamente: "{settings.no_answer_message}".
3. Nunca invente, deduza além do texto ou complete informações que não
   estejam explicitamente no CONTEXTO.
4. Seja claro, objetivo e cordial, como um atendente experiente da
   secretaria acadêmica.
5. Quando fizer sentido, organize a resposta em passos ou tópicos.
6. Responda sempre em português do Brasil.
"""

QA_PROMPT_TEMPLATE = """\
{system_instructions}

CONTEXTO (trechos extraídos dos documentos oficiais da universidade):
---------------------
{context}
---------------------

PERGUNTA DO ALUNO:
{question}

RESPOSTA:"""


def build_qa_prompt() -> PromptTemplate:
    """Constrói o PromptTemplate utilizado pela cadeia de Perguntas e Respostas.
    """
    return PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"system_instructions": SYSTEM_INSTRUCTIONS},
    )
