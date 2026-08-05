"""
Motor de RAG (Retrieval-Augmented Generation) do Assistente Virtual da
Universidade Nova Geração.

Este módulo é responsável por:
    - Instanciar o LLM (via GROQ, gratuito e de baixa latência).
    - Montar a cadeia de recuperação + geração (equivalente ao RetrievalQA).
    - Orquestrar o fluxo de resposta com LangGraph (retrieve -> generate -> validate).
    - Garantir que o agente responda apenas com base no contexto recuperado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from config import settings
from prompts import build_qa_prompt
from utils.logger import get_logger

logger = get_logger(__name__, settings.logs_dir)


@dataclass
class RagAnswer:
    """Representa o resultado de uma consulta ao agente RAG."""

    question: str
    answer: str
    sources: list[Document] = field(default_factory=list)


def get_llm() -> ChatGroq:
    """Instancia o modelo de linguagem gratuito hospedado na GROQ.

    Returns:
        Instância configurada de `ChatGroq`.

    Raises:
        ValueError: Se a variável de ambiente GROQ_API_KEY não estiver definida.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY não encontrada. Defina essa variável no arquivo .env "
            "(veja .env.example) para utilizar o modelo de linguagem."
        )
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def format_docs(docs: list[Document]) -> str:
    """Concatena os documentos recuperados em uma única string de contexto.

    Args:
        docs: Lista de documentos (chunks) retornados pelo retriever.

    Returns:
        Texto único, com cada trecho identificado por sua fonte.
    """
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        formatted_chunks.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)


class GraphState(TypedDict):
    """Estado compartilhado entre os nós do grafo LangGraph."""

    question: str
    documents: list[Document]
    answer: str


class RagEngine:
    """Encapsula o pipeline de RAG completo, orquestrado com LangGraph.

    O fluxo segue três etapas (nós do grafo):
        1. retrieve  -> busca os chunks mais relevantes no índice vetorial.
        2. generate  -> gera a resposta com o LLM, restrita ao contexto.
        3. validate  -> garante que respostas vazias/genéricas virem a
                        mensagem padrão de "informação não encontrada".
    """

    def __init__(self, vectorstore: FAISS) -> None:
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": settings.retrieval_k}
        )
        self.llm = get_llm()
        self.prompt = build_qa_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.graph = self._build_graph()

    # --- Nós do grafo -------------------------------------------------

    def _retrieve_node(self, state: GraphState) -> GraphState:
        docs = self.retriever.invoke(state["question"])
        logger.info("Recuperados %d chunks para a pergunta.", len(docs))
        return {**state, "documents": docs}

    def _generate_node(self, state: GraphState) -> GraphState:
        context = format_docs(state["documents"])
        if not context.strip():
            return {**state, "answer": settings.no_answer_message}

        answer = self.chain.invoke({"context": context, "question": state["question"]})
        return {**state, "answer": answer.strip()}

    def _validate_node(self, state: GraphState) -> GraphState:
        answer = state["answer"]
        # Salvaguarda extra: se o modelo "fugir" do contexto e não tiver
        # documentos de apoio, força a mensagem padrão.
        if not state["documents"] or not answer:
            answer = settings.no_answer_message
        return {**state, "answer": answer}

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate", self._validate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)

        return graph.compile()

    def ask(self, question: str) -> RagAnswer:
        """Responde a uma pergunta do aluno utilizando o pipeline de RAG.

        Args:
            question: Pergunta em linguagem natural feita pelo aluno.

        Returns:
            Objeto `RagAnswer` contendo a pergunta, a resposta gerada e os
            documentos-fonte utilizados como contexto.
        """
        question = question.strip()
        if not question:
            return RagAnswer(question=question, answer="Por favor, digite uma pergunta.")

        try:
            result: GraphState = self.graph.invoke({
                "question": question,
                "documents": [],
                "answer": "",
            })
        except Exception as exc:
            logger.error("Erro ao processar a pergunta '%s': %s", question, exc)
            return RagAnswer(
                question=question,
                answer=(
                    "Ocorreu um erro ao consultar o assistente. "
                    "Tente novamente em instantes."
                ),
            )

        return RagAnswer(
            question=question,
            answer=result["answer"],
            sources=result["documents"],
        )


def build_rag_engine(vectorstore: FAISS) -> RagEngine:
    """Função de fábrica para criação do `RagEngine`.

    Args:
        vectorstore: Índice vetorial FAISS já carregado ou construído.

    Returns:
        Instância pronta de `RagEngine`.
    """
    return RagEngine(vectorstore)
