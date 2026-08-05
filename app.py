
"""
Interface Streamlit do Assistente Virtual da Universidade Nova Geração.
Renderiza a interface da aplicação, gerencia o estado da sessão e integra o RAG Engine 
para responder perguntas sobre documentos institucionais. Também oferece ingestão de documentos, 
histórico de conversas, configuração do modelo LLM e personalização da interface com CSS.
"""

from __future__ import annotations

import time

import streamlit as st
from langchain_community.vectorstores import FAISS

from config import settings
from ingest import get_embeddings_model, load_vectorstore, rebuild_index
from utils.helpers import format_source_name, truncate_text, vectorstore_exists
from utils.logger import get_logger
from rag import RagAnswer, build_rag_engine

logger = get_logger(__name__, settings.logs_dir)

st.set_page_config(
    page_title="Universidade Nova Geração | Assistente Virtual",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>

/* ==============================
   BASE DA APLICAÇÃO
================================ */

.main {
    background-color: #F8FAFC;
}

/* Remove espaço excessivo */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* ==============================
   HEADER PRINCIPAL
================================ */

.app-header {

    padding: 2rem 2.5rem;

    border-radius: 20px;

    background:
    linear-gradient(
        135deg,
        #0F172A 0%,
        #1E3A8A 45%,
        #2563EB 100%
    );

    color:white;

    margin-bottom:2rem;

    box-shadow:
    0 15px 35px rgba(15,23,42,0.18);

}


.app-header h1 {

    margin:0;

    font-size:2rem;

    font-weight:800;

    letter-spacing:-0.5px;

}


.app-header p {

    margin-top:0.6rem;

    font-size:1rem;

    opacity:0.85;

}



/* ==============================
   CHAT
================================ */


/* usuário */

.chat-bubble-user {

    background:
    linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );


    color:white;

    padding:
    1rem 1.2rem;

    border-radius:
    20px 20px 6px 20px;


    margin:

    0.8rem 0 0.8rem auto;


    max-width:75%;


    box-shadow:
    0 8px 18px
    rgba(37,99,235,0.25);

    font-size:0.95rem;

}


/* IA */

.chat-bubble-bot {


    background:white;

    color:#111827;


    padding:
    1rem 1.2rem;


    border-radius:
    20px 20px 20px 6px;


    margin:
    0.8rem auto 0.8rem 0;


    max-width:80%;


    border:
    1px solid #E2E8F0;


    box-shadow:
    0 8px 20px
    rgba(15,23,42,0.08);


    line-height:1.6;

}



/* ==============================
   FONTES / DOCUMENTOS
================================ */


.source-tag {


    display:inline-flex;


    align-items:center;


    background:#DBEAFE;


    color:#1E40AF;


    font-size:0.75rem;


    font-weight:600;


    padding:
    0.35rem 0.75rem;


    border-radius:999px;


    margin:
    0.25rem;


}



/* ==============================
   SIDEBAR
================================ */


section[data-testid="stSidebar"] {


    background:
    linear-gradient(
        180deg,
        #0F172A,
        #1E293B
    );


}


section[data-testid="stSidebar"] * {

    color:#F8FAFC !important;

}


/* título sidebar */

section[data-testid="stSidebar"] h1 {

    font-weight:800;

}



/* ==============================
   BOTÕES
================================ */


.stButton > button {


    width:100%;


    background:
    linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );


    color:white;


    border:none;


    border-radius:12px;


    padding:
    0.65rem 1rem;


    font-weight:600;


    transition:all .25s ease;


}



.stButton > button:hover {


    transform:
    translateY(-2px);


    box-shadow:
    0 8px 20px
    rgba(37,99,235,.3);


}



/* ==============================
   INPUTS
================================ */


.stTextInput input,
.stTextArea textarea {


    background:white;


    border-radius:14px;


    border:
    1px solid #CBD5E1;


    padding:0.7rem;


}



.stTextInput input:focus,
.stTextArea textarea:focus {


    border-color:#2563EB;


    box-shadow:
    0 0 0 3px
    rgba(37,99,235,.15);


}



/* ==============================
   CARDS
================================ */


div[data-testid="stVerticalBlock"] {


    border-radius:16px;


}



/* caixas de informação */

div[data-testid="stAlert"] {


    border-radius:14px;

}



/* ==============================
   SCROLLBAR
================================ */


::-webkit-scrollbar {

width:8px;

}


::-webkit-scrollbar-thumb {


background:#CBD5E1;

border-radius:10px;

}


</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def init_session_state() -> None:
    """Inicializa as variáveis de estado da sessão Streamlit, se ainda não existirem."""
    defaults = {
        "chat_history": [],  # list[tuple[str, RagAnswer]]
        "vectorstore": None,
        "rag_engine": None,
        "index_ready": vectorstore_exists(settings.vectorstore_dir),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

@st.cache_resource(show_spinner=False)
def _cached_embeddings():
    """Cacheia o modelo de embeddings entre execuções (evita recarregar a cada rerun)."""
    return get_embeddings_model()


def load_existing_index() -> FAISS | None:
    """Tenta carregar um índice vetorial já persistido em disco."""
    if not vectorstore_exists(settings.vectorstore_dir):
        return None
    try:
        embeddings = _cached_embeddings()
        return load_vectorstore(embeddings, settings.vectorstore_dir)
    except Exception as exc: 
        logger.error("Falha ao carregar índice existente: %s", exc)
        st.error("Não foi possível carregar o índice salvo. Tente reprocessar os documentos.")
        return None


def run_ingestion() -> None:
    """Executa (ou reexecuta) a ingestão dos documentos, exibindo progresso na UI."""
    progress_bar = st.sidebar.progress(0, text="Iniciando ingestão...")

    def _progress(value: float, message: str) -> None:
        progress_bar.progress(min(max(value, 0.0), 1.0), text=message)

    try:
        with st.spinner("Processando documentos institucionais..."):
            vectorstore = rebuild_index(progress_callback=_progress)
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_engine = build_rag_engine(vectorstore)
        st.session_state.index_ready = True
        st.sidebar.success("✅ Documentos indexados com sucesso!")
        logger.info("Ingestão executada via interface Streamlit com sucesso.")
    except FileNotFoundError as exc:
        st.sidebar.error(f"⚠️ {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro inesperado durante a ingestão via UI.")
        st.sidebar.error(f"❌ Erro ao processar documentos: {exc}")
    finally:
        time.sleep(0.5)
        progress_bar.empty()


def ensure_rag_engine_loaded() -> None:
    """Garante que exista um `RagEngine` pronto para uso, carregando o índice se necessário."""
    if st.session_state.rag_engine is not None:
        return

    vectorstore = load_existing_index()
    if vectorstore is not None:
        st.session_state.vectorstore = vectorstore
        try:
            st.session_state.rag_engine = build_rag_engine(vectorstore)
        except ValueError as exc:
            st.sidebar.error(f"⚠️ {exc}")


ensure_rag_engine_loaded()

with st.sidebar:
    st.markdown("## 🎓 Universidade Nova Geração")
    st.markdown("### Painel de Controle")
    st.markdown("---")

    st.markdown("#### 📚 Base de Conhecimento")
    STATUS_LABEL = (
        "🟢 Índice carregado" 
        if st.session_state.index_ready
        else "🔴 Índice não encontrado"
    )
    st.markdown(STATUS_LABEL)

    if st.button("🔄 Carregar / Reprocessar Documentos", use_container_width=True):
        run_ingestion()

    st.caption(
        "Este botão lê todos os PDFs em `data/`, gera os embeddings e "
        "reconstrói o índice vetorial (FAISS)."
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Configurações Ativas")
    st.caption(f"**Modelo LLM (GROQ):** `{settings.groq_model}`")
    st.caption(f"**Embeddings:** `{settings.embedding_model_name}`")
    st.caption(f"**Chunk size / overlap:** `{settings.chunk_size}` / `{settings.chunk_overlap}`")
    st.caption(f"**Top-K recuperados:** `{settings.retrieval_k}`")

    st.markdown("---")
    st.markdown("#### 📄 Documentos Disponíveis")
    for doc_name in ["Calendário Acadêmico", "Matrícula", "TCC", "Biblioteca", "Estágio", "Regulamento Geral"]:
        st.caption(f"• {doc_name}")

    st.markdown("---")
    if st.button("🗑️ Limpar Histórico da Conversa", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Cabeçalho principal
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>{settings.app_title}</h1>
        <p>{settings.app_description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.index_ready:
    st.info(
        "👋 Bem-vindo! Para começar, clique em **'Carregar / Reprocessar Documentos'** "
        "na barra lateral para indexar os documentos da universidade."
    )

chat_container = st.container()

with chat_container:
    for question, rag_answer in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-bubble-user">🧑‍🎓 {question}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-bubble-bot">🤖 {rag_answer.answer}</div>',
            unsafe_allow_html=True,
        )
        if rag_answer.sources:
            unique_sources = sorted({format_source_name(d.metadata.get("source", "")) for d in rag_answer.sources})
            tags_html = "".join(f'<span class="source-tag">📄 {s}</span>' for s in unique_sources)
            st.markdown(tags_html, unsafe_allow_html=True)
            with st.expander("🔍 Ver trechos utilizados como fonte"):
                for doc in rag_answer.sources:
                    src = format_source_name(doc.metadata.get("source", ""))
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**{src}** (página {page})")
                    st.caption(truncate_text(doc.page_content))
        st.markdown("&nbsp;", unsafe_allow_html=True)

st.markdown("### 💬 Faça sua pergunta")

with st.form(key="question_form", clear_on_submit=True):
    col_input, col_button = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Pergunta",
            placeholder="Ex.: Como faço minha matrícula?",
            label_visibility="collapsed",
        )
    with col_button:
        submitted = st.form_submit_button("Perguntar 🚀", use_container_width=True)

st.caption(
    "Exemplos: *Como renovar livros?* · *Quantas horas de estágio preciso cumprir?* · "
    "*Qual a nota mínima para aprovação?* · *Quando começam as férias?*"
)

if submitted and user_question.strip():
    if not st.session_state.index_ready or st.session_state.rag_engine is None:
        st.warning(
            "⚠️ A base de conhecimento ainda não foi carregada. "
            "Clique em 'Carregar / Reprocessar Documentos' na barra lateral."
        )
    else:
        with st.spinner("🤔 Consultando os documentos da universidade..."):
            try:
                answer: RagAnswer = st.session_state.rag_engine.ask(user_question)
            except Exception as exc: 
                logger.exception("Erro ao gerar resposta.")
                answer = RagAnswer(
                    question=user_question,
                    answer=f"❌ Ocorreu um erro ao gerar a resposta: {exc}",
                )
        st.session_state.chat_history.append((user_question, answer))
        st.rerun()

elif submitted:
    st.warning("Digite uma pergunta antes de enviar.")

st.markdown("---")
st.caption(
    "Assistente Virtual desenvolvido com LangChain, LangGraph, FAISS, "
    "HuggingFace Embeddings, GROQ e Streamlit — Projeto acadêmico de portfólio."
)
