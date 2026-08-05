"""
Pipeline de ingestão de documentos: leitura dos PDFs -> chunking ->
geração de embeddings -> persistência do índice vetorial (FAISS).

Pode ser executado de forma independente:
    $ python ingest.py

Ou importado pela aplicação Streamlit / RAG engine para (re)construir
o índice sob demanda.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils.helpers import list_pdf_files
from utils.logger import get_logger

logger = get_logger(__name__, settings.logs_dir)

ProgressCallback = Optional[Callable[[float, str], None]]


def load_documents(data_dir: str | Path = settings.data_dir) -> list[Document]:
    """Carrega todos os arquivos PDF de um diretório em objetos `Document`.

    Args:
        data_dir: Diretório contendo os arquivos PDF da universidade.

    Returns:
        Lista de documentos (uma entrada por página de PDF), com metadados
        de origem preservados.

    Raises:
        FileNotFoundError: Se nenhum arquivo PDF for encontrado no diretório.
    """
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum arquivo PDF encontrado em '{data_dir}'. "
            "Adicione os documentos institucionais antes de rodar a ingestão."
        )

    documents: list[Document] = []
    for pdf_path in pdf_files:
        logger.info("Carregando documento: %s", pdf_path.name)
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            documents.extend(pages)
        except Exception as exc:  
            logger.error("Falha ao carregar '%s': %s", pdf_path.name, exc)

    logger.info("Total de páginas carregadas: %d", len(documents))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    """Divide os documentos em chunks menores, mais adequados para embeddings.

    Args:
        documents: Lista de documentos carregados (por página).
        chunk_size: Tamanho máximo (em caracteres) de cada chunk.
        chunk_overlap: Sobreposição entre chunks consecutivos, para preservar contexto.

    Returns:
        Lista de chunks (`Document`) prontos para vetorização.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Documentos divididos em %d chunks.", len(chunks))
    return chunks


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Instancia o modelo de embeddings gratuito da HuggingFace (executado localmente).

    Returns:
        Instância configurada de `HuggingFaceEmbeddings`.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    progress_callback: ProgressCallback = None,
) -> FAISS:
    """Gera os embeddings dos chunks e constrói o índice vetorial FAISS em memória.

    Args:
        chunks: Lista de chunks de texto já divididos.
        embeddings: Modelo de embeddings a ser utilizado.
        progress_callback: Função opcional `(progresso: float, mensagem: str)`
            chamada periodicamente para reportar o andamento (útil para barras
            de progresso na interface Streamlit).

    Returns:
        Índice vetorial FAISS pronto para consultas de similaridade.
    """
    total = len(chunks)
    batch_size = max(1, total // 10)
    vectorstore: Optional[FAISS] = None

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        processed = min(start + batch_size, total)
        if progress_callback:
            progress_callback(
                processed / total, f"Vetorizando chunks... ({processed}/{total})"
            )
        logger.info("Progresso da vetorização: %d/%d", processed, total)

    assert vectorstore is not None, "Nenhum chunk foi processado."
    return vectorstore


def persist_vectorstore(
    vectorstore: FAISS, vectorstore_dir: str | Path = settings.vectorstore_dir
) -> None:
    """Salva o índice FAISS em disco para reutilização futura sem reprocessamento.

    Args:
        vectorstore: Índice vetorial já construído.
        vectorstore_dir: Diretório de destino para persistência.
    """
    vectorstore_dir = Path(vectorstore_dir)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    logger.info("Índice vetorial salvo em: %s", vectorstore_dir)


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    vectorstore_dir: str | Path = settings.vectorstore_dir,
) -> FAISS:
    """Carrega um índice FAISS previamente persistido em disco.

    Args:
        embeddings: Modelo de embeddings (deve ser o mesmo usado na criação do índice).
        vectorstore_dir: Diretório onde o índice está salvo.

    Returns:
        Índice vetorial FAISS carregado e pronto para uso.
    """
    logger.info("Carregando índice vetorial existente de: %s", vectorstore_dir)
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def rebuild_index(
    data_dir: str | Path = settings.data_dir,
    vectorstore_dir: str | Path = settings.vectorstore_dir,
    progress_callback: ProgressCallback = None,
) -> FAISS:
    """Executa o pipeline completo de ingestão: carregar -> dividir -> vetorizar -> salvar.

    Args:
        data_dir: Diretório com os PDFs de origem.
        vectorstore_dir: Diretório de destino do índice vetorial.
        progress_callback: Callback opcional de progresso para a UI.

    Returns:
        Índice vetorial FAISS recém-construído.
    """
    start_time = time.time()

    if progress_callback:
        progress_callback(0.05, "Lendo documentos PDF...")
    documents = load_documents(data_dir)

    if progress_callback:
        progress_callback(0.25, "Dividindo documentos em chunks...")
    chunks = split_documents(documents)

    if progress_callback:
        progress_callback(0.35, "Carregando modelo de embeddings...")
    embeddings = get_embeddings_model()

    vectorstore = build_vectorstore(chunks, embeddings, progress_callback)

    if progress_callback:
        progress_callback(0.95, "Salvando índice vetorial em disco...")
    persist_vectorstore(vectorstore, vectorstore_dir)

    elapsed = time.time() - start_time
    logger.info("Ingestão concluída em %.2f segundos.", elapsed)
    if progress_callback:
        progress_callback(1.0, f"Ingestão concluída em {elapsed:.1f}s.")

    return vectorstore


def clear_vectorstore(vectorstore_dir: str | Path = settings.vectorstore_dir) -> None:
    """Remove o índice vetorial persistido, forçando uma nova ingestão.

    Args:
        vectorstore_dir: Diretório do índice a ser removido.
    """
    vectorstore_dir = Path(vectorstore_dir)
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Índice vetorial removido: %s", vectorstore_dir)


if __name__ == "__main__":
    def _cli_progress(progress: float, message: str) -> None:
        print(f"[{progress * 100:5.1f}%] {message}")

    rebuild_index(progress_callback=_cli_progress)
    print("\n✅ Ingestão finalizada com sucesso! Índice salvo em 'vectorstore/'.")
