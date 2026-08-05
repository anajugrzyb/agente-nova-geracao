"""
 Funções utilitárias e reutilizáveis para o projeto.
"""

from __future__ import annotations

from pathlib import Path


def list_pdf_files(directory: str | Path) -> list[Path]:
    """
    Retorna uma lista de arquivos PDF presentes no diretório especificado.
    Args:
        directory (str | Path): Caminho do diretório a ser verificado.
    Returns:
        list[Path]: Lista de objetos Path representando os arquivos PDF encontrados.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pdf"))


def vectorstore_exists(vectorstore_dir: str | Path) -> bool:
    """
    Verifica se o vetor de armazenamento existe no diretório especificado.
    Args:
        vectorstore_dir (str | Path): Caminho do diretório a ser verificado.
    Returns:
        bool: True se o vetor de armazenamento existir, False caso contrário.
    """
    vectorstore_dir = Path(vectorstore_dir)
    return (vectorstore_dir / "index.faiss").exists() and (
        vectorstore_dir / "index.pkl"
    ).exists()


def format_source_name(source_path: str) -> str:
    """
    Formata o nome da fonte com base no caminho do arquivo.
    Args:
        source_path (str): Caminho do arquivo de origem.
    Returns:
        str: Nome formatado da fonte.
    """
    mapping = {
        "calendario": "Calendário Acadêmico",
        "matricula": "Matrícula",
        "biblioteca": "Biblioteca",
        "estagio": "Estágio",
        "tcc": "TCC",
        "regulamento": "Regulamento Geral",
    }
    stem = Path(source_path).stem.lower()
    return mapping.get(stem, stem.capitalize())


def truncate_text(text: str, max_chars: int = 220) -> str:
    """
    Trunca um texto para exibição resumida, preservando palavras inteiras.
    Args:
        text (str): Texto a ser truncado.
        max_chars (int): Número máximo de caracteres.
    Returns:
        str: Texto truncado.    
    """
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
