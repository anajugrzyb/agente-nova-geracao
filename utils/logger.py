"""
Configuração e criação de um logger para o projeto, gerando logs tanto
no console quanto em um arquivo de log, facilitando o rastreamento de 
eventos e erros durante a execução do programa.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """
    Cria e retorna um logger configurado para o projeto.
    Args:
        name (str): Nome do logger.
        log_dir (str | Path): Diretório onde os logs serão armazenados.
    Returns:
        logging.Logger: Logger configurado.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Não foi possível criar o arquivo de log em disco.")

    logger.propagate = False
    return logger
