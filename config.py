"""
Configurações centrais do aplicativo, incluindo diretórios de dados, 
chaves de API e parâmetros de LLM.

As chaves de API e outros parâmetros sensíveis são carregados do arquivo .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """
    Agrupa as configurações centrais do aplicativo, incluindo diretórios de dados,
    chaves de API e parâmetros de LLM.
    
    """

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    logs_dir: Path = BASE_DIR / "logs"

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )

    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))

    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4")))

    vectorstore_collection_name: str = "universidade_nova_geracao"

    app_title: str = "🎓 Assistente Virtual — Universidade Nova Geração"
    app_description: str = (
        "Tire suas dúvidas sobre matrícula, TCC, estágio, biblioteca, "
        "calendário acadêmico e regulamento geral, com respostas baseadas "
        "exclusivamente nos documentos oficiais da universidade."
    )

    no_answer_message: str = (
        "Essa informação não está presente na documentação da universidade."
    )


settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
