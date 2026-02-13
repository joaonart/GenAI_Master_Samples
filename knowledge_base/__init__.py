"""
=============================================================================
KNOWLEDGE BASE - Módulo de Base de Conhecimento (RAG)
=============================================================================

RAG = Retrieval Augmented Generation (Geração Aumentada por Recuperação)

O que é RAG?
- Uma técnica que combina busca de documentos com geração de texto
- Permite que o LLM responda perguntas sobre SEUS documentos
- Resolve o problema de "alucinação" do LLM

Como funciona:
1. INDEXAÇÃO: Documentos são divididos em "chunks" e convertidos em vetores
2. BUSCA: Quando o usuário pergunta, buscamos os chunks mais relevantes
3. GERAÇÃO: O LLM usa os chunks encontrados para gerar a resposta

Formatos de arquivo suportados:
- .txt, .md (texto)
- .pdf (requer pypdf)
- .csv (requer pandas)
- .docx (requer python-docx)
- .json

=============================================================================
"""

from .vector_store import VectorStoreManager, create_simple_knowledge_base
from .document_loader import (
    load_document,
    load_text_file,
    load_pdf_file,
    load_csv_file,
    load_docx_file,
    load_json_file,
    load_documents_from_directory,
    split_documents,
    get_supported_formats,
    SUPPORTED_FORMATS
)

__all__ = [
    # Vector Store
    "VectorStoreManager",
    "create_simple_knowledge_base",
    # Document Loaders
    "load_document",
    "load_text_file",
    "load_pdf_file",
    "load_csv_file",
    "load_docx_file",
    "load_json_file",
    "load_documents_from_directory",
    "split_documents",
    "get_supported_formats",
    "SUPPORTED_FORMATS",
]

