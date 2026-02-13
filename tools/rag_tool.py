"""
=============================================================================
RAG TOOL - Ferramenta de Busca na Base de Conhecimento
=============================================================================

Esta tool permite que o agente busque informações na base de conhecimento
(RAG - Retrieval Augmented Generation).

Como funciona:
1. O agente decide usar esta tool quando precisa de informações específicas
2. A query é enviada para o vector store
3. Os documentos mais relevantes são retornados
4. O agente usa esses documentos para formular sua resposta

=============================================================================
"""

from langchain_core.tools import tool

# Variável global para armazenar a referência ao vector store
# Será configurada pelo agente quando o RAG estiver habilitado
_vector_store_manager = None


def set_vector_store(manager):
    """
    Configura o vector store manager para a tool de RAG.

    Args:
        manager: Instância de VectorStoreManager
    """
    global _vector_store_manager
    _vector_store_manager = manager


def get_vector_store():
    """Retorna o vector store manager atual."""
    return _vector_store_manager


@tool
def knowledge_base_search(query: str) -> str:
    """
    Busca informações na base de conhecimento.

    Use esta ferramenta quando precisar buscar informações específicas
    nos documentos carregados na base de conhecimento.

    Args:
        query: A pergunta ou termo de busca

    Returns:
        Os trechos mais relevantes encontrados nos documentos
    """
    global _vector_store_manager

    if _vector_store_manager is None:
        return "❌ Base de conhecimento não configurada. Nenhum documento foi carregado."

    try:
        # Busca os documentos mais relevantes
        results = _vector_store_manager.similarity_search(query, k=3)

        if not results:
            return "Nenhum documento relevante encontrado na base de conhecimento."

        # Formata os resultados
        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Desconhecido")
            filename = doc.metadata.get("filename", source)
            content = doc.page_content.strip()

            formatted_results.append(
                f"📄 **Documento {i}** ({filename}):\n{content}"
            )

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"❌ Erro ao buscar na base de conhecimento: {str(e)}"


# Para uso como tool no agente
rag_search_tool = knowledge_base_search

