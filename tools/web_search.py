"""
=============================================================================
WEB SEARCH TOOL - Ferramenta de Busca na Web (Real)
=============================================================================

Esta ferramenta permite buscar informações reais na web usando APIs públicas.

APIs Implementadas:
1. DuckDuckGo (gratuito, sem API key)
2. Tavily (gratuito com limite, precisa API key - excelente para LLMs)

Como obter API Keys:
- Tavily: https://tavily.com (gratuito, 1000 buscas/mês)

=============================================================================
"""

import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict


class WebSearchInput(BaseModel):
    """Schema de entrada para busca na web."""
    query: str = Field(
        description="O termo de busca. Exemplo: 'clima em São Paulo', 'últimas notícias sobre IA'"
    )
    num_results: int = Field(
        default=5,
        description="Número de resultados a retornar (1-10)"
    )


def search_with_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Busca usando DuckDuckGo (gratuito, sem API key).

    Usa a biblioteca ddgs (antigo duckduckgo-search).
    Instalar: pip install ddgs
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            # Busca textual
            search_results = list(ddgs.text(query, max_results=num_results))

            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })

        return results

    except ImportError:
        raise ImportError(
            "ddgs não está instalado.\n"
            "Execute: pip install ddgs"
        )
    except Exception as e:
        raise Exception(f"Erro na busca DuckDuckGo: {str(e)}")


def search_with_tavily(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Busca usando Tavily API (excelente para LLMs).

    Requer TAVILY_API_KEY configurada.
    Obtenha grátis em: https://tavily.com
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY não configurada.\n"
            "Obtenha grátis em: https://tavily.com"
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=num_results)

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "url": r.get("url", "")
            })

        return results

    except ImportError:
        raise ImportError(
            "tavily-python não está instalado.\n"
            "Execute: pip install tavily-python"
        )
    except Exception as e:
        raise Exception(f"Erro na busca Tavily: {str(e)}")


def search_with_wikipedia(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Busca na Wikipedia (gratuito, sem API key).

    Usa a biblioteca wikipedia.
    Instalar: pip install wikipedia
    """
    try:
        import wikipedia
        wikipedia.set_lang("pt")  # Português

        results = []

        # Busca páginas relacionadas
        search_results = wikipedia.search(query, results=num_results)

        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "title": page.title,
                    "snippet": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                    "url": page.url
                })
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue

        return results

    except ImportError:
        raise ImportError(
            "wikipedia não está instalado.\n"
            "Execute: pip install wikipedia"
        )
    except Exception as e:
        raise Exception(f"Erro na busca Wikipedia: {str(e)}")


# Configuração do provedor de busca padrão
# Opções: "duckduckgo", "tavily", "wikipedia"
DEFAULT_SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    Busca informações atualizadas na web.

    Use esta ferramenta quando precisar:
    - Buscar informações atuais ou recentes
    - Verificar fatos ou dados
    - Encontrar notícias ou eventos
    - Pesquisar sobre qualquer assunto

    Args:
        query: O termo de busca
        num_results: Quantidade de resultados (padrão: 5)

    Returns:
        Resultados da busca formatados com título, resumo e URL
    """
    provider = DEFAULT_SEARCH_PROVIDER.lower()

    try:
        # Tenta usar o provedor configurado
        if provider == "tavily" and os.getenv("TAVILY_API_KEY"):
            results = search_with_tavily(query, num_results)
            source = "Tavily"
        elif provider == "wikipedia":
            results = search_with_wikipedia(query, num_results)
            source = "Wikipedia"
        else:
            # DuckDuckGo como padrão (gratuito, sem API key)
            results = search_with_duckduckgo(query, num_results)
            source = "DuckDuckGo"

        if not results:
            return f'Nenhum resultado encontrado para: "{query}"'

        # Formata os resultados
        output = f'🔍 Resultados da busca ({source}) para: "{query}"\n\n'

        for i, result in enumerate(results, 1):
            output += f"""📄 **Resultado {i}:**
   **Título:** {result['title']}
   **Resumo:** {result['snippet'][:300]}{'...' if len(result['snippet']) > 300 else ''}
   **URL:** {result['url']}

"""

        return output

    except ImportError as e:
        return f"""❌ Biblioteca não instalada: {str(e)}

Para habilitar busca na web, instale uma das opções:
• pip install ddgs              (recomendado, gratuito)
• pip install tavily-python     (precisa API key)
• pip install wikipedia         (apenas Wikipedia)
"""
    except Exception as e:
        return f'❌ Erro ao buscar: {str(e)}'


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    print("🔍 Testando Web Search Tool")
    print("=" * 50)

    # Teste de busca
    result = web_search_tool.invoke({"query": "Python programming language"})
    print(result)


