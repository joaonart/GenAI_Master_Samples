"""
=============================================================================
WIKIPEDIA TOOL - Ferramenta de Consulta à Wikipedia
=============================================================================

Esta ferramenta permite consultar a Wikipedia para obter resumos de artigos,
buscar artigos e extrair informações enciclopédicas.

API: https://en.wikipedia.org/api/rest_v1/
Documentação: https://www.mediawiki.org/wiki/REST_API

IMPORTANTE:
- API gratuita, não requer autenticação
- Suporta múltiplos idiomas (pt, en, es, etc.)
- Rate limit: Seja responsável, ~200 req/s máximo
- Ideal para: FAQ bots, assistentes de conhecimento, pesquisa

Funcionalidades:
1. Buscar resumo de um artigo
2. Pesquisar artigos por termo
3. Obter artigo completo
4. Obter artigo do dia (featured)

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import requests
import time
from typing import Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# URLs base para diferentes idiomas
WIKIPEDIA_API_URLS = {
    "pt": "https://pt.wikipedia.org/api/rest_v1",
    "en": "https://en.wikipedia.org/api/rest_v1",
    "es": "https://es.wikipedia.org/api/rest_v1",
    "fr": "https://fr.wikipedia.org/api/rest_v1",
    "de": "https://de.wikipedia.org/api/rest_v1",
    "it": "https://it.wikipedia.org/api/rest_v1",
    "ja": "https://ja.wikipedia.org/api/rest_v1",
}

# Headers padrão
DEFAULT_HEADERS = {
    "User-Agent": "GenAI_Master_Samples/1.0 (Educational Project; Python)",
    "Accept": "application/json"
}

# Controle de rate limiting
_last_request_time = 0


def _rate_limit():
    """Implementa rate limiting básico."""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < 0.1:  # Máximo 10 req/s para ser conservador
        time.sleep(0.1 - time_since_last)

    _last_request_time = time.time()


def _get_api_url(language: str = "pt") -> str:
    """Retorna a URL da API para o idioma especificado."""
    return WIKIPEDIA_API_URLS.get(language.lower(), WIKIPEDIA_API_URLS["pt"])


# =============================================================================
# FUNÇÕES DE CONSULTA À API
# =============================================================================

def get_article_summary(
    title: str,
    language: str = "pt"
) -> Dict[str, Any]:
    """
    Obtém o resumo de um artigo da Wikipedia.

    Args:
        title: Título do artigo (ex: "Albert Einstein", "Python")
        language: Código do idioma (pt, en, es, etc.)

    Returns:
        Dicionário com dados do resumo
    """
    _rate_limit()

    base_url = _get_api_url(language)
    # Substitui espaços por underscore para a URL
    encoded_title = title.replace(" ", "_")

    try:
        response = requests.get(
            f"{base_url}/page/summary/{encoded_title}",
            headers=DEFAULT_HEADERS,
            timeout=10
        )

        if response.status_code == 404:
            return {"error": f"Artigo '{title}' não encontrado na Wikipedia ({language})."}

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def search_articles(
    query: str,
    language: str = "pt",
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Pesquisa artigos na Wikipedia.

    Args:
        query: Termo de busca
        language: Código do idioma
        limit: Número máximo de resultados

    Returns:
        Lista de artigos encontrados
    """
    _rate_limit()

    base_url = _get_api_url(language)

    try:
        response = requests.get(
            f"{base_url}/page/related/{query.replace(' ', '_')}",
            headers=DEFAULT_HEADERS,
            timeout=10
        )

        if response.status_code == 404:
            # Tenta busca alternativa usando a API de pesquisa
            return _search_fallback(query, language, limit)

        response.raise_for_status()
        data = response.json()

        pages = data.get("pages", [])[:limit]
        return pages

    except requests.exceptions.RequestException:
        return _search_fallback(query, language, limit)


def _search_fallback(
    query: str,
    language: str,
    limit: int
) -> List[Dict[str, Any]]:
    """Busca alternativa usando a API de pesquisa da MediaWiki."""
    _rate_limit()

    # Usa a API action da MediaWiki
    wiki_url = f"https://{language}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1
    }

    try:
        response = requests.get(
            wiki_url,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title"),
                "description": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                "pageid": item.get("pageid")
            })

        return results

    except requests.exceptions.RequestException as e:
        return [{"error": f"Erro na busca: {str(e)}"}]


def get_article_sections(
    title: str,
    language: str = "pt"
) -> Dict[str, Any]:
    """
    Obtém as seções de um artigo.

    Args:
        title: Título do artigo
        language: Código do idioma

    Returns:
        Dicionário com seções do artigo
    """
    _rate_limit()

    base_url = _get_api_url(language)
    encoded_title = title.replace(" ", "_")

    try:
        response = requests.get(
            f"{base_url}/page/mobile-sections/{encoded_title}",
            headers=DEFAULT_HEADERS,
            timeout=15
        )

        if response.status_code == 404:
            return {"error": f"Artigo '{title}' não encontrado."}

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def get_featured_article(language: str = "pt") -> Dict[str, Any]:
    """
    Obtém o artigo em destaque do dia.

    Args:
        language: Código do idioma

    Returns:
        Dicionário com dados do artigo em destaque
    """
    _rate_limit()

    base_url = _get_api_url(language)

    # Obtém a data atual
    from datetime import datetime
    today = datetime.now()
    date_str = today.strftime("%Y/%m/%d")

    try:
        response = requests.get(
            f"{base_url}/feed/featured/{date_str}",
            headers=DEFAULT_HEADERS,
            timeout=10
        )

        if response.status_code == 404:
            return {"error": "Artigo em destaque não disponível para esta data/idioma."}

        response.raise_for_status()
        data = response.json()

        # Extrai o artigo em destaque (tfa = today's featured article)
        if "tfa" in data:
            return data["tfa"]
        elif "mostread" in data:
            # Se não tem artigo em destaque, retorna os mais lidos
            return {
                "type": "most_read",
                "articles": data["mostread"].get("articles", [])[:5]
            }

        return data

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


# =============================================================================
# FUNÇÕES DE FORMATAÇÃO
# =============================================================================

def format_summary(data: Dict[str, Any]) -> str:
    """Formata o resumo de um artigo para exibição."""
    if "error" in data:
        return f"❌ {data['error']}"

    title = data.get("title", "Sem título")
    description = data.get("description", "")
    extract = data.get("extract", "Resumo não disponível.")
    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    # Monta a resposta formatada
    output = []
    output.append(f"# 📚 {title}")

    if description:
        output.append(f"*{description}*")

    output.append("")
    output.append("## 📝 Resumo")
    output.append(extract)

    # Thumbnail se disponível
    thumbnail = data.get("thumbnail", {})
    if thumbnail.get("source"):
        output.append("")
        output.append(f"🖼️ **Imagem:** {thumbnail.get('source')}")

    # Link para o artigo completo
    if url:
        output.append("")
        output.append(f"🔗 **Leia mais:** {url}")

    return "\n".join(output)


def format_search_results(results: List[Dict[str, Any]], query: str) -> str:
    """Formata resultados de busca."""
    if not results:
        return f"❌ Nenhum resultado encontrado para '{query}'."

    if "error" in results[0]:
        return f"❌ {results[0]['error']}"

    output = []
    output.append(f"# 🔍 Resultados para: '{query}'")
    output.append("")

    for i, result in enumerate(results, 1):
        title = result.get("title", result.get("normalizedtitle", "Sem título"))
        description = result.get("description", result.get("snippet", ""))

        output.append(f"**{i}. {title}**")
        if description:
            # Limpa HTML se houver
            clean_desc = description.replace("<span class=\"searchmatch\">", "**").replace("</span>", "**")
            output.append(f"   {clean_desc[:200]}...")
        output.append("")

    output.append("---")
    output.append("💡 *Use o título exato para obter mais detalhes.*")

    return "\n".join(output)


# =============================================================================
# SCHEMAS PARA AS TOOLS
# =============================================================================

class WikipediaSummaryInput(BaseModel):
    """Schema para buscar resumo de artigo."""
    topic: str = Field(
        description="O tópico ou título do artigo para buscar na Wikipedia. "
                    "Exemplos: 'Albert Einstein', 'Inteligência Artificial', "
                    "'Brasil', 'Python (linguagem de programação)'"
    )
    language: str = Field(
        default="pt",
        description="Código do idioma: 'pt' (português), 'en' (inglês), "
                    "'es' (espanhol), 'fr' (francês). Padrão: 'pt'"
    )


class WikipediaSearchInput(BaseModel):
    """Schema para pesquisar artigos."""
    query: str = Field(
        description="Termo de busca para encontrar artigos relacionados. "
                    "Exemplos: 'história do Brasil', 'machine learning', 'arte renascentista'"
    )
    language: str = Field(
        default="pt",
        description="Código do idioma para a busca. Padrão: 'pt'"
    )


# =============================================================================
# TOOLS PARA O LANGCHAIN
# =============================================================================

@tool("wikipedia_summary", args_schema=WikipediaSummaryInput)
def wikipedia_summary_tool(topic: str, language: str = "pt") -> str:
    """
    Busca o resumo de um artigo na Wikipedia.

    Use esta ferramenta quando o usuário:
    - Perguntar "o que é X?" ou "quem é Y?"
    - Quiser saber sobre um conceito, pessoa, lugar ou evento
    - Precisar de uma definição ou explicação enciclopédica
    - Perguntar sobre história, ciência, geografia, etc.

    Exemplos de uso:
    - "O que é inteligência artificial?"
    - "Quem foi Albert Einstein?"
    - "Me fale sobre a Torre Eiffel"
    - "O que é fotossíntese?"

    Args:
        topic: O tópico para buscar (pessoa, lugar, conceito, etc.)
        language: Idioma do artigo (pt, en, es, fr, de, it, ja)

    Returns:
        Resumo do artigo da Wikipedia
    """
    # Valida o idioma
    if language.lower() not in WIKIPEDIA_API_URLS:
        language = "pt"

    # Busca o resumo
    data = get_article_summary(topic, language)

    # Se não encontrou, tenta buscar artigos relacionados
    if "error" in data and "não encontrado" in data["error"]:
        # Tenta buscar sugestões
        suggestions = search_articles(topic, language, limit=5)
        if suggestions and "error" not in suggestions[0]:
            suggestion_text = "\n".join([f"• {s.get('title', 'N/A')}" for s in suggestions[:5]])
            return (
                f"❌ Artigo '{topic}' não encontrado na Wikipedia ({language}).\n\n"
                f"💡 **Você quis dizer:**\n{suggestion_text}\n\n"
                f"Tente buscar com um desses termos."
            )

    return format_summary(data)


@tool("wikipedia_search", args_schema=WikipediaSearchInput)
def wikipedia_search_tool(query: str, language: str = "pt") -> str:
    """
    Pesquisa artigos na Wikipedia por um termo.

    Use esta ferramenta quando:
    - O usuário quer explorar um tema mas não sabe o termo exato
    - Precisa encontrar artigos relacionados a um assunto
    - Quer ver opções de artigos sobre um tema

    Exemplos de uso:
    - "Pesquise sobre guerras mundiais"
    - "Encontre artigos sobre física quântica"
    - "Busque informações sobre pintores renascentistas"

    Args:
        query: Termo de busca
        language: Idioma (pt, en, es, fr, de, it, ja)

    Returns:
        Lista de artigos encontrados
    """
    if language.lower() not in WIKIPEDIA_API_URLS:
        language = "pt"

    results = search_articles(query, language, limit=8)
    return format_search_results(results, query)


# =============================================================================
# EXEMPLO DE USO STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📚 TESTE DA TOOL DA WIKIPEDIA")
    print("=" * 60)

    # Teste 1: Resumo de artigo
    print("\n📝 Teste 1: Resumo - Albert Einstein")
    print("-" * 40)
    result = wikipedia_summary_tool.invoke({"topic": "Albert Einstein", "language": "pt"})
    print(result)

    # Teste 2: Pesquisa
    print("\n🔍 Teste 2: Pesquisa - Inteligência Artificial")
    print("-" * 40)
    result = wikipedia_search_tool.invoke({"query": "inteligência artificial", "language": "pt"})
    print(result)

    # Teste 3: Artigo em inglês
    print("\n📝 Teste 3: Resumo em Inglês - Python")
    print("-" * 40)
    result = wikipedia_summary_tool.invoke({"topic": "Python (programming language)", "language": "en"})
    print(result)

