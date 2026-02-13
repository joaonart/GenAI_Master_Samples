"""
=============================================================================
CRYPTO TOOL - Ferramenta de Consulta de Criptomoedas
=============================================================================

Esta ferramenta permite consultar informações sobre criptomoedas usando
a API gratuita do CoinGecko.

API: https://www.coingecko.com/en/api
Documentação: https://docs.coingecko.com/reference/introduction

IMPORTANTE - Limites da API Gratuita:
- 10-30 requisições por minuto
- Sem necessidade de API key
- Dados com delay de alguns minutos

Funcionalidades:
1. Consultar preço atual de uma criptomoeda
2. Listar top criptomoedas por market cap
3. Consultar dados detalhados de uma moeda
4. Converter valores entre criptomoedas e moedas fiat

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import requests
import time
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# URL base da API CoinGecko
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Headers padrão para as requisições
DEFAULT_HEADERS = {
    "Accept": "application/json"
}

# Controle de rate limiting
_last_request_time = 0


def _rate_limit():
    """
    Implementa rate limiting para respeitar os limites da API.
    Garante no mínimo 2 segundos entre requisições para segurança.
    """
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < 2.0:
        time.sleep(2.0 - time_since_last)

    _last_request_time = time.time()


# =============================================================================
# MAPEAMENTO DE MOEDAS POPULARES
# =============================================================================

# IDs do CoinGecko para moedas populares (facilita a busca)
POPULAR_COINS = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "cardano": "cardano",
    "ada": "cardano",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "polkadot": "polkadot",
    "dot": "polkadot",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
    "chainlink": "chainlink",
    "link": "chainlink",
    "polygon": "matic-network",
    "matic": "matic-network",
    "litecoin": "litecoin",
    "ltc": "litecoin",
    "uniswap": "uniswap",
    "uni": "uniswap",
    "stellar": "stellar",
    "xlm": "stellar",
    "tether": "tether",
    "usdt": "tether",
    "usd coin": "usd-coin",
    "usdc": "usd-coin",
    "binance coin": "binancecoin",
    "bnb": "binancecoin",
    "tron": "tron",
    "trx": "tron",
    "shiba inu": "shiba-inu",
    "shib": "shiba-inu",
}

# Moedas fiat suportadas
SUPPORTED_CURRENCIES = [
    "usd", "eur", "gbp", "brl", "jpy", "cny", "krw",
    "inr", "cad", "aud", "chf", "mxn", "ars"
]


def _normalize_coin_id(coin: str) -> str:
    """
    Normaliza o nome/símbolo da moeda para o ID do CoinGecko.

    Args:
        coin: Nome ou símbolo da moeda

    Returns:
        ID da moeda no CoinGecko
    """
    coin_lower = coin.lower().strip()
    return POPULAR_COINS.get(coin_lower, coin_lower)


# =============================================================================
# FUNÇÕES DE CONSULTA À API
# =============================================================================

def get_coin_price(
    coin_id: str,
    currencies: Optional[List[str]] = None,
    include_24h_change: bool = True,
    include_market_cap: bool = True
) -> Dict[str, Any]:
    """
    Obtém o preço atual de uma criptomoeda.

    Args:
        coin_id: ID da moeda no CoinGecko
        currencies: Lista de moedas fiat para conversão
        include_24h_change: Incluir variação de 24h
        include_market_cap: Incluir market cap

    Returns:
        Dicionário com os dados de preço
    """
    _rate_limit()

    if currencies is None:
        currencies = ["usd", "brl"]

    params = {
        "ids": coin_id,
        "vs_currencies": ",".join(currencies),
        "include_24hr_change": str(include_24h_change).lower(),
        "include_market_cap": str(include_market_cap).lower()
    }

    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/simple/price",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def get_coin_details(coin_id: str) -> Dict[str, Any]:
    """
    Obtém informações detalhadas de uma criptomoeda.

    Args:
        coin_id: ID da moeda no CoinGecko

    Returns:
        Dicionário com dados detalhados da moeda
    """
    _rate_limit()

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false"
    }

    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/coins/{coin_id}",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def get_top_coins(
    limit: int = 10,
    currency: str = "usd"
) -> List[Dict[str, Any]]:
    """
    Lista as top criptomoedas por market cap.

    Args:
        limit: Número de moedas a retornar (max 250)
        currency: Moeda fiat para os valores

    Returns:
        Lista de dicionários com dados das moedas
    """
    _rate_limit()

    params = {
        "vs_currency": currency,
        "order": "market_cap_desc",
        "per_page": min(limit, 250),
        "page": 1,
        "sparkline": "false"
    }

    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/coins/markets",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return [{"error": f"Erro na requisição: {str(e)}"}]


def search_coin(query: str) -> List[Dict[str, Any]]:
    """
    Busca criptomoedas por nome ou símbolo.

    Args:
        query: Termo de busca

    Returns:
        Lista de moedas encontradas
    """
    _rate_limit()

    try:
        response = requests.get(
            f"{COINGECKO_BASE_URL}/search",
            params={"query": query},
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("coins", [])[:10]  # Limita a 10 resultados

    except requests.exceptions.RequestException as e:
        return [{"error": f"Erro na requisição: {str(e)}"}]


# =============================================================================
# FUNÇÕES DE FORMATAÇÃO
# =============================================================================

def format_price(value: float, currency: str = "usd") -> str:
    """Formata um valor monetário."""
    if currency.lower() == "brl":
        return f"R$ {value:,.2f}"
    elif currency.lower() == "usd":
        return f"$ {value:,.2f}"
    elif currency.lower() == "eur":
        return f"€ {value:,.2f}"
    else:
        return f"{value:,.2f} {currency.upper()}"


def format_large_number(value: float) -> str:
    """Formata números grandes de forma legível."""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def format_percentage(value: float) -> str:
    """Formata porcentagem com emoji indicativo."""
    if value > 0:
        return f"📈 +{value:.2f}%"
    elif value < 0:
        return f"📉 {value:.2f}%"
    return f"➡️ {value:.2f}%"


# =============================================================================
# SCHEMAS PARA AS TOOLS
# =============================================================================

class CryptoPriceInput(BaseModel):
    """Schema de entrada para consulta de preço de criptomoeda."""
    coin: str = Field(
        description="Nome ou símbolo da criptomoeda. "
                    "Exemplos: 'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', "
                    "'dogecoin', 'doge', 'cardano', 'ada'"
    )
    currency: str = Field(
        default="usd",
        description="Moeda fiat para conversão. "
                    "Opções: 'usd' (dólar), 'brl' (real), 'eur' (euro), etc."
    )


class TopCryptosInput(BaseModel):
    """Schema de entrada para listar top criptomoedas."""
    limit: int = Field(
        default=10,
        description="Número de criptomoedas a listar (1-50). Padrão: 10"
    )
    currency: str = Field(
        default="usd",
        description="Moeda fiat para os valores. "
                    "Opções: 'usd' (dólar), 'brl' (real), 'eur' (euro)"
    )


# =============================================================================
# TOOLS PARA O LANGCHAIN
# =============================================================================

@tool("crypto_price", args_schema=CryptoPriceInput)
def crypto_price_tool(coin: str, currency: str = "usd") -> str:
    """
    Consulta o preço atual de uma criptomoeda.

    Use esta ferramenta quando o usuário:
    - Perguntar o preço de uma criptomoeda
    - Quiser saber quanto vale Bitcoin, Ethereum, etc.
    - Perguntar sobre variação de preço de crypto
    - Quiser converter crypto para moeda fiat

    Exemplos de uso:
    - "Qual o preço do Bitcoin?"
    - "Quanto vale 1 Ethereum em reais?"
    - "Como está o Solana hoje?"
    - "Preço do Dogecoin em dólares"

    Args:
        coin: Nome ou símbolo da criptomoeda (btc, eth, sol, etc.)
        currency: Moeda fiat (usd, brl, eur)

    Returns:
        Informações de preço da criptomoeda
    """
    # Normaliza o ID da moeda
    coin_id = _normalize_coin_id(coin)
    currency = currency.lower()

    if currency not in SUPPORTED_CURRENCIES:
        currency = "usd"

    # Busca preço simples primeiro
    price_data = get_coin_price(coin_id, currencies=[currency, "usd", "brl"])

    if "error" in price_data:
        # Tenta buscar a moeda
        search_results = search_coin(coin)
        if search_results and "error" not in search_results[0]:
            suggestions = [f"{c['name']} ({c['symbol'].upper()})" for c in search_results[:5]]
            return f"❌ Moeda '{coin}' não encontrada.\n\n💡 Você quis dizer:\n• " + "\n• ".join(suggestions)
        return f"❌ Erro ao buscar preço: {price_data.get('error', 'Moeda não encontrada')}"

    if coin_id not in price_data:
        return f"❌ Moeda '{coin}' não encontrada. Verifique o nome ou símbolo."

    data = price_data[coin_id]

    # Busca detalhes adicionais
    details = get_coin_details(coin_id)

    # Monta a resposta
    output_parts = []

    # Nome e símbolo
    if "error" not in details:
        name = details.get("name", coin_id.title())
        symbol = details.get("symbol", "").upper()
        output_parts.append(f"# 🪙 {name} ({symbol})")
    else:
        output_parts.append(f"# 🪙 {coin_id.title()}")

    output_parts.append("")

    # Preços
    output_parts.append("## 💰 Preços Atuais")

    for curr in ["usd", "brl"]:
        if curr in data:
            price = data[curr]
            change_key = f"{curr}_24h_change"
            market_cap_key = f"{curr}_market_cap"

            price_str = format_price(price, curr)
            output_parts.append(f"• **{curr.upper()}:** {price_str}")

            if change_key in data and data[change_key] is not None:
                change_str = format_percentage(data[change_key])
                output_parts.append(f"  └ Variação 24h: {change_str}")

    # Market Cap
    if "error" not in details:
        market_data = details.get("market_data", {})

        output_parts.append("")
        output_parts.append("## 📊 Dados de Mercado")

        # Market Cap
        market_cap = market_data.get("market_cap", {}).get("usd")
        if market_cap:
            output_parts.append(f"• **Market Cap:** ${format_large_number(market_cap)}")

        # Volume 24h
        volume = market_data.get("total_volume", {}).get("usd")
        if volume:
            output_parts.append(f"• **Volume 24h:** ${format_large_number(volume)}")

        # Ranking
        rank = details.get("market_cap_rank")
        if rank:
            output_parts.append(f"• **Ranking:** #{rank}")

        # ATH (All Time High)
        ath = market_data.get("ath", {}).get("usd")
        ath_change = market_data.get("ath_change_percentage", {}).get("usd")
        if ath:
            output_parts.append(f"• **ATH:** ${ath:,.2f} ({ath_change:.1f}% do ATH)")

        # Circulação
        circulating = market_data.get("circulating_supply")
        max_supply = market_data.get("max_supply")
        if circulating:
            supply_str = format_large_number(circulating)
            if max_supply:
                supply_str += f" / {format_large_number(max_supply)}"
            output_parts.append(f"• **Supply:** {supply_str}")

    return "\n".join(output_parts)


@tool("top_cryptos", args_schema=TopCryptosInput)
def top_cryptos_tool(limit: int = 10, currency: str = "usd") -> str:
    """
    Lista as principais criptomoedas por capitalização de mercado.

    Use esta ferramenta quando o usuário:
    - Perguntar quais são as maiores criptomoedas
    - Quiser ver um ranking de cryptos
    - Perguntar sobre o mercado de criptomoedas em geral
    - Quiser saber as top 10, top 20 moedas

    Exemplos de uso:
    - "Quais são as top 10 criptomoedas?"
    - "Liste as maiores cryptos por market cap"
    - "Ranking das criptomoedas"
    - "Quais as principais moedas digitais?"

    Args:
        limit: Número de moedas a listar (1-50)
        currency: Moeda fiat para os valores (usd, brl, eur)

    Returns:
        Lista das top criptomoedas com preços e variações
    """
    limit = max(1, min(limit, 50))
    currency = currency.lower()

    if currency not in SUPPORTED_CURRENCIES:
        currency = "usd"

    coins = get_top_coins(limit=limit, currency=currency)

    if not coins or "error" in coins[0]:
        return "❌ Erro ao buscar dados das criptomoedas. Tente novamente."

    # Símbolo da moeda
    currency_symbols = {"usd": "$", "brl": "R$", "eur": "€"}
    symbol = currency_symbols.get(currency, currency.upper() + " ")

    # Monta a resposta
    output_parts = [
        f"# 🏆 Top {limit} Criptomoedas por Market Cap",
        f"*Valores em {currency.upper()}*",
        ""
    ]

    for i, coin in enumerate(coins, 1):
        name = coin.get("name", "Unknown")
        ticker = coin.get("symbol", "???").upper()
        price = coin.get("current_price", 0)
        change_24h = coin.get("price_change_percentage_24h", 0)
        market_cap = coin.get("market_cap", 0)

        # Emoji baseado na variação
        if change_24h and change_24h > 0:
            trend = "📈"
        elif change_24h and change_24h < 0:
            trend = "📉"
        else:
            trend = "➡️"

        # Formata variação
        change_str = f"+{change_24h:.2f}%" if change_24h and change_24h > 0 else f"{change_24h:.2f}%" if change_24h else "N/A"

        output_parts.append(
            f"**{i}. {name} ({ticker})** {trend}\n"
            f"   💰 {symbol}{price:,.2f} | 24h: {change_str} | MCap: {symbol}{format_large_number(market_cap)}"
        )
        output_parts.append("")

    # Adiciona timestamp
    output_parts.append("---")
    output_parts.append("*Dados fornecidos por CoinGecko*")

    return "\n".join(output_parts)


# =============================================================================
# EXEMPLO DE USO STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🪙 TESTE DA TOOL DE CRIPTOMOEDAS")
    print("=" * 60)

    # Teste 1: Preço do Bitcoin
    print("\n💰 Teste 1: Preço do Bitcoin")
    print("-" * 40)
    result = crypto_price_tool.invoke({"coin": "bitcoin", "currency": "usd"})
    print(result)

    # Teste 2: Top 5 criptomoedas
    print("\n🏆 Teste 2: Top 5 Criptomoedas")
    print("-" * 40)
    result = top_cryptos_tool.invoke({"limit": 5, "currency": "brl"})
    print(result)

    # Teste 3: Ethereum em reais
    print("\n💰 Teste 3: Ethereum em Reais")
    print("-" * 40)
    result = crypto_price_tool.invoke({"coin": "eth", "currency": "brl"})
    print(result)

