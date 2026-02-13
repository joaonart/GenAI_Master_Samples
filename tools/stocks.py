"""
=============================================================================
STOCKS & FOREX TOOL - Ferramenta de Cotações de Ações e Câmbio
=============================================================================

Esta ferramenta permite consultar cotações de ações e taxas de câmbio
usando a API do Alpha Vantage.

API: https://www.alphavantage.co/
Documentação: https://www.alphavantage.co/documentation/

IMPORTANTE - API Key:
- Requer API key gratuita (cadastro em https://www.alphavantage.co/support/#api-key)
- Limite gratuito: 25 requisições por dia
- Configure a variável de ambiente: ALPHA_VANTAGE_API_KEY

Funcionalidades:
1. Cotação atual de ações (AAPL, GOOGL, MSFT, etc.)
2. Cotação de ações brasileiras (PETR4.SAO, VALE3.SAO, etc.)
3. Taxa de câmbio entre moedas (USD/BRL, EUR/USD, etc.)
4. Informações da empresa (setor, descrição, etc.)

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import os
import requests
import time
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# URL base da API Alpha Vantage
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Controle de rate limiting
_last_request_time = 0


def _get_api_key() -> str:
    """Obtém a API key do Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ API Key do Alpha Vantage não encontrada!\n"
            "1. Cadastre-se em: https://www.alphavantage.co/support/#api-key\n"
            "2. Configure a variável: ALPHA_VANTAGE_API_KEY"
        )
    return api_key


def _rate_limit():
    """
    Implementa rate limiting para respeitar os limites da API.
    Garante no mínimo 12 segundos entre requisições (5 por minuto no plano gratuito).
    """
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < 12.0:
        time.sleep(12.0 - time_since_last)

    _last_request_time = time.time()


# =============================================================================
# MAPEAMENTO DE SÍMBOLOS POPULARES
# =============================================================================

# Ações brasileiras (B3) - precisam do sufixo .SAO
BRAZILIAN_STOCKS = {
    "petr4": "PETR4.SAO",
    "petrobras": "PETR4.SAO",
    "vale3": "VALE3.SAO",
    "vale": "VALE3.SAO",
    "itub4": "ITUB4.SAO",
    "itau": "ITUB4.SAO",
    "bbdc4": "BBDC4.SAO",
    "bradesco": "BBDC4.SAO",
    "abev3": "ABEV3.SAO",
    "ambev": "ABEV3.SAO",
    "wege3": "WEGE3.SAO",
    "weg": "WEGE3.SAO",
    "bbas3": "BBAS3.SAO",
    "banco do brasil": "BBAS3.SAO",
    "mglu3": "MGLU3.SAO",
    "magalu": "MGLU3.SAO",
    "magazine luiza": "MGLU3.SAO",
    "b3sa3": "B3SA3.SAO",
    "b3": "B3SA3.SAO",
    "rent3": "RENT3.SAO",
    "localiza": "RENT3.SAO",
    "suzb3": "SUZB3.SAO",
    "suzano": "SUZB3.SAO",
    "jbss3": "JBSS3.SAO",
    "jbs": "JBSS3.SAO",
    "elet3": "ELET3.SAO",
    "eletrobras": "ELET3.SAO",
    "lren3": "LREN3.SAO",
    "renner": "LREN3.SAO",
    "lojas renner": "LREN3.SAO",
    "hapv3": "HAPV3.SAO",
    "hapvida": "HAPV3.SAO",
    "rdor3": "RDOR3.SAO",
    "rede dor": "RDOR3.SAO",
    "rail3": "RAIL3.SAO",
    "rumo": "RAIL3.SAO",
    "vivt3": "VIVT3.SAO",
    "vivo": "VIVT3.SAO",
    "telefonica": "VIVT3.SAO",
    "tots3": "TOTS3.SAO",
    "totvs": "TOTS3.SAO",
    "prio3": "PRIO3.SAO",
    "prio": "PRIO3.SAO",
    "petro rio": "PRIO3.SAO",
}

# Ações americanas populares
US_STOCKS = {
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "disney": "DIS",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "mcdonalds": "MCD",
    "nike": "NKE",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "visa": "V",
    "mastercard": "MA",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "berkshire": "BRK.B",
    "berkshire hathaway": "BRK.B",
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "starbucks": "SBUX",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "robinhood": "HOOD",
}

# Moedas para forex
CURRENCY_CODES = {
    "dolar": "USD",
    "dólar": "USD",
    "dollar": "USD",
    "real": "BRL",
    "reais": "BRL",
    "euro": "EUR",
    "libra": "GBP",
    "pound": "GBP",
    "iene": "JPY",
    "yen": "JPY",
    "yuan": "CNY",
    "renminbi": "CNY",
    "franco suico": "CHF",
    "franco suíço": "CHF",
    "peso argentino": "ARS",
    "peso mexicano": "MXN",
    "dolar canadense": "CAD",
    "dolar australiano": "AUD",
    "won": "KRW",
    "rupia": "INR",
    "bitcoin": "BTC",
    "btc": "BTC",
}


def _normalize_stock_symbol(symbol: str) -> str:
    """Normaliza o símbolo da ação."""
    symbol_lower = symbol.lower().strip()

    # Verifica se é uma ação brasileira
    if symbol_lower in BRAZILIAN_STOCKS:
        return BRAZILIAN_STOCKS[symbol_lower]

    # Verifica se é uma ação americana pelo nome
    if symbol_lower in US_STOCKS:
        return US_STOCKS[symbol_lower]

    # Se já tem sufixo .SAO, retorna em maiúsculas
    if ".sao" in symbol_lower:
        return symbol.upper()

    # Retorna o símbolo em maiúsculas
    return symbol.upper()


def _normalize_currency(currency: str) -> str:
    """Normaliza o código da moeda."""
    currency_lower = currency.lower().strip()
    return CURRENCY_CODES.get(currency_lower, currency.upper())


# =============================================================================
# FUNÇÕES DE CONSULTA À API
# =============================================================================

def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """
    Obtém a cotação atual de uma ação.

    Args:
        symbol: Símbolo da ação (ex: AAPL, PETR4.SAO)

    Returns:
        Dicionário com dados da cotação
    """
    _rate_limit()

    try:
        api_key = _get_api_key()
    except ValueError as e:
        return {"error": str(e)}

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        # Verifica erros da API
        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data:
            return {"error": "Limite de requisições atingido. Tente novamente mais tarde."}
        if "Global Quote" not in data or not data["Global Quote"]:
            return {"error": f"Ação '{symbol}' não encontrada."}

        return data["Global Quote"]

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def get_company_overview(symbol: str) -> Dict[str, Any]:
    """
    Obtém informações detalhadas de uma empresa.

    Args:
        symbol: Símbolo da ação

    Returns:
        Dicionário com dados da empresa
    """
    _rate_limit()

    try:
        api_key = _get_api_key()
    except ValueError as e:
        return {"error": str(e)}

    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data:
            return {"error": "Limite de requisições atingido."}
        if not data or "Symbol" not in data:
            return {"error": f"Informações não disponíveis para '{symbol}'."}

        return data

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def get_forex_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    Obtém a taxa de câmbio entre duas moedas.

    Args:
        from_currency: Moeda de origem (ex: USD)
        to_currency: Moeda de destino (ex: BRL)

    Returns:
        Dicionário com dados do câmbio
    """
    _rate_limit()

    try:
        api_key = _get_api_key()
    except ValueError as e:
        return {"error": str(e)}

    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "apikey": api_key
    }

    try:
        response = requests.get(
            ALPHA_VANTAGE_BASE_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data:
            return {"error": "Limite de requisições atingido."}
        if "Realtime Currency Exchange Rate" not in data:
            return {"error": f"Câmbio {from_currency}/{to_currency} não encontrado."}

        return data["Realtime Currency Exchange Rate"]

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


# =============================================================================
# FUNÇÕES DE FORMATAÇÃO
# =============================================================================

def format_price(value: float, currency: str = "USD") -> str:
    """Formata um valor monetário."""
    currency = currency.upper()
    if currency == "BRL":
        return f"R$ {value:,.2f}"
    elif currency == "USD":
        return f"$ {value:,.2f}"
    elif currency == "EUR":
        return f"€ {value:,.2f}"
    elif currency == "GBP":
        return f"£ {value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_large_number(value: float) -> str:
    """Formata números grandes."""
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
    """Formata porcentagem com emoji."""
    if value > 0:
        return f"📈 +{value:.2f}%"
    elif value < 0:
        return f"📉 {value:.2f}%"
    return f"➡️ {value:.2f}%"


# =============================================================================
# SCHEMAS PARA AS TOOLS
# =============================================================================

class StockQuoteInput(BaseModel):
    """Schema de entrada para cotação de ação."""
    symbol: str = Field(
        description="Símbolo ou nome da ação. "
                    "Exemplos: 'AAPL', 'Apple', 'GOOGL', 'Microsoft', 'PETR4', 'Petrobras', 'VALE3', 'Vale'. "
                    "Para ações brasileiras, pode usar o código (PETR4) ou nome da empresa."
    )


class ForexRateInput(BaseModel):
    """Schema de entrada para taxa de câmbio."""
    from_currency: str = Field(
        description="Moeda de origem. Exemplos: 'USD', 'dólar', 'EUR', 'euro', 'BRL', 'real'"
    )
    to_currency: str = Field(
        description="Moeda de destino. Exemplos: 'BRL', 'real', 'USD', 'dólar', 'EUR', 'euro'"
    )


# =============================================================================
# TOOLS PARA O LANGCHAIN
# =============================================================================

@tool("stock_quote", args_schema=StockQuoteInput)
def stock_quote_tool(symbol: str) -> str:
    """
    Consulta a cotação atual de uma ação (brasileira ou americana).

    Use esta ferramenta quando o usuário:
    - Perguntar o preço de uma ação
    - Quiser saber a cotação de uma empresa na bolsa
    - Perguntar sobre ações como Apple, Google, Petrobras, Vale, etc.
    - Quiser saber como está uma ação hoje

    Exemplos de uso:
    - "Qual o preço da ação da Apple?"
    - "Como está a Petrobras hoje?"
    - "Cotação da VALE3"
    - "Preço das ações da Microsoft"
    - "Quanto está a Tesla?"

    Args:
        symbol: Símbolo ou nome da ação

    Returns:
        Informações da cotação da ação
    """
    # Normaliza o símbolo
    normalized_symbol = _normalize_stock_symbol(symbol)

    # Busca a cotação
    quote = get_stock_quote(normalized_symbol)

    if "error" in quote:
        return f"❌ {quote['error']}"

    # Extrai os dados
    try:
        price = float(quote.get("05. price", 0))
        change = float(quote.get("09. change", 0))
        change_percent = quote.get("10. change percent", "0%").replace("%", "")
        change_percent = float(change_percent)
        volume = int(quote.get("06. volume", 0))
        high = float(quote.get("03. high", 0))
        low = float(quote.get("04. low", 0))
        prev_close = float(quote.get("08. previous close", 0))
        open_price = float(quote.get("02. open", 0))
    except (ValueError, TypeError):
        return f"❌ Erro ao processar dados da ação '{symbol}'."

    # Determina a moeda (BRL para ações brasileiras)
    currency = "BRL" if ".SAO" in normalized_symbol else "USD"

    # Monta a resposta
    output_parts = []

    # Header
    output_parts.append(f"# 📊 {normalized_symbol}")
    output_parts.append("")

    # Preço atual
    output_parts.append("## 💰 Cotação Atual")
    output_parts.append(f"• **Preço:** {format_price(price, currency)}")
    output_parts.append(f"• **Variação:** {format_percentage(change_percent)} ({format_price(change, currency)})")
    output_parts.append("")

    # Dados do dia
    output_parts.append("## 📈 Dados do Dia")
    output_parts.append(f"• **Abertura:** {format_price(open_price, currency)}")
    output_parts.append(f"• **Máxima:** {format_price(high, currency)}")
    output_parts.append(f"• **Mínima:** {format_price(low, currency)}")
    output_parts.append(f"• **Fech. Anterior:** {format_price(prev_close, currency)}")
    output_parts.append(f"• **Volume:** {format_large_number(volume)}")

    # Tenta buscar informações da empresa (apenas para ações americanas)
    if ".SAO" not in normalized_symbol:
        overview = get_company_overview(normalized_symbol)
        if "error" not in overview and overview:
            output_parts.append("")
            output_parts.append("## 🏢 Sobre a Empresa")

            name = overview.get("Name", "")
            sector = overview.get("Sector", "")
            industry = overview.get("Industry", "")
            market_cap = overview.get("MarketCapitalization", "")
            pe_ratio = overview.get("PERatio", "")
            dividend_yield = overview.get("DividendYield", "")

            if name:
                output_parts.append(f"• **Nome:** {name}")
            if sector:
                output_parts.append(f"• **Setor:** {sector}")
            if industry:
                output_parts.append(f"• **Indústria:** {industry}")
            if market_cap:
                try:
                    mc = float(market_cap)
                    output_parts.append(f"• **Market Cap:** ${format_large_number(mc)}")
                except ValueError:
                    pass
            if pe_ratio and pe_ratio != "None":
                output_parts.append(f"• **P/E Ratio:** {pe_ratio}")
            if dividend_yield and dividend_yield != "None":
                try:
                    dy = float(dividend_yield) * 100
                    output_parts.append(f"• **Dividend Yield:** {dy:.2f}%")
                except ValueError:
                    pass

    output_parts.append("")
    output_parts.append("---")
    output_parts.append("*Dados fornecidos por Alpha Vantage*")

    return "\n".join(output_parts)


@tool("forex_rate", args_schema=ForexRateInput)
def forex_rate_tool(from_currency: str, to_currency: str) -> str:
    """
    Consulta a taxa de câmbio entre duas moedas.

    Use esta ferramenta quando o usuário:
    - Perguntar a cotação do dólar, euro, etc.
    - Quiser converter valores entre moedas
    - Perguntar quanto vale uma moeda em relação a outra
    - Quiser saber a taxa de câmbio atual

    Exemplos de uso:
    - "Qual a cotação do dólar hoje?"
    - "Quanto está o euro em reais?"
    - "Taxa de câmbio USD/BRL"
    - "Converter dólar para real"
    - "Cotação da libra"

    Args:
        from_currency: Moeda de origem (USD, EUR, BRL, etc.)
        to_currency: Moeda de destino (BRL, USD, EUR, etc.)

    Returns:
        Taxa de câmbio entre as moedas
    """
    # Normaliza as moedas
    from_curr = _normalize_currency(from_currency)
    to_curr = _normalize_currency(to_currency)

    # Se o usuário só perguntou "cotação do dólar" sem destino, assume BRL
    if to_curr == from_curr:
        to_curr = "BRL" if from_curr != "BRL" else "USD"

    # Busca a taxa de câmbio
    rate_data = get_forex_rate(from_curr, to_curr)

    if "error" in rate_data:
        return f"❌ {rate_data['error']}"

    try:
        from_code = rate_data.get("1. From_Currency Code", from_curr)
        from_name = rate_data.get("2. From_Currency Name", "")
        to_code = rate_data.get("3. To_Currency Code", to_curr)
        to_name = rate_data.get("4. To_Currency Name", "")
        rate = float(rate_data.get("5. Exchange Rate", 0))
        last_refreshed = rate_data.get("6. Last Refreshed", "")
        bid_price = rate_data.get("8. Bid Price", "")
        ask_price = rate_data.get("9. Ask Price", "")
    except (ValueError, TypeError):
        return f"❌ Erro ao processar dados de câmbio."

    # Monta a resposta
    output_parts = []

    # Header
    output_parts.append(f"# 💱 {from_code}/{to_code}")
    if from_name and to_name:
        output_parts.append(f"*{from_name} → {to_name}*")
    output_parts.append("")

    # Taxa atual
    output_parts.append("## 💰 Taxa de Câmbio")
    output_parts.append(f"• **1 {from_code}** = **{rate:,.4f} {to_code}**")
    output_parts.append("")

    # Exemplo de conversão
    output_parts.append("## 🔄 Exemplos de Conversão")
    examples = [1, 10, 100, 1000]
    for amount in examples:
        converted = amount * rate
        output_parts.append(f"• {amount:,} {from_code} = {converted:,.2f} {to_code}")

    # Bid/Ask se disponível
    if bid_price and ask_price:
        output_parts.append("")
        output_parts.append("## 📊 Spread")
        try:
            bid = float(bid_price)
            ask = float(ask_price)
            spread = ask - bid
            output_parts.append(f"• **Bid (Compra):** {bid:,.4f}")
            output_parts.append(f"• **Ask (Venda):** {ask:,.4f}")
            output_parts.append(f"• **Spread:** {spread:,.4f}")
        except ValueError:
            pass

    if last_refreshed:
        output_parts.append("")
        output_parts.append(f"*Atualizado em: {last_refreshed}*")

    output_parts.append("")
    output_parts.append("---")
    output_parts.append("*Dados fornecidos por Alpha Vantage*")

    return "\n".join(output_parts)


# =============================================================================
# EXEMPLO DE USO STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📊 TESTE DA TOOL DE AÇÕES E FOREX")
    print("=" * 60)

    # Verifica se a API key está configurada
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("\n⚠️  ALPHA_VANTAGE_API_KEY não configurada!")
        print("Configure a variável de ambiente para testar.")
        print("Cadastre-se em: https://www.alphavantage.co/support/#api-key")
    else:
        # Teste 1: Cotação de ação americana
        print("\n📈 Teste 1: Cotação da Apple (AAPL)")
        print("-" * 40)
        result = stock_quote_tool.invoke({"symbol": "AAPL"})
        print(result)

        # Teste 2: Taxa de câmbio
        print("\n💱 Teste 2: Cotação do Dólar")
        print("-" * 40)
        result = forex_rate_tool.invoke({"from_currency": "USD", "to_currency": "BRL"})
        print(result)

        # Teste 3: Ação brasileira
        print("\n📈 Teste 3: Cotação da Petrobras")
        print("-" * 40)
        result = stock_quote_tool.invoke({"symbol": "Petrobras"})
        print(result)

