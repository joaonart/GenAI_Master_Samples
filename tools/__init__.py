"""
=============================================================================
TOOLS - Módulo de Ferramentas para os Agentes
=============================================================================

Tools (Ferramentas) permitem que agentes de IA executem ações no mundo real,
como buscar informações, fazer cálculos, acessar APIs, etc.

Conceito: Um agente com tools pode "decidir" quando usar cada ferramenta
baseado na pergunta do usuário.

Exemplo de uso:
    Usuário: "Qual é a temperatura em São Paulo?"
    Agente: (decide usar a tool de clima)
    Tool: (busca a temperatura)
    Agente: "A temperatura em São Paulo é 25°C"

=============================================================================
"""

from .calculator import calculator_tool, CalculatorInput
from .datetime_tool import get_current_datetime, DateTimeInput
from .web_search import web_search_tool, WebSearchInput
from .rag_tool import rag_search_tool, set_vector_store, get_vector_store
from .geocoding import (
    geocode_address_tool,
    reverse_geocode_tool,
    GeocodeInput,
    ReverseGeocodeInput
)
from .crypto import (
    crypto_price_tool,
    top_cryptos_tool,
    CryptoPriceInput,
    TopCryptosInput
)
from .stocks import (
    stock_quote_tool,
    forex_rate_tool,
    StockQuoteInput,
    ForexRateInput
)
from .wikipedia import (
    wikipedia_summary_tool,
    wikipedia_search_tool,
    WikipediaSummaryInput,
    WikipediaSearchInput
)

__all__ = [
    "calculator_tool",
    "CalculatorInput",
    "get_current_datetime",
    "DateTimeInput",
    "web_search_tool",
    "WebSearchInput",
    "rag_search_tool",
    "set_vector_store",
    "get_vector_store",
    "geocode_address_tool",
    "reverse_geocode_tool",
    "GeocodeInput",
    "ReverseGeocodeInput",
    "crypto_price_tool",
    "top_cryptos_tool",
    "CryptoPriceInput",
    "TopCryptosInput",
    "stock_quote_tool",
    "forex_rate_tool",
    "StockQuoteInput",
    "ForexRateInput",
    "wikipedia_summary_tool",
    "wikipedia_search_tool",
    "WikipediaSummaryInput",
    "WikipediaSearchInput",
]

