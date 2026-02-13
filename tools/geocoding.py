"""
=============================================================================
GEOCODING TOOL - Ferramenta de Geocodificação
=============================================================================

Esta ferramenta permite converter endereços em coordenadas geográficas
(geocoding) e coordenadas em endereços (reverse geocoding).

Usa a API gratuita do Nominatim (OpenStreetMap):
https://nominatim.org/release-docs/latest/api/Overview/

IMPORTANTE - Política de Uso do Nominatim:
- Máximo 1 requisição por segundo
- Identificar a aplicação via User-Agent
- Não usar para uso comercial pesado sem permissão
- Mais detalhes: https://operations.osmfoundation.org/policies/nominatim/

Funcionalidades:
1. Geocoding: Endereço → Coordenadas (lat, lon)
2. Reverse Geocoding: Coordenadas → Endereço

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import requests
import time
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# URL base da API Nominatim
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"

# User-Agent obrigatório pela política do Nominatim
USER_AGENT = "GenAI_Master_Samples/1.0 (Educational Project)"

# Headers padrão para as requisições
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json"
}

# Controle de rate limiting (1 req/segundo)
_last_request_time = 0


def _rate_limit():
    """
    Implementa rate limiting para respeitar a política do Nominatim.
    Garante no mínimo 1 segundo entre requisições.
    """
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < 1.0:
        time.sleep(1.0 - time_since_last)

    _last_request_time = time.time()


# =============================================================================
# FUNÇÕES DE GEOCODING
# =============================================================================

def geocode_address(
    address: str,
    limit: int = 5,
    country_codes: Optional[str] = None,
    language: str = "pt-BR"
) -> List[Dict[str, Any]]:
    """
    Converte um endereço em coordenadas geográficas.

    Args:
        address: Endereço a ser geocodificado (ex: "Av. Paulista, São Paulo")
        limit: Número máximo de resultados (1-50)
        country_codes: Códigos de países para filtrar (ex: "br,pt")
        language: Idioma preferido para os resultados

    Returns:
        Lista de dicionários com os resultados encontrados
    """
    _rate_limit()

    params = {
        "q": address,
        "format": "json",
        "limit": min(limit, 50),
        "addressdetails": 1,
        "accept-language": language
    }

    if country_codes:
        params["countrycodes"] = country_codes

    try:
        response = requests.get(
            f"{NOMINATIM_BASE_URL}/search",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return [{"error": f"Erro na requisição: {str(e)}"}]


def reverse_geocode(
    latitude: float,
    longitude: float,
    zoom: int = 18,
    language: str = "pt-BR"
) -> Dict[str, Any]:
    """
    Converte coordenadas geográficas em um endereço.

    Args:
        latitude: Latitude (-90 a 90)
        longitude: Longitude (-180 a 180)
        zoom: Nível de detalhe (0-18, maior = mais detalhado)
              3 = país, 10 = cidade, 14 = bairro, 18 = edifício
        language: Idioma preferido para os resultados

    Returns:
        Dicionário com o endereço encontrado
    """
    _rate_limit()

    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
        "zoom": zoom,
        "addressdetails": 1,
        "accept-language": language
    }

    try:
        response = requests.get(
            f"{NOMINATIM_BASE_URL}/reverse",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}"}


def format_geocode_result(result: Dict[str, Any]) -> str:
    """
    Formata um resultado de geocoding para exibição amigável.

    Args:
        result: Resultado do geocoding

    Returns:
        String formatada com as informações
    """
    if "error" in result:
        return f"❌ {result['error']}"

    display_name = result.get("display_name", "Endereço não disponível")
    lat = result.get("lat", "N/A")
    lon = result.get("lon", "N/A")
    osm_type = result.get("type", "unknown")

    # Extrai detalhes do endereço se disponíveis
    address = result.get("address", {})

    parts = []
    parts.append(f"📍 **Endereço:** {display_name}")
    parts.append(f"🌐 **Coordenadas:** {lat}, {lon}")
    parts.append(f"📂 **Tipo:** {osm_type}")

    # Adiciona detalhes do endereço se disponíveis
    if address:
        details = []
        if "road" in address:
            details.append(f"Rua: {address['road']}")
        if "house_number" in address:
            details.append(f"Número: {address['house_number']}")
        if "suburb" in address or "neighbourhood" in address:
            bairro = address.get("suburb") or address.get("neighbourhood")
            details.append(f"Bairro: {bairro}")
        if "city" in address or "town" in address or "municipality" in address:
            cidade = address.get("city") or address.get("town") or address.get("municipality")
            details.append(f"Cidade: {cidade}")
        if "state" in address:
            details.append(f"Estado: {address['state']}")
        if "country" in address:
            details.append(f"País: {address['country']}")
        if "postcode" in address:
            details.append(f"CEP: {address['postcode']}")

        if details:
            parts.append("📋 **Detalhes:** " + " | ".join(details))

    return "\n".join(parts)


# =============================================================================
# SCHEMAS PARA AS TOOLS
# =============================================================================

class GeocodeInput(BaseModel):
    """Schema de entrada para geocoding (endereço → coordenadas)."""
    address: str = Field(
        description="O endereço a ser convertido em coordenadas. "
                    "Pode ser um endereço completo (ex: 'Av. Paulista, 1000, São Paulo, SP') "
                    "ou parcial (ex: 'Torre Eiffel, Paris')."
    )
    country_codes: Optional[str] = Field(
        default=None,
        description="Códigos ISO 3166-1 alpha-2 dos países para filtrar a busca, "
                    "separados por vírgula. Ex: 'br' para Brasil, 'br,pt' para Brasil e Portugal."
    )


class ReverseGeocodeInput(BaseModel):
    """Schema de entrada para reverse geocoding (coordenadas → endereço)."""
    latitude: float = Field(
        description="Latitude em graus decimais (-90 a 90). Ex: -23.5505"
    )
    longitude: float = Field(
        description="Longitude em graus decimais (-180 a 180). Ex: -46.6333"
    )


# =============================================================================
# TOOLS PARA O LANGCHAIN
# =============================================================================

@tool("geocode_address", args_schema=GeocodeInput)
def geocode_address_tool(address: str, country_codes: Optional[str] = None) -> str:
    """
    Converte um endereço em coordenadas geográficas (latitude e longitude).

    Use esta ferramenta quando o usuário:
    - Perguntar as coordenadas de um lugar
    - Quiser saber a localização de um endereço
    - Precisar de latitude/longitude de algum local
    - Perguntar "onde fica..." seguido de um endereço

    Exemplos de uso:
    - "Quais as coordenadas da Av. Paulista?"
    - "Onde fica o Coliseu de Roma?"
    - "Me dê a latitude e longitude do Cristo Redentor"

    Args:
        address: Endereço ou nome do local
        country_codes: Filtro de países (opcional, ex: "br" para Brasil)

    Returns:
        Informações sobre o local com coordenadas
    """
    results = geocode_address(address, limit=3, country_codes=country_codes)

    if not results:
        return f"❌ Nenhum resultado encontrado para: '{address}'"

    if "error" in results[0]:
        return results[0]["error"]

    # Formata os resultados
    output_parts = [f"🔍 **Resultados para:** '{address}'\n"]

    for i, result in enumerate(results, 1):
        output_parts.append(f"\n**Resultado {i}:**")
        output_parts.append(format_geocode_result(result))

    return "\n".join(output_parts)


@tool("reverse_geocode", args_schema=ReverseGeocodeInput)
def reverse_geocode_tool(latitude: float, longitude: float) -> str:
    """
    Converte coordenadas geográficas (latitude e longitude) em um endereço.

    Use esta ferramenta quando o usuário:
    - Fornecer coordenadas e perguntar qual é o lugar
    - Quiser saber o endereço de uma latitude/longitude
    - Perguntar "que lugar é esse" com coordenadas
    - Precisar identificar um local por suas coordenadas

    Exemplos de uso:
    - "Que lugar fica em -23.5505, -46.6333?"
    - "Qual é o endereço das coordenadas 48.8584, 2.2945?"
    - "Identifique o local: lat -22.9068, lon -43.1729"

    Args:
        latitude: Latitude em graus decimais (-90 a 90)
        longitude: Longitude em graus decimais (-180 a 180)

    Returns:
        Endereço completo do local
    """
    # Valida as coordenadas
    if not -90 <= latitude <= 90:
        return f"❌ Latitude inválida: {latitude}. Deve estar entre -90 e 90."

    if not -180 <= longitude <= 180:
        return f"❌ Longitude inválida: {longitude}. Deve estar entre -180 e 180."

    result = reverse_geocode(latitude, longitude)

    if "error" in result:
        return f"❌ {result['error']}"

    if not result or "display_name" not in result:
        return f"❌ Nenhum endereço encontrado para as coordenadas: {latitude}, {longitude}"

    # Formata o resultado
    output_parts = [f"🔍 **Reverse Geocoding:** {latitude}, {longitude}\n"]
    output_parts.append(format_geocode_result(result))

    # Adiciona link para o mapa
    osm_link = f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}&zoom=17"
    output_parts.append(f"\n🗺️ **Ver no mapa:** {osm_link}")

    return "\n".join(output_parts)


# =============================================================================
# EXEMPLO DE USO STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌍 TESTE DA TOOL DE GEOCODING")
    print("=" * 60)

    # Teste 1: Geocoding
    print("\n📍 Teste 1: Geocoding (endereço → coordenadas)")
    print("-" * 40)
    result = geocode_address_tool.invoke({"address": "Av. Paulista, São Paulo, Brasil"})
    print(result)

    # Teste 2: Reverse Geocoding
    print("\n📍 Teste 2: Reverse Geocoding (coordenadas → endereço)")
    print("-" * 40)
    result = reverse_geocode_tool.invoke({"latitude": -23.5505, "longitude": -46.6333})
    print(result)

    # Teste 3: Lugar famoso
    print("\n📍 Teste 3: Lugar famoso")
    print("-" * 40)
    result = geocode_address_tool.invoke({"address": "Torre Eiffel, Paris"})
    print(result)

