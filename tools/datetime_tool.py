"""
=============================================================================
DATETIME TOOL - Ferramenta de Data e Hora
=============================================================================

Esta ferramenta permite ao agente saber a data e hora atual.

Por que isso é importante?
- LLMs são treinados com dados até uma data específica
- Eles NÃO sabem a data/hora atual
- Esta tool permite responder perguntas como "Que dia é hoje?"

=============================================================================
"""

from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    """
    Schema de entrada para a ferramenta de data/hora.

    O timezone é opcional - se não fornecido, usa o horário local.
    """
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="O timezone desejado. Exemplos: 'America/Sao_Paulo', 'UTC', 'America/New_York'"
    )
    format: str = Field(
        default="%d/%m/%Y %H:%M:%S",
        description="Formato da data. Padrão: DD/MM/YYYY HH:MM:SS"
    )


@tool("get_current_datetime", args_schema=DateTimeInput)
def get_current_datetime(timezone: str = "America/Sao_Paulo", format: str = "%d/%m/%Y %H:%M:%S") -> str:
    """
    Retorna a data e hora atual.

    Use esta ferramenta quando o usuário perguntar sobre:
    - Que dia é hoje
    - Que horas são
    - Data atual
    - Hora atual

    Args:
        timezone: O fuso horário desejado (padrão: São Paulo)
        format: O formato de saída da data

    Returns:
        A data e hora formatada como string
    """
    try:
        # Tenta usar pytz se disponível para timezones
        try:
            import pytz
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        except ImportError:
            # Se pytz não estiver instalado, usa horário local
            now = datetime.now()

        formatted_date = now.strftime(format)

        # Retorna informações detalhadas
        weekdays_pt = {
            0: "Segunda-feira",
            1: "Terça-feira",
            2: "Quarta-feira",
            3: "Quinta-feira",
            4: "Sexta-feira",
            5: "Sábado",
            6: "Domingo"
        }

        weekday = weekdays_pt[now.weekday()]

        return f"Agora são {formatted_date} ({weekday}), timezone: {timezone}"

    except Exception as e:
        return f"Erro ao obter data/hora: {str(e)}"


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    print(get_current_datetime.invoke({}))
    print(get_current_datetime.invoke({"timezone": "UTC"}))

