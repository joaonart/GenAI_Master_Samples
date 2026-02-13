"""
=============================================================================
CALCULATOR TOOL - Ferramenta de Calculadora
=============================================================================

Esta é uma ferramenta simples que permite ao agente fazer cálculos matemáticos.

Conceitos importantes:
- @tool decorator: Transforma uma função Python em uma "tool" do LangChain
- Pydantic BaseModel: Define o schema dos parâmetros da tool
- Docstring: A descrição é usada pelo LLM para decidir quando usar a tool

Como funciona:
1. O usuário faz uma pergunta que envolve cálculo
2. O LLM identifica que precisa da calculadora
3. O LLM extrai os parâmetros (expressão matemática)
4. A tool executa o cálculo
5. O resultado é retornado ao LLM
6. O LLM formula a resposta final

=============================================================================
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """
    Schema de entrada para a calculadora.

    O Pydantic valida automaticamente os dados de entrada.
    O Field() permite adicionar descrições que ajudam o LLM.
    """
    expression: str = Field(
        description="A expressão matemática a ser calculada. Exemplo: '2 + 2', '10 * 5', 'sqrt(16)'"
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    Calcula expressões matemáticas.

    Use esta ferramenta quando precisar fazer cálculos matemáticos.
    Suporta operações básicas (+, -, *, /) e funções como sqrt, pow, etc.

    Exemplos de expressões válidas:
    - "2 + 2" -> 4
    - "10 * 5" -> 50
    - "100 / 4" -> 25
    - "2 ** 8" -> 256 (potenciação)

    Args:
        expression: A expressão matemática como string

    Returns:
        O resultado do cálculo como string
    """
    import math

    try:
        # ATENÇÃO: eval() pode ser perigoso em produção!
        # Aqui usamos apenas para fins didáticos.
        # Em produção, use uma biblioteca segura como 'numexpr' ou 'asteval'

        # Criamos um ambiente seguro com apenas funções matemáticas
        safe_dict = {
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        # Calcula a expressão
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return f"O resultado de {expression} é: {result}"

    except Exception as e:
        return f"Erro ao calcular '{expression}': {str(e)}"


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Teste a tool diretamente
    print(calculator_tool.invoke({"expression": "2 + 2"}))
    print(calculator_tool.invoke({"expression": "sqrt(16)"}))
    print(calculator_tool.invoke({"expression": "2 ** 10"}))

