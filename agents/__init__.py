"""
=============================================================================
AGENTS - Módulo de Agentes de IA
=============================================================================

Este módulo contém os agentes de IA disponíveis no projeto.

O que é um Agente?
- Um programa que usa um LLM para "pensar" e tomar decisões
- Pode usar ferramentas (tools) para realizar ações
- Mantém contexto através do histórico de conversas

Agentes disponíveis:
- BaseAgent: Classe base abstrata (não use diretamente)
- SimpleAgent: Agente simples de chat (sem tools/RAG)
- OpenAIAgent: Agente com GPT-4 e tools
- GeminiAgent: Agente com Google Gemini e tools
- OllamaAgent: Agente com modelos locais via Ollama (sem API key!)
- FinanceAgent: Agente especialista em finanças
- KnowledgeAgent: Agente especialista em conhecimento (Wikipedia)
- WebSearchAgent: Agente especialista em pesquisa web
- MCPAgent: Agente que conecta a servidores MCP externos
- MCPAgentDemo: Demonstração do conceito MCP (sem dependências)

Tipos de Memória:
- ShortTermMemory: Memória de curto prazo (últimas N mensagens)
- LongTermMemory: Memória de longo prazo (persiste em disco)
- CombinedMemory: Combina curto e longo prazo

Exemplo de uso:
    from agents import OpenAIAgent, GeminiAgent, OllamaAgent
    from agents import FinanceAgent, KnowledgeAgent, WebSearchAgent

    # Agente OpenAI (requer OPENAI_API_KEY)
    agent = OpenAIAgent()
    response = agent.process_message("Olá, quanto é 2+2?")

    # Agente Ollama (LOCAL - sem API key!)
    ollama = OllamaAgent(model="llama3.2")
    response = ollama.process_message("Olá! Quanto é 15 * 7?")

    # Ver modelos disponíveis no Ollama
    print(OllamaAgent.list_popular_models())

    # Agente especialista em finanças
    finance = FinanceAgent(provider="openai")
    response = finance.process_message("Qual o preço do Bitcoin?")

    # Agente especialista em conhecimento
    knowledge = KnowledgeAgent(provider="openai")
    response = knowledge.process_message("Quem foi Albert Einstein?")

    # Agente especialista em pesquisa web
    search = WebSearchAgent(provider="openai")
    response = search.process_message("Pesquise sobre LangChain")

    # Agente MCP (Model Context Protocol)
    mcp = MCPAgentDemo(provider="openai", mcp_server_name="fetch")
    response = mcp.process_message("Explique o MCP")

=============================================================================
"""

from .base_agent import BaseAgent
from .openai_agent import OpenAIAgent
from .gemini_agent import GeminiAgent
from .ollama_agent import OllamaAgent
from .simple_agent import SimpleAgent
from .finance_agent import FinanceAgent
from .knowledge_agent import KnowledgeAgent
from .websearch_agent import WebSearchAgent
from .mcp_agent import MCPAgentDemo

# Tenta importar MCPAgent completo (requer dependências extras)
try:
    from .mcp_agent import MCPAgent
except ImportError:
    MCPAgent = None  # Não disponível sem dependências MCP

from core.memory import (
    ShortTermMemory,
    LongTermMemory,
    CombinedMemory,
    MEMORY_TYPES,
    get_memory_types
)

__all__ = [
    "BaseAgent",
    "OpenAIAgent",
    "GeminiAgent",
    "OllamaAgent",
    "SimpleAgent",
    "FinanceAgent",
    "KnowledgeAgent",
    "WebSearchAgent",
    "MCPAgent",
    "MCPAgentDemo",
    "ShortTermMemory",
    "LongTermMemory",
    "CombinedMemory",
    "MEMORY_TYPES",
    "get_memory_types"
]
