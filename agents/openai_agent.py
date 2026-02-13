"""
=============================================================================
OPENAI AGENT WITH TOOLS - Agente OpenAI com Ferramentas
=============================================================================

Este agente demonstra como usar TOOLS (ferramentas) com LangGraph.

O que são Tools?
- Funções que o agente pode "decidir" chamar
- Permitem que o LLM execute ações no mundo real
- Exemplo: calculadora, busca web, consulta a APIs

Como funciona o ciclo de decisão (ReAct Pattern):
1. Usuário envia mensagem
2. LLM analisa se precisa usar alguma tool
3. Se sim: executa a tool e usa o resultado
4. Se não: responde diretamente
5. Retorna resposta ao usuário

RAG (Retrieval Augmented Generation):
- Permite que o agente consulte uma base de conhecimento
- Documentos são convertidos em vetores (embeddings)
- O agente pode buscar informações relevantes

MEMÓRIA:
- Curto Prazo: Mantém as últimas N mensagens
- Longo Prazo: Persiste informações importantes entre sessões
- Combinada: Usa ambas as estratégias

NOTA: Usamos langgraph.prebuilt.create_react_agent (forma moderna)

=============================================================================
"""

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory

# Importa as tools disponíveis
from tools import calculator_tool, get_current_datetime, web_search_tool
from tools import rag_search_tool, set_vector_store


class OpenAIAgent(BaseAgent):
    """
    Agente OpenAI com suporte a Tools, RAG e Memória usando LangGraph.

    Este agente pode:
    - Conversar naturalmente
    - Usar calculadora para cálculos
    - Consultar data/hora atual
    - Fazer buscas (simuladas)
    - Consultar base de conhecimento (RAG)
    - Lembrar conversas anteriores (Memória)
    """

    # Tools padrão disponíveis
    DEFAULT_TOOLS = [
        calculator_tool,
        get_current_datetime,
        web_search_tool,
    ]

    def __init__(
        self,
        name: str = "OpenAI Assistant",
        description: str = "Assistente IA com ferramentas",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        vector_store_manager=None,
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "default"
    ):
        """
        Inicializa o agente OpenAI com parâmetros configuráveis.

        Args:
            name: Nome do agente
            description: Descrição do agente
            model: Modelo a usar (gpt-4, gpt-4o, etc.)
            temperature: Controla aleatoriedade (0.0 = determinístico, 2.0 = muito criativo)
            max_tokens: Número máximo de tokens na resposta (None = sem limite)
            top_p: Nucleus sampling - considera tokens com probabilidade acumulada até top_p
            presence_penalty: Penaliza tokens já mencionados (-2.0 a 2.0)
            frequency_penalty: Penaliza tokens frequentes (-2.0 a 2.0)
            system_prompt: Prompt de sistema customizado
            api_key: API key da OpenAI (ou usa OPENAI_API_KEY)
            tools: Lista de tools customizadas
            vector_store_manager: Gerenciador do vector store para RAG
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            memory_max_messages: Número máximo de mensagens no curto prazo
            memory_storage_path: Caminho para salvar memória de longo prazo
            memory_session_id: ID da sessão para memória de longo prazo
        """
        super().__init__(name, description)

        # Validar API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ API Key da OpenAI não encontrada!\n"
                "Configure OPENAI_API_KEY ou passe api_key."
            )

        # Inicializar o LLM com todos os parâmetros
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            api_key=self.api_key
        )

        # Configurar Tools
        self.tools = list(tools) if tools is not None else list(self.DEFAULT_TOOLS)

        # Configurar RAG se fornecido
        self.vector_store_manager = vector_store_manager
        self._setup_rag()

        # Configurar Memória
        self.memory_type = memory_type
        self.memory = self._setup_memory(
            memory_type=memory_type,
            max_messages=memory_max_messages,
            storage_path=memory_storage_path,
            session_id=memory_session_id
        )

        # System Prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Criar agente ReAct com LangGraph
        self._create_agent()

    def _setup_memory(
        self,
        memory_type: str,
        max_messages: int,
        storage_path: str,
        session_id: str
    ):
        """
        Configura o sistema de memória baseado no tipo selecionado.

        Args:
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            max_messages: Número máximo de mensagens no curto prazo
            storage_path: Caminho para salvar memória de longo prazo
            session_id: ID da sessão

        Returns:
            Instância da memória ou None
        """
        if memory_type == "none":
            return None

        elif memory_type == "short_term":
            return ShortTermMemory(max_messages=max_messages)

        elif memory_type == "long_term":
            return LongTermMemory(
                storage_path=storage_path,
                session_id=session_id
            )

        elif memory_type == "combined":
            return CombinedMemory(
                max_short_term_messages=max_messages,
                storage_path=storage_path,
                session_id=session_id
            )

        return None

    def _setup_rag(self):
        """Configura o RAG se o vector store estiver disponível."""
        if self.vector_store_manager is not None:
            # Configura o vector store global para a tool
            set_vector_store(self.vector_store_manager)
            # Adiciona a tool de RAG se ainda não estiver na lista
            if rag_search_tool not in self.tools:
                self.tools.append(rag_search_tool)

    def _get_default_system_prompt(self):
        """Retorna o system prompt padrão."""
        base_prompt = f"""
Você é {self.name}, {self.description}.

Você tem acesso às seguintes ferramentas:
- calculator: Para fazer cálculos matemáticos
- get_current_datetime: Para saber a data e hora atual
- web_search: Para buscar informações na web
"""
        # Adiciona instruções de RAG se habilitado
        if self.vector_store_manager is not None:
            base_prompt += """- knowledge_base_search: Para buscar informações na base de conhecimento

IMPORTANTE SOBRE A BASE DE CONHECIMENTO:
- Use knowledge_base_search quando o usuário perguntar sobre documentos específicos
- A base de conhecimento contém documentos que foram carregados pelo usuário
- Sempre cite a fonte quando usar informações da base de conhecimento
"""

        base_prompt += """
INSTRUÇÕES:
1. Use as ferramentas quando necessário
2. Para data/hora, use get_current_datetime
3. Para cálculos, use calculator
4. Responda em português brasileiro
"""
        return base_prompt

    def _create_agent(self):
        """Cria o agente ReAct."""
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )

    def set_vector_store(self, manager):
        """
        Define o vector store manager para RAG.

        Args:
            manager: Instância de VectorStoreManager
        """
        self.vector_store_manager = manager
        self._setup_rag()
        # Atualiza o system prompt se estava usando o padrão
        if "knowledge_base_search" not in self.system_prompt:
            self.system_prompt = self._get_default_system_prompt()
        # Recria o agente com a nova tool
        self._create_agent()

    def _extract_text_from_content(self, content) -> str:
        """
        Extrai texto do conteúdo da resposta.

        Alguns modelos podem retornar o conteúdo em diferentes formatos:
        - String simples: "texto da resposta"
        - Lista de blocos: [{"type": "text", "text": "..."}, ...]

        Esta função normaliza para string.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Extrai texto de cada bloco do tipo "text"
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)

        # Fallback: converte para string
        return str(content)

    def process_message(self, message: str) -> str:
        try:
            # Prepara mensagens com system prompt
            messages = [SystemMessage(content=self.system_prompt)]

            # Adiciona contexto da memória de longo prazo se disponível
            if self.memory_type in ["long_term", "combined"]:
                long_term_context = self._get_long_term_context()
                if long_term_context:
                    messages.append(SystemMessage(content=long_term_context))

            # Adiciona histórico de curto prazo
            if self.memory_type == "combined" and self.memory:
                # Usa as mensagens do curto prazo da memória combinada
                messages.extend(self.memory.get_short_term_messages())
            elif self.memory_type == "short_term" and self.memory:
                # Usa as mensagens da memória de curto prazo
                messages.extend(self.memory.messages)
            else:
                # Usa o histórico padrão do agente base
                messages.extend(self.chat_history)

            messages.append(HumanMessage(content=message))

            # Invoca o agente
            result = self.agent.invoke({"messages": messages})

            # Extrai resposta
            response_messages = result.get("messages", [])

            if response_messages:
                # Extrai o conteúdo da última mensagem
                last_message = response_messages[-1]
                raw_content = last_message.content

                # Normaliza o conteúdo para string
                response = self._extract_text_from_content(raw_content)
            else:
                response = "Erro ao processar."

            # Atualiza a memória
            self._update_memory(message, response)

            return response

        except Exception as e:
            return f"❌ Erro: {str(e)}"

    def _get_long_term_context(self) -> str:
        """Retorna o contexto da memória de longo prazo."""
        if self.memory is None:
            return ""

        if self.memory_type == "long_term":
            return self.memory.get_memories_as_text(limit=5)
        elif self.memory_type == "combined":
            return self.memory.long_term.get_memories_as_text(limit=5)

        return ""

    def _update_memory(self, user_message: str, ai_response: str) -> None:
        """Atualiza a memória com a nova interação."""
        # Sempre atualiza o histórico do agente base
        self.add_to_history(user_message, ai_response)

        if self.memory is None:
            return

        if self.memory_type == "short_term":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)

        elif self.memory_type == "long_term":
            # Para longo prazo, não adicionamos cada mensagem
            # O usuário pode salvar fatos importantes manualmente
            pass

        elif self.memory_type == "combined":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)

    def save_to_long_term(self, content: str, memory_type: str = "fact", importance: int = 5) -> None:
        """
        Salva uma informação na memória de longo prazo.

        Args:
            content: Conteúdo a salvar
            memory_type: Tipo (fact, preference, summary)
            importance: Importância de 1 a 10
        """
        if self.memory is None:
            return

        if self.memory_type == "long_term":
            self.memory.add_memory(content, memory_type, importance)
        elif self.memory_type == "combined":
            self.memory.add_to_long_term(content, memory_type, importance)

    def get_memory_info(self) -> dict:
        """Retorna informações sobre a memória atual."""
        info = {
            "type": self.memory_type,
            "enabled": self.memory is not None
        }

        if self.memory_type == "short_term" and self.memory:
            info["messages_count"] = len(self.memory.messages)
            info["max_messages"] = self.memory.max_messages

        elif self.memory_type == "long_term" and self.memory:
            info["memories_count"] = len(self.memory.memories)

        elif self.memory_type == "combined" and self.memory:
            info["short_term_messages"] = len(self.memory.short_term.messages)
            info["long_term_memories"] = len(self.memory.long_term.memories)

        return info

    def add_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)
        self._create_agent()

    def list_tools(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def has_rag(self) -> bool:
        """Retorna True se o RAG está habilitado."""
        return self.vector_store_manager is not None
