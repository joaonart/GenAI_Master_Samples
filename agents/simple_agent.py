"""
=============================================================================
SIMPLE AGENT - Agente Simples sem Tools e RAG
=============================================================================

Este módulo implementa um agente simples que usa apenas o LLM para conversar,
sem ferramentas (tools) ou base de conhecimento (RAG).

Ideal para:
- Aprender os conceitos básicos de agentes
- Casos de uso simples de chat
- Demonstrações e testes rápidos
- Quando não precisa de funcionalidades avançadas

Características:
- Conversa natural com o LLM
- Suporte a memória (curto/longo prazo)
- System prompt configurável
- Parâmetros do modelo ajustáveis (temperatura, top_p, etc.)

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import os
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory


class SimpleAgent(BaseAgent):
    """
    Agente simples de chat sem Tools e RAG.

    Este agente é ideal para conversas simples onde não é necessário
    executar ações externas ou consultar uma base de conhecimento.

    Providers suportados:
    - OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
    - Google (Gemini 2.5 Flash, Gemini 2.0 Flash, Gemini 1.5 Pro)

    Example:
        >>> agent = SimpleAgent(
        ...     provider="openai",
        ...     model="gpt-4o-mini",
        ...     temperature=0.7
        ... )
        >>> response = agent.process_message("Olá! Como você está?")
        >>> print(response)
    """

    # Modelos disponíveis por provider
    AVAILABLE_MODELS = {
        "openai": ["gpt-4", "gpt-4o", "gpt-4o-mini"],
        "google": ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-1.5-pro"]
    }

    def __init__(
        self,
        name: str = "Simple Assistant",
        description: str = "Assistente de IA simples para conversas",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        # Parâmetros específicos OpenAI
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        # Parâmetros específicos Google
        top_k: Optional[int] = None,
        # API Keys
        api_key: Optional[str] = None,
        # System Prompt
        system_prompt: Optional[str] = None,
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "default"
    ):
        """
        Inicializa o agente simples.

        Args:
            name: Nome do agente
            description: Descrição do agente
            provider: Provider do LLM ("openai" ou "google")
            model: Modelo a usar (se None, usa o padrão do provider)
            temperature: Controla aleatoriedade (0.0 = determinístico, 2.0 = muito criativo)
            max_tokens: Número máximo de tokens na resposta (None = sem limite)
            top_p: Nucleus sampling - considera tokens com probabilidade acumulada até top_p
            presence_penalty: Penaliza tokens já mencionados (apenas OpenAI)
            frequency_penalty: Penaliza tokens frequentes (apenas OpenAI)
            top_k: Considera apenas os top_k tokens mais prováveis (apenas Google)
            api_key: API key do provider (ou usa variável de ambiente)
            system_prompt: Prompt de sistema customizado
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            memory_max_messages: Número máximo de mensagens no curto prazo
            memory_storage_path: Caminho para salvar memória de longo prazo
            memory_session_id: ID da sessão para memória de longo prazo
        """
        super().__init__(name, description)

        self.provider = provider.lower()
        self.model = model

        # Validar provider
        if self.provider not in ["openai", "google"]:
            raise ValueError(
                f"❌ Provider '{provider}' não suportado!\n"
                "Use 'openai' ou 'google'."
            )

        # Configurar modelo padrão se não especificado
        if self.model is None:
            self.model = "gpt-4o-mini" if self.provider == "openai" else "gemini-2.0-flash"

        # Configurar API Key
        self.api_key = self._get_api_key(api_key)

        # Inicializar o LLM
        self.llm = self._create_llm(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_k=top_k
        )

        # System Prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Configurar Memória
        self.memory_type = memory_type
        self.memory = self._setup_memory(
            memory_type=memory_type,
            max_messages=memory_max_messages,
            storage_path=memory_storage_path,
            session_id=memory_session_id
        )

    def _get_api_key(self, api_key: Optional[str]) -> str:
        """Obtém a API key do provider."""
        if api_key:
            return api_key

        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "❌ API Key da OpenAI não encontrada!\n"
                    "Configure OPENAI_API_KEY ou passe api_key."
                )
            return key
        else:  # google
            key = os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "❌ API Key do Google não encontrada!\n"
                    "Configure GOOGLE_API_KEY ou passe api_key."
                )
            return key

    def _create_llm(
        self,
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        top_k: Optional[int]
    ):
        """Cria a instância do LLM baseado no provider."""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                api_key=self.api_key
            )
        else:  # google
            kwargs = {
                "model": self.model,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
                "google_api_key": self.api_key
            }
            if top_k is not None:
                kwargs["top_k"] = top_k

            return ChatGoogleGenerativeAI(**kwargs)

    def _get_default_system_prompt(self) -> str:
        """Retorna o system prompt padrão."""
        return f"""Você é {self.name}, {self.description}.

INSTRUÇÕES:
1. Seja sempre educado e prestativo
2. Responda em português brasileiro
3. Seja conciso, mas completo nas respostas
4. Se não souber algo, admita honestamente
5. Mantenha o contexto da conversa
"""

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

    def _extract_text_from_content(self, content) -> str:
        """
        Extrai texto do conteúdo da resposta.

        Alguns modelos podem retornar o conteúdo em diferentes formatos.
        Esta função normaliza para string.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
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

        return str(content)

    def process_message(self, message: str) -> str:
        """
        Processa uma mensagem do usuário e retorna uma resposta.

        Este método:
        1. Prepara o contexto com system prompt e memória
        2. Envia para o LLM
        3. Atualiza a memória
        4. Retorna a resposta

        Args:
            message: A mensagem enviada pelo usuário

        Returns:
            A resposta gerada pelo agente
        """
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
                messages.extend(self.memory.get_short_term_messages())
            elif self.memory_type == "short_term" and self.memory:
                messages.extend(self.memory.messages)
            else:
                messages.extend(self.chat_history)

            # Adiciona a mensagem atual
            messages.append(HumanMessage(content=message))

            # Invoca o LLM diretamente (sem tools/agent)
            result = self.llm.invoke(messages)

            # Extrai a resposta
            response = self._extract_text_from_content(result.content)

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

    def get_memory_info(self) -> Dict[str, Any]:
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

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo atual."""
        return {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "description": self.description
        }

    def list_tools(self) -> List[str]:
        """
        Retorna lista vazia de tools (agente simples não usa tools).

        Este método existe para compatibilidade com a interface Streamlit.
        """
        return []

    def has_rag(self) -> bool:
        """
        Retorna False (agente simples não usa RAG).

        Este método existe para compatibilidade com a interface.
        """
        return False

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """
        Retorna os modelos disponíveis para um provider.

        Args:
            provider: "openai" ou "google"

        Returns:
            Lista de modelos disponíveis
        """
        return cls.AVAILABLE_MODELS.get(provider.lower(), [])

