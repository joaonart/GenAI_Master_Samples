"""
=============================================================================
FINANCE AGENT - Agente Especialista em Finanças
=============================================================================

Este módulo implementa um agente especializado em consultas financeiras,
demonstrando como criar agentes com ferramentas específicas para um domínio.

CONCEITOS DIDÁTICOS:
1. Agente Especialista: Foca em um domínio específico (finanças)
2. Tools Selecionadas: Usa apenas ferramentas relevantes ao domínio
3. System Prompt Especializado: Instruções específicas para o contexto
4. Persona: O agente tem uma "personalidade" de analista financeiro

Ferramentas utilizadas:
- crypto_price: Consulta preços de criptomoedas
- top_cryptos: Lista top criptomoedas por market cap
- stock_quote: Cotação de ações (BR e US)
- forex_rate: Taxa de câmbio entre moedas
- calculator: Cálculos financeiros
- get_current_datetime: Data/hora para contexto temporal

IMPORTANTE:
- Para stocks/forex: Requer ALPHA_VANTAGE_API_KEY
- Para crypto: Não requer API key (CoinGecko gratuito)

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import os
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory

# Importa as tools de finanças
from tools import (
    calculator_tool,
    get_current_datetime,
    crypto_price_tool,
    top_cryptos_tool,
    stock_quote_tool,
    forex_rate_tool
)


class FinanceAgent(BaseAgent):
    """
    Agente especializado em consultas financeiras.

    Este agente é um "analista financeiro virtual" que pode:
    - Consultar cotações de ações brasileiras e americanas
    - Verificar preços de criptomoedas
    - Consultar taxas de câmbio
    - Realizar cálculos financeiros
    - Fornecer análises e insights básicos

    CONCEITO: Agente Especialista
    -----------------------------
    Diferente de um agente genérico, um agente especialista:
    1. Tem conhecimento focado em um domínio
    2. Usa ferramentas específicas para esse domínio
    3. Tem um system prompt otimizado para o contexto
    4. Pode ter "personalidade" adequada (ex: formal, técnico)

    Example:
        >>> agent = FinanceAgent(provider="openai")
        >>> response = agent.process_message("Como está o Bitcoin hoje?")
        >>> print(response)

        >>> response = agent.process_message("Cotação da Petrobras")
        >>> print(response)
    """

    # Tools específicas para finanças
    FINANCE_TOOLS = [
        calculator_tool,      # Para cálculos financeiros
        get_current_datetime, # Para contexto temporal
        crypto_price_tool,    # Preços de criptomoedas
        top_cryptos_tool,     # Ranking de cryptos
        stock_quote_tool,     # Cotações de ações
        forex_rate_tool,      # Taxas de câmbio
    ]

    # Modelos disponíveis por provider
    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4"],
        "google": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-flash-preview"]
    }

    def __init__(
        self,
        name: str = "Analista Financeiro",
        description: str = "Especialista em mercado financeiro, ações, criptomoedas e câmbio",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,  # Mais baixa para respostas precisas
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        # Parâmetros específicos OpenAI
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        # Parâmetros específicos Google
        top_k: Optional[int] = None,
        # API Keys
        api_key: Optional[str] = None,
        # System Prompt customizado
        system_prompt: Optional[str] = None,
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "finance_agent"
    ):
        """
        Inicializa o Agente Financeiro.

        Args:
            name: Nome do agente (ex: "Analista Financeiro")
            description: Descrição do agente
            provider: Provider do LLM ("openai" ou "google")
            model: Modelo a usar (se None, usa o padrão do provider)
            temperature: Criatividade (0.0-2.0). Recomendado: 0.3 para finanças
            max_tokens: Limite de tokens na resposta
            top_p: Nucleus sampling
            presence_penalty: Penaliza repetição de tópicos (OpenAI)
            frequency_penalty: Penaliza repetição de palavras (OpenAI)
            top_k: Top-K sampling (Google)
            api_key: API key do provider
            system_prompt: Prompt de sistema customizado
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            memory_max_messages: Máximo de mensagens no curto prazo
            memory_storage_path: Caminho para memória de longo prazo
            memory_session_id: ID da sessão de memória

        Note:
            Para usar stock_quote e forex_rate, configure ALPHA_VANTAGE_API_KEY.
            crypto_price e top_cryptos funcionam sem API key.
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

        # Modelo padrão
        if self.model is None:
            self.model = "gpt-4o-mini" if self.provider == "openai" else "gemini-2.0-flash"

        # Obter API Key
        self.api_key = self._get_api_key(api_key)

        # Criar o LLM
        self.llm = self._create_llm(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_k=top_k
        )

        # Configurar Tools
        self.tools = list(self.FINANCE_TOOLS)

        # System Prompt especializado
        self.system_prompt = system_prompt or self._get_finance_system_prompt()

        # Configurar Memória
        self.memory_type = memory_type
        self.memory = self._setup_memory(
            memory_type=memory_type,
            max_messages=memory_max_messages,
            storage_path=memory_storage_path,
            session_id=memory_session_id
        )

        # Criar o agente ReAct
        self._create_agent()

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

    def _get_finance_system_prompt(self) -> str:
        """
        Retorna o system prompt especializado para finanças.

        CONCEITO: System Prompt Especializado
        -------------------------------------
        O system prompt define a "personalidade" e comportamento do agente.
        Para um agente especialista, ele deve:
        1. Definir o papel/persona claramente
        2. Listar as capacidades específicas
        3. Dar instruções sobre como usar as ferramentas
        4. Definir o tom e formato das respostas
        """
        return f"""Você é o {self.name}, um {self.description}.

        ## 🎯 SEU PAPEL
        Você é um analista financeiro virtual especializado em fornecer informações
        sobre mercados financeiros, incluindo ações, criptomoedas e câmbio.
        
        ## 🛠️ SUAS FERRAMENTAS
        Você tem acesso às seguintes ferramentas:
        
        1. **stock_quote**: Consulta cotações de ações
           - Ações americanas: Apple (AAPL), Google (GOOGL), Tesla (TSLA), etc.
           - Ações brasileiras: Petrobras (PETR4), Vale (VALE3), Itaú (ITUB4), etc.
           - ⚠️ Requer ALPHA_VANTAGE_API_KEY configurada
        
        2. **forex_rate**: Taxa de câmbio entre moedas
           - Exemplos: USD/BRL (dólar/real), EUR/USD (euro/dólar)
           - ⚠️ Requer ALPHA_VANTAGE_API_KEY configurada
        
        3. **crypto_price**: Preço de criptomoedas
           - Bitcoin (BTC), Ethereum (ETH), Solana (SOL), etc.
           - ✅ Não requer API key
        
        4. **top_cryptos**: Ranking das maiores criptomoedas
           - Lista por market cap
           - ✅ Não requer API key
        
        5. **calculator**: Cálculos financeiros
           - Porcentagens, variações, conversões
        
        6. **get_current_datetime**: Data e hora atual
           - Para contextualizar informações
        
        ## 📋 INSTRUÇÕES
        
        1. **Sempre use as ferramentas** para obter dados atualizados
        2. **Seja preciso** com números e valores
        3. **Forneça contexto** (variação, comparações, tendências)
        4. **Use formatação clara** com emojis e markdown
        5. **Seja objetivo** mas informativo
        6. **Avise sobre limitações** (dados com delay, necessidade de API key)
        
        ## ⚠️ AVISOS IMPORTANTES
        
        - Os dados são para fins informativos apenas
        - Não constitui recomendação de investimento
        - Preços podem ter delay de alguns minutos
        - Para ações e forex, é necessário ter ALPHA_VANTAGE_API_KEY configurada
        
        ## 🗣️ TOM DE COMUNICAÇÃO
        
        - Profissional mas acessível
        - Técnico quando necessário
        - Sempre em português brasileiro
        - Use emojis para melhorar a legibilidade
        """

    def _setup_memory(
        self,
        memory_type: str,
        max_messages: int,
        storage_path: str,
        session_id: str
    ):
        """Configura o sistema de memória."""
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

    def _create_agent(self):
        """Cria o agente ReAct com LangGraph."""
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )

    def _extract_text_from_content(self, content) -> str:
        """Extrai texto do conteúdo da resposta."""
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

        O fluxo de processamento:
        1. Prepara o contexto (system prompt + memória)
        2. Adiciona a mensagem do usuário
        3. Invoca o agente ReAct (que pode usar tools)
        4. Extrai e retorna a resposta
        5. Atualiza a memória

        Args:
            message: Pergunta ou solicitação do usuário

        Returns:
            Resposta do agente com informações financeiras
        """
        try:
            # Prepara mensagens com system prompt
            messages = [SystemMessage(content=self.system_prompt)]

            # Adiciona contexto da memória de longo prazo
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

            # Invoca o agente ReAct
            result = self.agent.invoke({"messages": messages})

            # Extrai a resposta
            response_messages = result.get("messages", [])

            if response_messages:
                last_message = response_messages[-1]
                response = self._extract_text_from_content(last_message.content)
            else:
                response = "❌ Erro ao processar sua solicitação."

            # Atualiza a memória
            self._update_memory(message, response)

            return response

        except Exception as e:
            error_msg = str(e)
            if "ALPHA_VANTAGE_API_KEY" in error_msg:
                return (
                    "❌ Para consultar ações e câmbio, configure a API key:\n\n"
                    "1. Cadastre-se em: https://www.alphavantage.co/support/#api-key\n"
                    "2. Configure: `export ALPHA_VANTAGE_API_KEY='sua_key'`\n\n"
                    "💡 Consultas de criptomoedas funcionam sem API key!"
                )
            return f"❌ Erro: {error_msg}"

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
        self.add_to_history(user_message, ai_response)

        if self.memory is None:
            return

        if self.memory_type == "short_term":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)
        elif self.memory_type == "combined":
            self.memory.add_user_message(user_message)
            self.memory.add_ai_message(ai_response)

    def save_to_long_term(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 5
    ) -> None:
        """
        Salva uma informação na memória de longo prazo.

        Útil para salvar preferências do usuário, como:
        - Ações favoritas
        - Moedas de interesse
        - Perfil de investidor

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

    def list_tools(self) -> List[str]:
        """Retorna a lista de tools disponíveis."""
        return [tool.name for tool in self.tools]

    def has_rag(self) -> bool:
        """Retorna False (este agente não usa RAG)."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "description": self.description,
            "specialization": "finance"
        }

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """Retorna os modelos disponíveis para um provider."""
        return cls.AVAILABLE_MODELS.get(provider.lower(), [])

    def check_api_keys(self) -> Dict[str, bool]:
        """
        Verifica quais API keys estão configuradas.

        Returns:
            Dicionário indicando status de cada API key
        """
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
            "alpha_vantage": bool(os.getenv("ALPHA_VANTAGE_API_KEY")),
            "coingecko": True  # Não requer API key
        }
