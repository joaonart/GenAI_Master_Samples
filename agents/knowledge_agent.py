"""
=============================================================================
KNOWLEDGE AGENT - Agente Especialista em Conhecimento
=============================================================================

Este módulo implementa um agente especializado em consultas de conhecimento
geral, usando a Wikipedia como principal fonte de informação.

CONCEITOS DIDÁTICOS:
1. Agente de FAQ/Knowledge: Especializado em responder perguntas
2. Wikipedia como fonte: Informações enciclopédicas confiáveis
3. Multi-idioma: Suporte a consultas em diferentes idiomas
4. Pesquisa inteligente: Busca e resumo de artigos

Ferramentas utilizadas:
- wikipedia_summary: Obtém resumo de um artigo
- wikipedia_search: Pesquisa artigos por termo
- web_search: Busca complementar na web
- calculator: Para cálculos simples
- get_current_datetime: Contexto temporal

Ideal para:
- Assistentes de FAQ
- Bots educacionais
- Pesquisa de informações
- Tutores virtuais

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

# Importa as tools de conhecimento
from tools import (
    calculator_tool,
    get_current_datetime,
    web_search_tool,
    wikipedia_summary_tool,
    wikipedia_search_tool
)


class KnowledgeAgent(BaseAgent):
    """
    Agente especializado em consultas de conhecimento geral.

    Este agente é um "assistente de conhecimento" que pode:
    - Responder perguntas sobre qualquer assunto usando a Wikipedia
    - Pesquisar informações enciclopédicas
    - Explicar conceitos, biografias, eventos históricos
    - Buscar informações complementares na web
    - Realizar cálculos quando necessário

    CONCEITO: Knowledge Assistant
    -----------------------------
    Diferente de um chatbot genérico, um knowledge assistant:
    1. Prioriza informações de fontes confiáveis (Wikipedia)
    2. Sempre cita as fontes das informações
    3. Admite quando não sabe algo
    4. Oferece links para aprofundamento

    Example:
        >>> agent = KnowledgeAgent(provider="openai")
        >>> response = agent.process_message("Quem foi Albert Einstein?")
        >>> print(response)

        >>> response = agent.process_message("O que é fotossíntese?")
        >>> print(response)
    """

    # Tools específicas para conhecimento
    KNOWLEDGE_TOOLS = [
        wikipedia_summary_tool,  # Resumos da Wikipedia
        wikipedia_search_tool,   # Pesquisa na Wikipedia
        web_search_tool,         # Busca complementar
        calculator_tool,         # Cálculos
        get_current_datetime,    # Contexto temporal
    ]

    # Modelos disponíveis por provider
    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4"],
        "google": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"]
    }

    def __init__(
        self,
        name: str = "Assistente de Conhecimento",
        description: str = "Especialista em informações enciclopédicas e conhecimento geral",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,  # Baixa para respostas precisas
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
        # Idioma padrão para Wikipedia
        default_language: str = "pt",
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "knowledge_agent"
    ):
        """
        Inicializa o Agente de Conhecimento.

        Args:
            name: Nome do agente
            description: Descrição do agente
            provider: Provider do LLM ("openai" ou "google")
            model: Modelo a usar
            temperature: Criatividade (recomendado: 0.3 para precisão)
            max_tokens: Limite de tokens na resposta
            top_p: Nucleus sampling
            presence_penalty: Penaliza repetição de tópicos (OpenAI)
            frequency_penalty: Penaliza repetição de palavras (OpenAI)
            top_k: Top-K sampling (Google)
            api_key: API key do provider
            system_prompt: Prompt de sistema customizado
            default_language: Idioma padrão para consultas Wikipedia (pt, en, es, etc.)
            memory_type: Tipo de memória
            memory_max_messages: Máximo de mensagens no curto prazo
            memory_storage_path: Caminho para memória de longo prazo
            memory_session_id: ID da sessão de memória
        """
        super().__init__(name, description)

        self.provider = provider.lower()
        self.model = model
        self.default_language = default_language

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
        self.tools = list(self.KNOWLEDGE_TOOLS)

        # System Prompt especializado
        self.system_prompt = system_prompt or self._get_knowledge_system_prompt()

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

    def _get_knowledge_system_prompt(self) -> str:
        """
        Retorna o system prompt especializado para conhecimento.

        CONCEITO: Prompt para Knowledge Assistant
        -----------------------------------------
        O prompt define o agente como uma fonte confiável de informação,
        enfatizando:
        - Uso da Wikipedia como fonte primária
        - Citação de fontes
        - Honestidade sobre limitações
        """
        return f"""Você é o {self.name}, um {self.description}.

## 🎯 SEU PAPEL
Você é um assistente de conhecimento especializado em responder perguntas
sobre qualquer assunto, usando a Wikipedia como principal fonte de informação.

## 🛠️ SUAS FERRAMENTAS

1. **wikipedia_summary**: Obtém resumo de um artigo da Wikipedia
   - Use para responder "O que é X?" ou "Quem foi Y?"
   - Fornece informações enciclopédicas confiáveis
   - Suporta múltiplos idiomas (pt, en, es, fr, etc.)

2. **wikipedia_search**: Pesquisa artigos na Wikipedia
   - Use quando não souber o termo exato
   - Retorna lista de artigos relacionados
   - Útil para explorar um tema

3. **web_search**: Busca na web (DuckDuckGo)
   - Use para informações não encontradas na Wikipedia
   - Útil para notícias recentes ou tópicos específicos

4. **calculator**: Calculadora
   - Para cálculos matemáticos simples

5. **get_current_datetime**: Data e hora atual
   - Para contextualizar informações temporais

## 📋 INSTRUÇÕES

1. **Sempre consulte a Wikipedia primeiro** para perguntas de conhecimento
2. **Cite as fontes** - mencione que a informação vem da Wikipedia
3. **Se não encontrar**, use wikipedia_search para buscar termos relacionados
4. **Seja preciso** - não invente informações
5. **Admita limitações** - se não souber, diga claramente
6. **Ofereça aprofundamento** - sugira o link do artigo completo

## 🗣️ FORMATO DE RESPOSTA

Ao responder perguntas de conhecimento:

1. **Comece com um resumo direto** da resposta
2. **Adicione detalhes relevantes** do artigo
3. **Mencione a fonte** (Wikipedia)
4. **Sugira tópicos relacionados** se apropriado

## 💡 EXEMPLOS DE PERGUNTAS QUE VOCÊ RESPONDE BEM

- "O que é inteligência artificial?"
- "Quem foi Marie Curie?"
- "Explique a fotossíntese"
- "Qual a história do Brasil?"
- "O que é a teoria da relatividade?"
- "Me fale sobre a Torre Eiffel"

## ⚠️ AVISOS IMPORTANTES

- As informações da Wikipedia são geralmente confiáveis, mas podem conter erros
- Para temas controversos, mencione diferentes perspectivas
- Sempre incentive o usuário a verificar fontes adicionais para decisões importantes

## 🌐 IDIOMA

- Idioma padrão para consultas: **{self.default_language}**
- Responda sempre em português brasileiro
- Você pode consultar Wikipedias em outros idiomas se necessário
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

        O agente irá:
        1. Analisar a pergunta
        2. Decidir se precisa consultar a Wikipedia
        3. Buscar informações relevantes
        4. Formular uma resposta informativa

        Args:
            message: Pergunta ou solicitação do usuário

        Returns:
            Resposta com informações da Wikipedia
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
                response = "❌ Erro ao processar sua pergunta."

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
        """Salva uma informação na memória de longo prazo."""
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
        """Retorna False (este agente não usa RAG local)."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "description": self.description,
            "specialization": "knowledge",
            "default_language": self.default_language
        }

    @classmethod
    def get_available_models(cls, provider: str) -> List[str]:
        """Retorna os modelos disponíveis para um provider."""
        return cls.AVAILABLE_MODELS.get(provider.lower(), [])


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📚 TESTE DO KNOWLEDGE AGENT")
    print("=" * 60)

    # Verifica API key
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY não configurada")
    else:
        print("✅ OPENAI_API_KEY configurada")

        print("\n" + "-" * 60)
        print("Criando Knowledge Agent...")

        agent = KnowledgeAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.3,
            default_language="pt"
        )

        print(f"✅ Agente criado: {agent.name}")
        print(f"📋 Tools disponíveis: {agent.list_tools()}")

        # Teste
        print("\n" + "-" * 60)
        print("📝 Teste: Perguntando sobre Albert Einstein...")
        response = agent.process_message("Quem foi Albert Einstein?")
        print(response)

