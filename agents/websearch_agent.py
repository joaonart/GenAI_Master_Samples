"""
=============================================================================
WEB SEARCH AGENT - Agente Especialista em Pesquisa Web
=============================================================================

Este módulo implementa um agente especializado em buscas na web,
usando a ferramenta de pesquisa DuckDuckGo.

CONCEITOS DIDÁTICOS:
1. Agente de Pesquisa: Especializado em encontrar informações atualizadas
2. Single Tool Agent: Demonstra um agente focado em uma única ferramenta
3. Busca Inteligente: Reformula queries para melhores resultados

Ferramenta utilizada:
- web_search: Busca na web via DuckDuckGo

Ideal para:
- Pesquisas de notícias recentes
- Buscar informações atualizadas
- Encontrar artigos e referências
- Pesquisar sobre eventos atuais

IMPORTANTE:
- Não requer API key (DuckDuckGo é gratuito)
- Resultados podem variar conforme disponibilidade

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

# Importa a tool de web search
from tools import web_search_tool


class WebSearchAgent(BaseAgent):
    """
    Agente especializado em pesquisas na web.

    Este agente é um "pesquisador virtual" que pode:
    - Buscar informações atualizadas na web
    - Encontrar notícias recentes
    - Pesquisar artigos e referências
    - Responder sobre eventos atuais

    CONCEITO: Single Tool Agent
    ---------------------------
    Diferente de agentes com múltiplas ferramentas, este agente
    demonstra como criar um especialista focado em uma única
    capacidade - a pesquisa web. Isso resulta em:
    1. Respostas mais focadas
    2. Menor confusão sobre qual ferramenta usar
    3. System prompt otimizado para pesquisa

    Example:
        >>> agent = WebSearchAgent(provider="openai")
        >>> response = agent.process_message("Quais as últimas notícias sobre IA?")
        >>> print(response)

        >>> response = agent.process_message("Pesquise sobre Python 3.12")
        >>> print(response)
    """

    # Tool única: web search
    SEARCH_TOOLS = [
        web_search_tool,
    ]

    # Modelos disponíveis por provider
    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4"],
        "google": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"]
    }

    def __init__(
        self,
        name: str = "Pesquisador Web",
        description: str = "Especialista em buscar informações atualizadas na internet",
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.5,
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
        memory_session_id: str = "websearch_agent"
    ):
        """
        Inicializa o Agente de Pesquisa Web.

        Args:
            name: Nome do agente
            description: Descrição do agente
            provider: Provider do LLM ("openai" ou "google")
            model: Modelo a usar
            temperature: Criatividade (0.5 é um bom equilíbrio)
            max_tokens: Limite de tokens na resposta
            top_p: Nucleus sampling
            presence_penalty: Penaliza repetição de tópicos (OpenAI)
            frequency_penalty: Penaliza repetição de palavras (OpenAI)
            top_k: Top-K sampling (Google)
            api_key: API key do provider
            system_prompt: Prompt de sistema customizado
            memory_type: Tipo de memória
            memory_max_messages: Máximo de mensagens no curto prazo
            memory_storage_path: Caminho para memória de longo prazo
            memory_session_id: ID da sessão de memória

        Note:
            A ferramenta web_search usa DuckDuckGo e não requer API key.
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

        # Configurar Tool (apenas web_search)
        self.tools = list(self.SEARCH_TOOLS)

        # System Prompt especializado
        self.system_prompt = system_prompt or self._get_search_system_prompt()

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

    def _get_search_system_prompt(self) -> str:
        """
        Retorna o system prompt especializado para pesquisa web.

        CONCEITO: Prompt para Search Agent
        ----------------------------------
        O prompt enfatiza:
        - Uso da ferramenta de busca para toda pergunta
        - Síntese de resultados de forma clara
        - Citação de fontes quando possível
        """
        return f"""Você é o {self.name}, um {self.description}.

## 🎯 SEU PAPEL
Você é um assistente especializado em pesquisar informações na internet.
Sua única ferramenta é a busca web, e você deve usá-la para responder perguntas.

## 🛠️ SUA FERRAMENTA

**web_search**: Busca informações na web usando DuckDuckGo
- Use para QUALQUER pergunta que precise de informações atualizadas
- Use para pesquisar notícias, artigos, tutoriais, etc.
- Use para verificar fatos e encontrar referências

## 📋 INSTRUÇÕES

1. **SEMPRE use web_search** para buscar informações antes de responder
2. **Reformule a query** se necessário para obter melhores resultados
3. **Sintetize os resultados** de forma clara e organizada
4. **Cite as fontes** quando possível (URLs dos resultados)
5. **Seja honesto** se não encontrar informações relevantes

## 🔍 ESTRATÉGIAS DE PESQUISA

Ao receber uma pergunta:

1. **Identifique os termos-chave** da pergunta
2. **Formule uma query de busca** eficiente
3. **Execute a busca** com web_search
4. **Analise os resultados** retornados
5. **Sintetize uma resposta** baseada nos resultados

## 📝 FORMATO DE RESPOSTA

Suas respostas devem incluir:

1. **Resumo direto** da informação encontrada
2. **Detalhes relevantes** dos resultados
3. **Fontes** (quando disponíveis nos resultados)
4. **Observações** sobre a qualidade/atualidade da informação

## 💡 EXEMPLOS DE USO

Perguntas que você responde bem:
- "Quais as últimas notícias sobre [tema]?"
- "Pesquise sobre [assunto]"
- "O que é [conceito]?"
- "Como fazer [tarefa]?"
- "Encontre informações sobre [tópico]"

## ⚠️ LIMITAÇÕES

- Os resultados dependem da disponibilidade do DuckDuckGo
- Informações podem estar desatualizadas
- Nem sempre é possível acessar o conteúdo completo das páginas
- Use os snippets/descrições retornados para sintetizar respostas

## 🗣️ TOM DE COMUNICAÇÃO

- Informativo e objetivo
- Sempre em português brasileiro
- Organize a informação de forma clara
- Use formatação (negrito, listas) para facilitar a leitura
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
        2. Formular uma query de busca
        3. Executar a busca web
        4. Sintetizar os resultados

        Args:
            message: Pergunta ou solicitação do usuário

        Returns:
            Resposta com informações da pesquisa web
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
                response = "❌ Erro ao processar sua pesquisa."

            # Atualiza a memória
            self._update_memory(message, response)

            return response

        except Exception as e:
            return f"❌ Erro na pesquisa: {str(e)}"

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
        """Retorna False (este agente não usa RAG)."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
            "description": self.description,
            "specialization": "web_search"
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
    print("🔍 TESTE DO WEB SEARCH AGENT")
    print("=" * 60)

    # Verifica API key
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY não configurada")
    else:
        print("✅ OPENAI_API_KEY configurada")

        print("\n" + "-" * 60)
        print("Criando Web Search Agent...")

        agent = WebSearchAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.5
        )

        print(f"✅ Agente criado: {agent.name}")
        print(f"📋 Tools disponíveis: {agent.list_tools()}")

        # Teste
        print("\n" + "-" * 60)
        print("🔍 Teste: Pesquisando sobre LangChain...")
        response = agent.process_message("Pesquise sobre LangChain e para que serve")
        print(response)

