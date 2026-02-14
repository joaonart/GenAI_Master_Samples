"""
=============================================================================
OLLAMA AGENT WITH TOOLS - Agente Ollama com Ferramentas
=============================================================================

Este agente utiliza modelos locais através do Ollama para conversação.

O que é Ollama?
- Ferramenta para rodar LLMs localmente no seu computador
- Não requer API key - totalmente gratuito e privado
- Suporta modelos como Llama, Mistral, CodeLlama, etc.

Por que usar Ollama?
- Privacidade: Seus dados nunca saem do seu computador
- Custo: Gratuito, sem limites de uso
- Offline: Funciona sem internet após baixar o modelo
- Customização: Você pode criar seus próprios modelos

Modelos populares do Ollama:
- llama3.2: Meta's Llama 3.2 (mais recente, bom equilíbrio)
- llama3.1: Meta's Llama 3.1 (excelente qualidade)
- mistral: Mistral 7B (leve e eficiente)
- codellama: Especializado em código
- phi3: Microsoft Phi-3 (compacto e capaz)
- gemma2: Google Gemma 2 (eficiente)
- qwen2.5: Alibaba Qwen 2.5 (bom em múltiplas línguas)

Como instalar Ollama:
1. Acesse https://ollama.ai
2. Baixe e instale para seu sistema operacional
3. Execute: ollama pull llama3.2 (ou outro modelo)
4. O servidor inicia automaticamente na porta 11434

RAG (Retrieval Augmented Generation):
- Permite que o agente consulte uma base de conhecimento
- Documentos são convertidos em vetores (embeddings)
- O agente pode buscar informações relevantes

MEMÓRIA:
- Curto Prazo: Mantém as últimas N mensagens
- Longo Prazo: Persiste informações importantes entre sessões
- Combinada: Usa ambas as estratégias

IMPORTANTE: Certifique-se de que o servidor Ollama está rodando!
Execute 'ollama serve' no terminal se necessário.

=============================================================================
"""

from typing import Optional, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .base_agent import BaseAgent
from core.memory import ShortTermMemory, LongTermMemory, CombinedMemory

# Importa as tools disponíveis
from tools import calculator_tool, get_current_datetime, web_search_tool
from tools import rag_search_tool, set_vector_store


class OllamaAgent(BaseAgent):
    """
    Agente Ollama com suporte a Tools, RAG e Memória usando LangGraph.

    Este agente roda modelos de IA LOCALMENTE usando Ollama.
    Não requer API key e seus dados ficam no seu computador.

    Pode:
    - Conversar naturalmente
    - Usar calculadora para cálculos
    - Consultar data/hora atual
    - Fazer buscas na web
    - Consultar base de conhecimento (RAG)
    - Lembrar conversas anteriores (Memória)

    Attributes:
        llm: Instância do modelo Ollama
        agent: Agente ReAct criado com LangGraph
        tools: Lista de ferramentas disponíveis
        memory: Sistema de memória configurado

    Example:
        >>> agent = OllamaAgent(model="llama3.2")
        >>> response = agent.process_message("Olá! Quanto é 15 * 7?")
        >>> print(response)
    """

    # Tools padrão disponíveis
    DEFAULT_TOOLS = [
        calculator_tool,
        get_current_datetime,
        web_search_tool,
    ]

    # Modelos populares e suas características
    POPULAR_MODELS = {
        "llama3.2": "Meta Llama 3.2 - Modelo mais recente, bom equilíbrio",
        "llama3.2:1b": "Llama 3.2 1B - Versão compacta, mais rápido",
        "llama3.1": "Meta Llama 3.1 - Excelente qualidade geral",
        "llama3.1:70b": "Llama 3.1 70B - Máxima qualidade (requer muita RAM)",
        "mistral": "Mistral 7B - Leve, rápido e eficiente",
        "mixtral": "Mixtral 8x7B - Mistura de especialistas",
        "codellama": "CodeLlama - Especializado em programação",
        "phi3": "Microsoft Phi-3 - Compacto mas capaz",
        "phi3:medium": "Phi-3 Medium - Versão intermediária",
        "gemma2": "Google Gemma 2 - Eficiente e bem treinado",
        "gemma2:27b": "Gemma 2 27B - Versão maior",
        "gemma3": "Google Gemma 3 - Mais recente (sem suporte a tools)",
        "qwen2.5": "Alibaba Qwen 2.5 - Bom em múltiplas línguas",
        "qwen2.5-coder": "Qwen 2.5 Coder - Especializado em código",
        "deepseek-coder-v2": "DeepSeek Coder V2 - Excelente para código",
    }

    # Modelos que NÃO suportam tools/function calling
    # Estes modelos serão usados em modo de chat simples
    MODELS_WITHOUT_TOOLS = [
        "gemma",       # Todas as versões do Gemma
        "gemma2",
        "gemma3",
        "phi",         # Phi-1, Phi-2
        "phi3",        # Phi-3 (algumas versões)
        "tinyllama",
        "orca-mini",
        "vicuna",
        "wizardlm",
        "falcon",
        "starcoder",
        "codellama",   # CodeLlama foca em código, tools limitado
        "deepseek-coder",
    ]

    def __init__(
        self,
        name: str = "Ollama Assistant",
        description: str = "Assistente IA local powered by Ollama",
        model: str = "gemma3",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        num_predict: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        system_prompt: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        tools: Optional[List[BaseTool]] = None,
        vector_store_manager=None,
        # Parâmetros de memória
        memory_type: str = "short_term",
        memory_max_messages: int = 20,
        memory_storage_path: str = "./memory_data",
        memory_session_id: str = "default"
    ):
        """
        Inicializa o agente Ollama com parâmetros configuráveis.

        Args:
            name: Nome do agente
            description: Descrição do agente
            model: Modelo Ollama a usar (llama3.2, mistral, codellama, etc.)
            temperature: Controla aleatoriedade (0.0 = determinístico, 2.0 = criativo)
            num_ctx: Tamanho do contexto em tokens (padrão: 4096)
            num_predict: Máximo de tokens a gerar (None = sem limite)
            top_p: Nucleus sampling - considera tokens até esta probabilidade
            top_k: Considera apenas os K tokens mais prováveis
            repeat_penalty: Penaliza repetição de tokens (1.0 = sem penalidade)
            system_prompt: Prompt de sistema customizado
            base_url: URL do servidor Ollama (padrão: http://localhost:11434)
            tools: Lista de tools customizadas
            vector_store_manager: Gerenciador do vector store para RAG
            memory_type: Tipo de memória (none, short_term, long_term, combined)
            memory_max_messages: Número máximo de mensagens no curto prazo
            memory_storage_path: Caminho para salvar memória de longo prazo
            memory_session_id: ID da sessão para memória de longo prazo

        Raises:
            ConnectionError: Se não conseguir conectar ao servidor Ollama

        Example:
            >>> # Usando modelo padrão
            >>> agent = OllamaAgent()
            >>>
            >>> # Usando modelo específico com temperatura baixa
            >>> agent = OllamaAgent(
            ...     model="codellama",
            ...     temperature=0.1,
            ...     description="Assistente de programação"
            ... )
        """
        super().__init__(name, description)

        # Guardar base_url para verificações
        self.base_url = base_url
        self.model_name = model

        # Verifica se o modelo suporta tools
        self.supports_tools = self._check_tools_support(model)

        # Inicializar o LLM Ollama
        # Nota: Ollama usa parâmetros diferentes da OpenAI
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,           # Tamanho do contexto
            num_predict=num_predict,    # Máximo de tokens na resposta
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            base_url=base_url,
        )

        # Configurar Tools (apenas se o modelo suportar)
        if self.supports_tools:
            self.tools = list(tools) if tools is not None else list(self.DEFAULT_TOOLS)
        else:
            self.tools = []  # Modelo não suporta tools

        # Configurar RAG se fornecido (apenas se suportar tools)
        self.vector_store_manager = vector_store_manager
        if self.supports_tools:
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

        # Criar agente ReAct com LangGraph (ou chat simples se não suportar tools)
        self._create_agent()

    def _check_tools_support(self, model: str) -> bool:
        """
        Verifica se o modelo suporta function calling/tools.

        Alguns modelos do Ollama não suportam tools nativamente.
        Nesse caso, usamos modo de chat simples.

        Args:
            model: Nome do modelo (ex: "llama3.2", "gemma3")

        Returns:
            True se suporta tools, False caso contrário
        """
        model_lower = model.lower()

        # Verifica se o modelo está na lista de modelos sem suporte
        for no_tools_model in self.MODELS_WITHOUT_TOOLS:
            if no_tools_model in model_lower:
                return False

        return True

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
        if self.vector_store_manager is not None and self.supports_tools:
            # Configura o vector store global para a tool
            set_vector_store(self.vector_store_manager)
            # Adiciona a tool de RAG se ainda não estiver na lista
            if rag_search_tool not in self.tools:
                self.tools.append(rag_search_tool)

    def _get_default_system_prompt(self):
        """Retorna o system prompt padrão."""
        base_prompt = f"""Você é {self.name}, {self.description}.

Você está rodando localmente através do Ollama usando o modelo {self.model_name}.
"""

        # Se o modelo suporta tools, lista as ferramentas disponíveis
        if self.supports_tools:
            base_prompt += """
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
5. Seja conciso e direto nas respostas
"""
        else:
            # Modelo sem suporte a tools - modo chat simples
            base_prompt += """
NOTA: Este modelo não suporta ferramentas externas (tools).
Responda às perguntas usando apenas seu conhecimento interno.

INSTRUÇÕES:
1. Responda em português brasileiro
2. Seja conciso e direto nas respostas
3. Se não souber algo, diga honestamente
4. Para cálculos complexos, explique o raciocínio passo a passo
"""
        return base_prompt

    def _create_agent(self):
        """
        Cria o agente ReAct usando LangGraph.

        Se o modelo não suportar tools, cria um agente sem ferramentas
        que funciona em modo de chat simples.
        """
        if self.supports_tools and self.tools:
            # Modelo suporta tools - cria agente ReAct completo
            self.agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
            )
        else:
            # Modelo não suporta tools - cria agente sem ferramentas
            self.agent = create_react_agent(
                model=self.llm,
                tools=[],  # Sem ferramentas
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

        Ollama pode retornar o conteúdo em diferentes formatos:
        - String simples: "texto da resposta"
        - Lista de blocos: [{"type": "text", "text": "..."}, ...]

        Esta função normaliza para string.

        Args:
            content: Conteúdo da resposta (string ou lista)

        Returns:
            String com o texto da resposta
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
        """
        Processa uma mensagem do usuário e retorna uma resposta.

        Este método:
        1. Monta o contexto com system prompt e histórico
        2. Adiciona contexto da memória de longo prazo (se disponível)
        3. Invoca o agente ReAct
        4. O agente decide se precisa usar tools
        5. Retorna a resposta final

        Args:
            message: A mensagem do usuário

        Returns:
            A resposta gerada pelo modelo

        Example:
            >>> agent = OllamaAgent()
            >>> response = agent.process_message("Quanto é 2 + 2?")
            >>> print(response)
            "2 + 2 = 4"
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
            error_msg = str(e)

            # Mensagens de erro mais amigáveis
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return (
                    "❌ Erro: Não foi possível conectar ao servidor Ollama.\n\n"
                    "Verifique se:\n"
                    "1. Ollama está instalado (https://ollama.ai)\n"
                    "2. O servidor está rodando (execute 'ollama serve')\n"
                    f"3. A URL está correta: {self.base_url}\n"
                    f"4. O modelo '{self.model_name}' está instalado "
                    f"(execute 'ollama pull {self.model_name}')"
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return (
                    f"❌ Erro: Modelo '{self.model_name}' não encontrado.\n\n"
                    f"Execute: ollama pull {self.model_name}\n\n"
                    "Modelos disponíveis populares:\n"
                    "- llama3.2 (recomendado)\n"
                    "- mistral\n"
                    "- codellama\n"
                    "- phi3"
                )
            elif "does not support tools" in error_msg.lower() or "status code: 400" in error_msg.lower():
                # Modelo não suporta tools - desabilita e tenta novamente
                if self.supports_tools:
                    self.supports_tools = False
                    self.tools = []
                    self._create_agent()
                    self.system_prompt = self._get_default_system_prompt()
                    # Tenta processar novamente sem tools
                    return self.process_message(message)
                else:
                    return (
                        f"❌ Erro: O modelo '{self.model_name}' não suporta ferramentas (tools).\n\n"
                        "Este modelo funciona apenas em modo de chat simples.\n"
                        "Tente perguntar novamente ou use um modelo com suporte a tools como:\n"
                        "- llama3.2\n"
                        "- llama3.1\n"
                        "- mistral\n"
                        "- qwen2.5"
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
            "enabled": self.memory is not None,
            "model": self.model_name,
            "base_url": self.base_url,
            "supports_tools": self.supports_tools
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
        """Adiciona uma nova tool ao agente (se o modelo suportar)."""
        if not self.supports_tools:
            print(f"⚠️ Modelo '{self.model_name}' não suporta tools. Tool não adicionada.")
            return
        self.tools.append(tool)
        self._create_agent()

    def list_tools(self) -> List[str]:
        """Lista os nomes das tools disponíveis."""
        if not self.supports_tools:
            return ["(modelo não suporta tools)"]
        return [tool.name for tool in self.tools]

    def has_rag(self) -> bool:
        """Retorna True se o RAG está habilitado e o modelo suporta tools."""
        return self.vector_store_manager is not None and self.supports_tools

    @classmethod
    def list_popular_models(cls) -> dict:
        """
        Retorna um dicionário com modelos populares e suas descrições.

        Returns:
            Dict com nome do modelo -> descrição

        Example:
            >>> models = OllamaAgent.list_popular_models()
            >>> for name, desc in models.items():
            ...     print(f"{name}: {desc}")
        """
        return cls.POPULAR_MODELS.copy()

    def change_model(self, model: str) -> None:
        """
        Troca o modelo sendo usado pelo agente.

        Útil para experimentar diferentes modelos sem criar
        um novo agente.

        Args:
            model: Nome do novo modelo (ex: "mistral", "codellama")

        Example:
            >>> agent = OllamaAgent(model="llama3.2")
            >>> agent.change_model("codellama")  # Troca para CodeLlama
        """
        self.model_name = model

        # Verifica se o novo modelo suporta tools
        self.supports_tools = self._check_tools_support(model)

        # Atualiza tools baseado no suporte
        if not self.supports_tools:
            self.tools = []
        elif not self.tools:
            # Se mudou para um modelo que suporta tools, restaura as tools padrão
            self.tools = list(self.DEFAULT_TOOLS)

        self.llm = ChatOllama(
            model=model,
            temperature=self.llm.temperature,
            num_ctx=self.llm.num_ctx,
            num_predict=self.llm.num_predict,
            top_p=self.llm.top_p,
            top_k=self.llm.top_k,
            repeat_penalty=self.llm.repeat_penalty,
            base_url=self.base_url,
        )
        self.system_prompt = self._get_default_system_prompt()
        self._create_agent()

