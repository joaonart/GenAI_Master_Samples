"""
=============================================================================
APP.PY - Aplicação Principal Streamlit
=============================================================================

Este é o ponto de entrada da aplicação.

Streamlit é um framework que permite criar interfaces web
interativas usando apenas Python, sem precisar de HTML/CSS/JS.

Como usar:
    poetry run streamlit run app.py

Conceitos demonstrados:
1. Interface de chat interativa
2. Seleção de diferentes agentes (OpenAI vs Gemini)
3. Configuração dinâmica de parâmetros
4. Gerenciamento de estado com session_state

=============================================================================
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Importa os agentes disponíveis
from agents import OpenAIAgent, GeminiAgent, OllamaAgent, SimpleAgent, FinanceAgent, KnowledgeAgent, WebSearchAgent, MCPAgentDemo, MEMORY_TYPES

# Tenta importar MCPAgent real (requer dependências extras)
try:
    from agents import MCPAgent
    MCP_REAL_AVAILABLE = True
except (ImportError, TypeError):
    MCPAgent = None
    MCP_REAL_AVAILABLE = False

# Importa módulos de RAG
from knowledge_base import (
    VectorStoreManager,
    load_document,
    split_documents,
    SUPPORTED_FORMATS
)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


# =============================================================================
# CONFIGURAÇÃO DOS AGENTES DISPONÍVEIS
# =============================================================================
# Adicione novos agentes aqui para que apareçam na interface
# Cada agente pode ter parâmetros específicos além dos comuns

AVAILABLE_AGENTS = {
    "🤖 Simple (OpenAI)": {
        "class": SimpleAgent,
        "description": "Agente simples sem tools e RAG (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
        "extra_params": ["presence_penalty", "frequency_penalty"],
        "provider": "openai"
    },
    "🤖 Simple (Gemini)": {
        "class": SimpleAgent,
        "description": "Agente simples sem tools e RAG (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google"
    },
    "🛠️ Tools (OpenAI)": {
        "class": OpenAIAgent,
        "description": "Agente usando OpenAI GPT-4 com tools",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4", "gpt-4o"],
        # Parâmetros específicos do OpenAI
        "extra_params": ["presence_penalty", "frequency_penalty"]
    },
    "🛠️ Tools (Gemini)": {
        "class": GeminiAgent,
        "description": "Agente usando Google Gemini com tools",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        # Parâmetros específicos do Gemini
        "extra_params": ["top_k"]
    },
    "🦙 Ollama (Local)": {
        "class": OllamaAgent,
        "description": "Agente usando modelos locais via Ollama (sem API key!)",
        "api_key_env": None,  # Ollama não precisa de API key
        "api_key_url": "https://ollama.ai",
        "models": ["llama3.2", "llama3.1", "mistral", "codellama", "phi3", "gemma2", "gemma3", "qwen2.5"],
        # Parâmetros específicos do Ollama
        "extra_params": ["num_ctx", "repeat_penalty"],
        "is_local": True
    },
    # "💰 Finance (OpenAI)": {
    #     "class": FinanceAgent,
    #     "description": "Especialista em finanças: ações, crypto e câmbio (OpenAI)",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "api_key_url": "https://platform.openai.com/api-keys",
    #     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
    #     "extra_params": ["presence_penalty", "frequency_penalty"],
    #     "provider": "openai",
    #     "is_specialist": True
    # },
    "💰 Finance (Gemini)": {
        "class": FinanceAgent,
        "description": "Especialista em finanças: ações, crypto e câmbio (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google",
        "is_specialist": True
    },
    # "📚 Knowledge (OpenAI)": {
    #     "class": KnowledgeAgent,
    #     "description": "Especialista em conhecimento: Wikipedia e informações enciclopédicas (OpenAI)",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "api_key_url": "https://platform.openai.com/api-keys",
    #     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
    #     "extra_params": ["presence_penalty", "frequency_penalty"],
    #     "provider": "openai",
    #     "is_specialist": True
    # },
    "📚 Knowledge (Gemini)": {
        "class": KnowledgeAgent,
        "description": "Especialista em conhecimento: Wikipedia e informações enciclopédicas (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google",
        "is_specialist": True
    },
    # "🔍 Web Search (OpenAI)": {
    #     "class": WebSearchAgent,
    #     "description": "Especialista em pesquisa web: busca informações atualizadas na internet (OpenAI)",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "api_key_url": "https://platform.openai.com/api-keys",
    #     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
    #     "extra_params": ["presence_penalty", "frequency_penalty"],
    #     "provider": "openai",
    #     "is_specialist": True
    # },
    "🔍 Web Search (Gemini)": {
        "class": WebSearchAgent,
        "description": "Especialista em pesquisa web: busca informações atualizadas na internet (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google",
        "is_specialist": True
    },
    # "🔌 MCP Demo (OpenAI)": {
    #     "class": MCPAgentDemo,
    #     "description": "Demonstração do Model Context Protocol (MCP) - conecta a servidores externos (OpenAI)",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "api_key_url": "https://platform.openai.com/api-keys",
    #     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
    #     "extra_params": ["presence_penalty", "frequency_penalty"],
    #     "provider": "openai",
    #     "is_specialist": True,
    #     "mcp_server": "fetch"
    # },
    "🔌 MCP Demo (Gemini)": {
        "class": MCPAgentDemo,
        "description": "Demonstração do Model Context Protocol (MCP) - conecta a servidores externos (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google",
        "is_specialist": True,
        "mcp_server": "fetch"
    }
}

# Adiciona MCPAgent real se disponível
if MCP_REAL_AVAILABLE and MCPAgent is not None:
    # AVAILABLE_AGENTS["🔌 MCP Fetch (OpenAI)"] = {
    #     "class": MCPAgent,
    #     "description": "🌐 MCP REAL: Busca e extrai conteúdo de URLs (OpenAI)",
    #     "api_key_env": "OPENAI_API_KEY",
    #     "api_key_url": "https://platform.openai.com/api-keys",
    #     "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
    #     "extra_params": ["presence_penalty", "frequency_penalty"],
    #     "provider": "openai",
    #     "is_specialist": True,
    #     "mcp_server": "fetch"
    # }
    AVAILABLE_AGENTS["🔌 MCP Fetch (Gemini)"] = {
        "class": MCPAgent,
        "description": "🌐 MCP REAL: Busca e extrai conteúdo de URLs (Gemini)",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_url": "https://makersuite.google.com/app/apikey",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.0-flash"],
        "extra_params": ["top_k"],
        "provider": "google",
        "is_specialist": True,
        "mcp_server": "fetch"
    }
    AVAILABLE_AGENTS["🔌 MCP Time (OpenAI)"] = {
        "class": MCPAgent,
        "description": "🕐 MCP REAL: Informações de data, hora e fuso horário (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
        "extra_params": ["presence_penalty", "frequency_penalty"],
        "provider": "openai",
        "is_specialist": True,
        "mcp_server": "time"
    }
    AVAILABLE_AGENTS["🔌 MCP Filesystem (OpenAI)"] = {
        "class": MCPAgent,
        "description": "📁 MCP REAL: Lê e escreve arquivos locais (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
        "extra_params": ["presence_penalty", "frequency_penalty"],
        "provider": "openai",
        "is_specialist": True,
        "mcp_server": "filesystem"
    }


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="AI Agent Chat - Trilha Master GenAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    /* Estilo do cabeçalho */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards de informação */
    .info-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    
    /* Destaque para tools */
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 1rem;
        background-color: #e1e5eb;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def create_agent(
    agent_name: str,
    model: str,
    temperature: float,
    system_prompt: str,
    guardrails: str = "",
    max_tokens: int = None,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    top_k: int = None,
    vector_store_manager=None,
    # Parâmetros de memória
    memory_type: str = "short_term",
    memory_max_messages: int = 20,
    memory_storage_path: str = "./memory_data",
    memory_session_id: str = "default"
):
    """
    Cria uma instância do agente selecionado com todos os parâmetros.

    Esta função é chamada quando o usuário:
    - Seleciona um novo agente
    - Muda o modelo
    - Clica em "Aplicar Configurações"

    Args:
        agent_name: Nome do agente selecionado
        model: Modelo a usar
        temperature: Temperatura para geração (criatividade)
        system_prompt: Prompt de sistema customizado
        guardrails: Regras de segurança adicionadas ao final do system prompt
        max_tokens: Limite de tokens na resposta (None = ilimitado)
        top_p: Nucleus sampling (0.0 a 1.0)
        presence_penalty: Penalidade por repetir tópicos (OpenAI)
        frequency_penalty: Penalidade por repetir palavras (OpenAI)
        top_k: Top-K sampling (Gemini)
        vector_store_manager: Gerenciador de vector store para RAG
        memory_type: Tipo de memória (none, short_term, long_term, combined)
        memory_max_messages: Máximo de mensagens no curto prazo
        memory_storage_path: Caminho para salvar memória de longo prazo
        memory_session_id: ID da sessão de memória

    Returns:
        Instância do agente ou None se houver erro
    """
    agent_config = AVAILABLE_AGENTS[agent_name]
    agent_class = agent_config["class"]

    try:
        # Combina system_prompt com guardrails
        full_system_prompt = system_prompt.strip()
        if guardrails.strip():
            full_system_prompt += f"\n\n{guardrails.strip()}"

        # Parâmetros comuns a todos os agentes
        common_params = {
            "name": agent_name,
            "description": agent_config["description"],
            "model": model,
            "temperature": temperature,
            "system_prompt": full_system_prompt if full_system_prompt else None,
            "max_tokens": max_tokens,
            "top_p": top_p,
            # Parâmetros de memória
            "memory_type": memory_type,
            "memory_max_messages": memory_max_messages,
            "memory_storage_path": memory_storage_path,
            "memory_session_id": memory_session_id,
        }

        # Verifica se é um agente com provider (SimpleAgent ou FinanceAgent)
        has_provider = "provider" in agent_config
        is_specialist = agent_config.get("is_specialist", False)

        if has_provider:
            # Agentes com provider (SimpleAgent, FinanceAgent)
            common_params["provider"] = agent_config["provider"]

        if not has_provider and not is_specialist:
            # Agentes genéricos com tools (OpenAI, Gemini) usam vector_store_manager
            common_params["vector_store_manager"] = vector_store_manager

        # Adiciona parâmetros específicos do OpenAI
        if "OpenAI" in agent_name or "Finance (OpenAI)" in agent_name:
            common_params["presence_penalty"] = presence_penalty
            common_params["frequency_penalty"] = frequency_penalty

        # Adiciona parâmetros específicos do Gemini
        if "Gemini" in agent_name or "Finance (Gemini)" in agent_name:
            common_params["top_k"] = top_k

        # Adiciona parâmetros específicos do Ollama
        if "Ollama" in agent_name:
            # Ollama não usa max_tokens, usa num_predict
            common_params["num_predict"] = max_tokens
            del common_params["max_tokens"]
            # Adiciona vector_store_manager para RAG
            common_params["vector_store_manager"] = vector_store_manager

        # Adiciona parâmetros específicos do MCP
        if "mcp_server" in agent_config:
            common_params["mcp_server_name"] = agent_config["mcp_server"]

        agent = agent_class(**common_params)
        return agent

    except ValueError as e:
        # Erro geralmente significa API key não configurada
        st.error(f"⚠️ {str(e)}")
        return None


def initialize_session_state():
    """
    Inicializa o estado da sessão do Streamlit.

    session_state é como uma "memória" que persiste entre
    as interações do usuário. Sem ela, cada clique
    recarregaria toda a página e perderia os dados.
    """
    # Dicionário de chats: {chat_id: {"name": str, "messages": list, "agent": obj}}
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "chat_1": {
                "name": "Chat 1",
                "messages": [],
                "agent": None,
                "agent_name": None,
                "model": None,
                "config": None  # Armazena toda a configuração do chat
            }
        }

    # ID do chat ativo
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat_1"

    # Contador para gerar IDs únicos
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 1

    # Manter compatibilidade com código existente
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "current_agent_name" not in st.session_state:
        st.session_state.current_agent_name = None

    # RAG - Base de Conhecimento
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None

    if "rag_documents" not in st.session_state:
        st.session_state.rag_documents = []  # Lista de documentos carregados

    if "rag_storage_path" not in st.session_state:
        st.session_state.rag_storage_path = None  # Caminho onde a base está salva (None = memória)

    # Configuração atual do chat (para persistir entre reloads)
    if "current_model" not in st.session_state:
        st.session_state.current_model = None

    if "current_config" not in st.session_state:
        st.session_state.current_config = None


def get_active_chat():
    """Retorna o chat ativo atual."""
    return st.session_state.chats.get(st.session_state.active_chat_id, None)


def create_new_chat():
    """Cria um novo chat e o torna ativo, preservando o agente selecionado."""
    # Guarda o agente e modelo do chat atual antes de criar o novo
    current_chat = st.session_state.chats.get(st.session_state.active_chat_id, {})
    current_agent_name = current_chat.get("agent_name") or st.session_state.current_agent_name
    current_model = current_chat.get("model") or st.session_state.current_model
    current_config = current_chat.get("config") or st.session_state.current_config

    st.session_state.chat_counter += 1
    new_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[new_id] = {
        "name": f"Chat {st.session_state.chat_counter}",
        "messages": [],
        "agent": None,  # Será recriado quando o usuário enviar mensagem
        "agent_name": current_agent_name,  # Preserva o agente selecionado
        "model": current_model,  # Preserva o modelo selecionado
        "config": current_config  # Preserva as configurações
    }
    st.session_state.active_chat_id = new_id
    # Sincroniza com variáveis de compatibilidade
    st.session_state.messages = []
    st.session_state.agent = None  # Será recriado
    # Mantém o agente e modelo selecionados (não reseta para None)
    # st.session_state.current_agent_name permanece o mesmo
    # st.session_state.current_model permanece o mesmo
    return new_id


def switch_chat(chat_id: str):
    """Alterna para um chat específico, carregando suas configurações."""
    if chat_id in st.session_state.chats:
        st.session_state.active_chat_id = chat_id
        chat = st.session_state.chats[chat_id]
        # Sincroniza com variáveis de compatibilidade
        st.session_state.messages = chat["messages"]
        st.session_state.agent = chat["agent"]
        st.session_state.current_agent_name = chat["agent_name"]
        # Carrega o modelo do chat se existir
        if "model" in chat:
            st.session_state.current_model = chat.get("model")
        if "config" in chat:
            st.session_state.current_config = chat.get("config")


def delete_chat(chat_id: str):
    """Remove um chat (não pode remover o último)."""
    if len(st.session_state.chats) > 1 and chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # Se deletou o chat ativo, muda para outro
        if st.session_state.active_chat_id == chat_id:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
            switch_chat(st.session_state.active_chat_id)


def rename_chat(chat_id: str, new_name: str):
    """Renomeia um chat."""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["name"] = new_name


def clear_chat(chat_id: str):
    """Limpa as mensagens de um chat (reinicia a conversa), mantendo o agente."""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["messages"] = []
        # Se for o chat ativo, sincroniza
        if chat_id == st.session_state.active_chat_id:
            st.session_state.messages = []
            # Limpa apenas o histórico do agente, mas mantém o agente
            if st.session_state.agent:
                st.session_state.agent.clear_history()
        # Mantém as configurações do agente (agent, agent_name, model, config)


def sync_active_chat(config: dict = None):
    """Sincroniza o estado do chat ativo com as variáveis globais."""
    chat_id = st.session_state.active_chat_id
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["messages"] = st.session_state.messages
        st.session_state.chats[chat_id]["agent"] = st.session_state.agent
        st.session_state.chats[chat_id]["agent_name"] = st.session_state.current_agent_name
        # Salva as configurações se fornecidas
        if config:
            st.session_state.chats[chat_id]["model"] = config.get("model")
            st.session_state.chats[chat_id]["config"] = config


def display_sidebar():
    """
    Renderiza a barra lateral com configurações.

    A sidebar contém:
    - Seleção de agente
    - Input de API key
    - Seleção de modelo
    - Slider de temperatura
    - Área de system prompt
    - Botões de ação

    Returns:
        Tupla com as configurações selecionadas
    """
    with st.sidebar:
        # -----------------------------------------------------------------
        # GERENCIAMENTO DE CHATS
        # -----------------------------------------------------------------
        st.subheader("💬 Chats")

        # Botão para criar novo chat
        if st.button("➕ Novo Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        with st.expander("Histórico das Conversas", expanded=False):
            # Lista de chats existentes
            for chat_id, chat_data in st.session_state.chats.items():
                is_active = chat_id == st.session_state.active_chat_id

                col1, col2 = st.columns([4, 1])

                with col1:
                    # Monta o label do chat com informações do agente
                    chat_label = chat_data['name']
                    if chat_data.get('agent_name'):
                        chat_label += f" ({chat_data['agent_name']})"

                    # Botão para selecionar o chat
                    if st.button(
                        f"{'' if is_active else ''}{chat_label}",
                        key=f"select_{chat_id}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True
                    ):
                        if not is_active:
                            sync_active_chat()  # Salva o chat atual antes de trocar
                            switch_chat(chat_id)
                            st.rerun()

                with col2:
                    # Botão sempre visível
                    if len(st.session_state.chats) > 1:
                        # Se há mais de 1 chat, exclui o chat
                        if st.button("🗑️", key=f"delete_{chat_id}", help="Excluir chat"):
                            delete_chat(chat_id)
                            st.rerun()
                    else:
                        # Se é o único chat, limpa a conversa
                        if st.button("🗑️", key=f"clear_{chat_id}", help="Limpar conversa"):
                            clear_chat(chat_id)
                            st.rerun()

        st.divider()

        # -----------------------------------------------------------------
        # CABEÇALHO
        # -----------------------------------------------------------------
        st.header("⚙️ Configurações")

        # -----------------------------------------------------------------
        # SELEÇÃO DE AGENTE
        # -----------------------------------------------------------------
        st.subheader("1. Escolha o Agente")

        # Obtém a configuração salva do chat ativo (se existir)
        active_chat = st.session_state.chats.get(st.session_state.active_chat_id, {})
        saved_agent_name = active_chat.get("agent_name")
        saved_model = active_chat.get("model")

        # Determina o índice do agente selecionado
        agent_options = list(AVAILABLE_AGENTS.keys())
        default_agent_index = 0
        if saved_agent_name and saved_agent_name in agent_options:
            default_agent_index = agent_options.index(saved_agent_name)

        # Usa key única para o chat ativo - isso garante que ao trocar de chat,
        # o selectbox seja recriado com o valor correto
        agent_select_key = f"agent_select_{st.session_state.active_chat_id}"

        selected_agent = st.selectbox(
            "Agente de IA",
            options=agent_options,
            index=default_agent_index,
            key=agent_select_key,
            help="Escolha qual agente de IA usar"
        )

        agent_config = AVAILABLE_AGENTS[selected_agent]

        # Determina o índice do modelo selecionado
        model_options = agent_config["models"]
        default_model_index = 0
        if saved_model and saved_model in model_options and saved_agent_name == selected_agent:
            default_model_index = model_options.index(saved_model)

        # Usa key única vinculada ao chat e ao agente - ao mudar de agente,
        # o selectbox de modelo é recriado com as opções corretas
        model_select_key = f"model_select_{st.session_state.active_chat_id}_{selected_agent}"

        model = st.selectbox(
            "Modelo",
            options=model_options,
            index=default_model_index,
            key=model_select_key,
            help="Modelos mais capazes geralmente são mais lentos e caros"
        )

        # Mostra descrição do agente
        st.caption(agent_config["description"])

        # -----------------------------------------------------------------
        # CONFIGURAÇÕES DO MODELO
        # -----------------------------------------------------------------
        st.subheader("2. Configurações do Modelo")

        # Parâmetros avançados dentro de um expander
        # Valores padrão (caso o expander não seja aberto)
        temperature = 0.7
        max_tokens = None
        top_p = 1.0
        presence_penalty = 0.0
        frequency_penalty = 0.0
        top_k = None

        with st.expander("Parâmetros Avançados", expanded=False):
            st.caption("Ajuste fino do comportamento do modelo")

            # Temperatura - controla criatividade (sempre visível)
            temperature = st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="""
                          Controla a criatividade/aleatoriedade:
                          - 0.0 = Respostas mais determinísticas e focadas
                          - 0.7 = Balanceado (recomendado)
                          - 2.0 = Muito criativo (pode ser confuso)
                          """
            )

            # Max Tokens - limite de resposta
            max_tokens = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=4096,
                value=1024,
                step=50,
                help="""
                       Número máximo de tokens na resposta.
                       1 token ≈ 4 caracteres em inglês, 2-3 em português.
                       - 256 = Respostas curtas
                       - 1024 = Respostas médias
                       - 4096 = Respostas longas
                       """
            )

            # Top P - Nucleus Sampling
            top_p = st.slider(
                "Top P (Nucleus Sampling)",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="""
                       Alternativa à temperatura para controlar aleatoriedade.
                       Considera apenas tokens cuja probabilidade acumulada ≤ top_p.
                       - 1.0 = Considera todos os tokens
                       - 0.9 = Considera os 90% mais prováveis
                       - 0.5 = Mais focado, menos diversidade

                       💡 Dica: Use temperatura OU top_p, não ambos!
                       """
            )

            # Parâmetros específicos do OpenAI
            if selected_agent == "OpenAI":
                st.markdown("---")
                st.markdown("##### Parâmetros OpenAI")

                presence_penalty = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                    help="""
                           Penaliza tokens que já apareceram no texto.
                           - Valores positivos = Incentiva novos tópicos
                           - Valores negativos = Permite repetição de tópicos
                           - 0.0 = Sem efeito
                           """
                )

                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                    help="""
                           Penaliza tokens baseado na frequência de uso.
                           - Valores positivos = Evita repetir palavras
                           - Valores negativos = Permite mais repetição
                           - 0.0 = Sem efeito
                           """
                )

            # Parâmetros específicos do Gemini
            if selected_agent == "Gemini":
                st.markdown("---")
                st.markdown("##### Parâmetros Gemini")

                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=40,
                    step=1,
                    help="""
                           Considera apenas os K tokens mais prováveis.
                           - Valores baixos (1-10) = Mais determinístico
                           - Valores médios (20-40) = Balanceado
                           - Valores altos (50-100) = Mais diversidade

                           💡 Use junto com top_p para controle fino.
                           """
                )

            # Parâmetros específicos do Ollama
            if "Ollama" in selected_agent:
                st.markdown("---")
                st.markdown("##### Parâmetros Ollama")
                st.info("🦙 Ollama roda localmente - não precisa de API key!")

                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=40,
                    step=1,
                    help="""
                           Considera apenas os K tokens mais prováveis.
                           - Valores baixos (1-10) = Mais determinístico
                           - Valores médios (20-40) = Balanceado
                           - Valores altos (50-100) = Mais diversidade
                           """
                )

        # -----------------------------------------------------------------
        # API KEY (não mostrar para Ollama)
        # -----------------------------------------------------------------
        st.subheader("3. Configurações de Acesso")

        # Ollama não precisa de API Key
        is_local_agent = agent_config.get("is_local", False)

        if is_local_agent:
            st.success("✅ Este agente roda localmente via Ollama - não precisa de API key!")
            st.markdown("""
            **Certifique-se de que:**
            1. Ollama está instalado ([baixe aqui](https://ollama.ai))
            2. O servidor está rodando (execute `ollama serve` no terminal)
            3. O modelo está baixado (execute `ollama pull llama3.2`)
            """)
        else:
            with st.expander("Informar API Key", expanded=False):
                st.caption("Configurar API Key para o agente funcionar")

                api_key_env = agent_config["api_key_env"]
                current_key = os.getenv(api_key_env, "")

                api_key = st.text_input(
                    f"{api_key_env}",
                    value=current_key,
                    type="password",
                    help=f"Obtenha em: {agent_config['api_key_url']}"
                )

                # Atualiza a variável de ambiente se mudou
                if api_key and api_key != current_key:
                    os.environ[api_key_env] = api_key
                    # Força recriar o agente
                    if "agent" in st.session_state:
                        st.session_state.agent = None

                # Link para obter API key
                st.markdown(f"[🔑 Obter API Key]({agent_config['api_key_url']})")

        st.divider()

        # -----------------------------------------------------------------
        # PERSONALIDADE DO AGENTE
        # -----------------------------------------------------------------
        st.subheader("4. Personalidade do Agente")

        # Mensagem de Boas-vindas
        with st.expander("👋 Boas-vindas", expanded=False):
            welcome_message = st.text_area(
                "Mensagem inicial exibida no chat",
                value="""Olá! 👋 Sou seu assistente de IA.

                Posso ajudá-lo com:
                • Cálculos matemáticos
                • Informações de data e hora
                • Pesquisas na web
                
                Como posso ajudar você hoje?""",
            height=120,
            help="""
            Mensagem exibida quando o chat é iniciado.
            Use para apresentar o agente e suas capacidades.
            """
        )

        # System Prompt
        with st.expander("🧠 System Prompt", expanded=False):
            system_prompt = st.text_area(
                "Define o comportamento do agente",
                value="""Você é um assistente útil e amigável.
                Responda de forma clara e educada.
                Use as ferramentas disponíveis quando necessário.""",
                height=120,
                help="""
                Define o comportamento e personalidade do agente.
                O usuário NÃO vê este prompt, mas ele influencia
                como o agente responde.
                """
            )

        # Guardrails
        with st.expander("🛡️ Guardrails", expanded=False):
            guardrails = st.text_area(
                "Regras de Segurança e Limites",
                value="""REGRAS QUE VOCÊ DEVE SEGUIR:
                1. Nunca forneça informações falsas
                2. Se não souber algo, admita
                3. Não discuta temas ilegais ou antiéticos
                4. Mantenha respostas respeitosas e profissionais
                5. Proteja a privacidade dos usuários""",
                height=120,
                help="""
                Regras de segurança e limites que o agente deve respeitar.
                Guardrails ajudam a evitar respostas inadequadas.
                São adicionados ao final do system prompt.
                """
            )

        # -----------------------------------------------------------------
        # BOTÕES DE AÇÃO
        # -----------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Aplicar", type="primary", use_container_width=True):
                st.session_state.agent = None
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("Limpar", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.clear_history()
                st.rerun()

        st.divider()

        # -----------------------------------------------------------------
        # CONFIGURAÇÃO DE MEMÓRIA
        # -----------------------------------------------------------------
        st.subheader("5. Memória do Agente")

        with st.expander("🧠 Configurar Memória", expanded=False):
            st.caption("""
            Configure como o agente lembra das conversas.
            """)

            # Seleção do tipo de memória
            memory_type = st.selectbox(
                "Tipo de Memória",
                options=list(MEMORY_TYPES.keys()),
                format_func=lambda x: f"{MEMORY_TYPES[x]['icon']} {MEMORY_TYPES[x]['name']}",
                help="Escolha como o agente vai lembrar das conversas"
            )

            # Mostra descrição do tipo selecionado
            st.caption(MEMORY_TYPES[memory_type]["description"])

            # Configurações específicas por tipo
            memory_max_messages = 20
            memory_storage_path = "./memory_data"
            memory_session_id = "default"

            if memory_type in ["short_term", "combined"]:
                memory_max_messages = st.slider(
                    "Máximo de Mensagens (Curto Prazo)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Quantas mensagens manter no histórico de curto prazo"
                )

            if memory_type in ["long_term", "combined"]:
                memory_storage_path = st.text_input(
                    "Caminho para Salvar Memória",
                    value="./memory_data",
                    help="Diretório onde as memórias de longo prazo serão salvas"
                )

                memory_session_id = st.text_input(
                    "ID da Sessão",
                    value="default",
                    help="Identificador único para esta sessão de memória"
                )

            # Placeholder para status da memória (será atualizado depois)
            memory_status_placeholder = st.empty()

        st.divider()

        # -----------------------------------------------------------------
        # BASE DE CONHECIMENTO (RAG)
        # -----------------------------------------------------------------
        st.subheader("6. Base de Conhecimento")

        with st.expander("📚 RAG - Upload de Documentos", expanded=False):
            st.caption("""
            Carregue documentos para o agente consultar.
            O agente poderá buscar informações relevantes
            para responder suas perguntas.
            """)

            # Mostra formatos suportados
            formats_list = ", ".join(SUPPORTED_FORMATS.keys())
            st.info(f"📁 Formatos suportados: {formats_list}")

            # Upload de arquivos - agora aceita mais formatos
            uploaded_files = st.file_uploader(
                "Selecione arquivos",
                type=["txt", "md", "pdf", "csv", "docx", "json"],
                accept_multiple_files=True,
                help="Arraste arquivos ou clique para selecionar"
            )

            # Configurações de chunking
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input(
                    "Tamanho do Chunk",
                    min_value=100,
                    max_value=4000,
                    value=1000,
                    step=100,
                    help="Tamanho de cada pedaço de texto"
                )
            with col2:
                chunk_overlap = st.number_input(
                    "Overlap",
                    min_value=0,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Sobreposição entre chunks"
                )

            # Opção de persistência
            st.markdown("---")
            st.markdown("##### 💾 Armazenamento")

            storage_option = st.radio(
                "Onde armazenar a base de conhecimento?",
                options=["memory", "disk"],
                format_func=lambda x: "🧠 Memória (temporário)" if x == "memory" else "💾 Disco (persistente)",
                help="""
                **Memória**: Mais rápido, mas perde os dados ao fechar o app.
                **Disco**: Mais lento para criar, mas mantém os dados entre sessões.
                """,
                horizontal=True
            )

            # Caminho para salvar (só aparece se escolher disco)
            save_path = None
            if storage_option == "disk":
                save_path = st.text_input(
                    "Caminho para salvar",
                    value="./knowledge_base_data",
                    help="Diretório onde os dados serão salvos"
                )

            # Botão para processar documentos
            if st.button("📥 Processar Documentos", use_container_width=True):
                if uploaded_files:
                    with st.spinner("Processando documentos..."):
                        try:
                            all_documents = []

                            for uploaded_file in uploaded_files:
                                st.text(f"📄 Processando: {uploaded_file.name}")

                                # Usa a função universal load_document
                                docs = load_document(
                                    file_content=uploaded_file.read(),
                                    filename=uploaded_file.name
                                )
                                all_documents.extend(docs)

                            # Divide em chunks
                            chunks = split_documents(
                                all_documents,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )

                            # Cria vector store
                            vector_manager = VectorStoreManager()
                            vector_manager.create_from_documents(chunks)

                            # Salva em disco se solicitado
                            if storage_option == "disk" and save_path:
                                vector_manager.save(save_path)
                                st.info(f"💾 Base salva em: {save_path}")

                            # Salva no session_state
                            st.session_state.vector_store_manager = vector_manager
                            st.session_state.rag_documents = [f.name for f in uploaded_files]
                            st.session_state.rag_storage_path = save_path if storage_option == "disk" else None

                            # Força recriar o agente com RAG
                            st.session_state.agent = None

                            st.success(
                                f"✅ {len(chunks)} chunks criados de "
                                f"{len(uploaded_files)} arquivo(s)!"
                            )
                            st.rerun()

                        except ImportError as e:
                            st.error(f"❌ Biblioteca necessária não instalada: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Erro ao processar: {str(e)}")
                else:
                    st.warning("Selecione pelo menos um arquivo.")

            # Seção para carregar base existente do disco
            st.markdown("---")
            st.markdown("##### 📂 Carregar Base Existente")

            load_path = st.text_input(
                "Caminho da base salva",
                value="./knowledge_base_data",
                key="load_kb_path",
                help="Diretório onde a base foi salva anteriormente"
            )

            if st.button("📂 Carregar do Disco", use_container_width=True):
                if load_path:
                    try:
                        with st.spinner("Carregando base de conhecimento..."):
                            vector_manager = VectorStoreManager()
                            vector_manager.load(load_path)

                            st.session_state.vector_store_manager = vector_manager
                            st.session_state.rag_documents = [f"Base carregada de: {load_path}"]
                            st.session_state.rag_storage_path = load_path
                            st.session_state.agent = None

                            st.success("✅ Base de conhecimento carregada!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar: {str(e)}")
                else:
                    st.warning("Informe o caminho da base.")

            # Mostra documentos carregados
            if st.session_state.rag_documents:
                st.markdown("---")
                st.markdown("**📋 Status da Base:**")
                for doc_name in st.session_state.rag_documents:
                    st.markdown(f"• {doc_name}")

                # Mostra onde está armazenado
                if hasattr(st.session_state, 'rag_storage_path') and st.session_state.rag_storage_path:
                    st.caption(f"💾 Salvo em: {st.session_state.rag_storage_path}")
                else:
                    st.caption("🧠 Armazenado em memória (temporário)")

                if st.button("🗑️ Limpar Base de Conhecimento", use_container_width=True):
                    st.session_state.vector_store_manager = None
                    st.session_state.rag_documents = []
                    st.session_state.rag_storage_path = None
                    st.session_state.agent = None
                    st.success("Base de conhecimento limpa!")
                    st.rerun()

        st.divider()

        # -----------------------------------------------------------------
        # SOBRE
        # -----------------------------------------------------------------
        st.subheader("Sobre o Projeto")
        with st.expander("Trilha Master GenAI", expanded=False):
            st.markdown("""
            Este projeto demonstra:
            - Criação de Agentes de IA
            - Uso do Framework LangChain
            - Uso de Tools (Ferramentas)
            - Uso de RAG (Base de Conhecimento)
            - Interface com Streamlit
            """)

        # Retorna todos os parâmetros configurados
        return {
            "agent_name": selected_agent,
            "model": model,
            "temperature": temperature,
            "welcome_message": welcome_message,
            "system_prompt": system_prompt,
            "guardrails": guardrails,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_k": top_k,
            # Parâmetros de memória
            "memory_type": memory_type,
            "memory_max_messages": memory_max_messages,
            "memory_storage_path": memory_storage_path,
            "memory_session_id": memory_session_id,
            # Placeholder para atualização dinâmica
            "memory_status_placeholder": memory_status_placeholder
        }


def display_agent_info(agent):
    """
    Mostra informações sobre o agente atual.

    Útil para debug e para o usuário saber
    quais ferramentas estão disponíveis.
    """
    if agent is None:
        return

    with st.expander("Informações do Agente", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Nome:** {agent.name}")
            st.write(f"**Descrição:** {agent.description}")

        with col2:
            st.write("**Tools Disponíveis:**")
            for tool_name in agent.list_tools():
                st.write(f"  • {tool_name}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """
    Função principal da aplicação.

    Fluxo:
    1. Inicializa o estado da sessão
    2. Renderiza a sidebar com configurações
    3. Cria/atualiza o agente se necessário
    4. Exibe o histórico de mensagens
    5. Processa novas mensagens do usuário
    """

    # -----------------------------------------------------------------
    # INICIALIZAÇÃO
    # -----------------------------------------------------------------
    initialize_session_state()

    # Carrega o chat ativo
    switch_chat(st.session_state.active_chat_id)

    # -----------------------------------------------------------------
    # CABEÇALHO
    # -----------------------------------------------------------------
    st.markdown("<h1 class='main-header'>AI Agent Chat</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Trilha Master GenAI • 2026"
        "</p>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------------------
    config = display_sidebar()

    # -----------------------------------------------------------------
    # CRIAR/ATUALIZAR AGENTE
    # -----------------------------------------------------------------
    # Verifica se precisa criar um novo agente
    # Isso acontece quando:
    # 1. Não existe agente ainda
    # 2. O tipo de agente mudou (ex: OpenAI -> Gemini)
    # 3. O modelo mudou (ex: gpt-4 -> gpt-4o)
    need_new_agent = (
        st.session_state.agent is None or
        st.session_state.current_agent_name != config["agent_name"] or
        st.session_state.current_model != config["model"]
    )

    if need_new_agent:
        with st.spinner("🔄 Inicializando agente..."):
            agent = create_agent(
                agent_name=config["agent_name"],
                model=config["model"],
                temperature=config["temperature"],
                system_prompt=config["system_prompt"],
                guardrails=config["guardrails"],
                max_tokens=config["max_tokens"],
                top_p=config["top_p"],
                presence_penalty=config["presence_penalty"],
                frequency_penalty=config["frequency_penalty"],
                top_k=config["top_k"],
                vector_store_manager=st.session_state.vector_store_manager,
                # Parâmetros de memória
                memory_type=config["memory_type"],
                memory_max_messages=config["memory_max_messages"],
                memory_storage_path=config["memory_storage_path"],
                memory_session_id=config["memory_session_id"]
            )
            if agent:
                st.session_state.agent = agent
                st.session_state.current_agent_name = config["agent_name"]
                st.session_state.current_model = config["model"]  # Atualiza o modelo atual
                # Salva a configuração no chat ativo
                sync_active_chat(config)
                st.toast(f"Agente {config['agent_name']} ({config['model']}) ativado!", icon="✅")

    agent = st.session_state.agent

    # -----------------------------------------------------------------
    # STATUS DO AGENTE
    # -----------------------------------------------------------------
    if agent:
        # Status básico
        status_msg = f"Agente: **{config['agent_name']}** | Modelo: **{config['model']}**"

        # Adiciona status do RAG se habilitado
        if st.session_state.vector_store_manager is not None:
            num_docs = len(st.session_state.rag_documents)
            status_msg += f" | 📚 RAG: **{num_docs} doc(s)**"

        st.success(status_msg)
        display_agent_info(agent)
    else:
        st.warning(
            f"⚠️ Configure a API Key para usar o {config['agent_name']}. "
            f"Veja a sidebar à esquerda."
        )

    st.divider()

    # -----------------------------------------------------------------
    # HISTÓRICO DE MENSAGENS
    # -----------------------------------------------------------------
    # Se não há mensagens, mostra a mensagem de boas-vindas
    if not st.session_state.messages and config["welcome_message"].strip():
        with st.chat_message("assistant"):
            st.markdown(config["welcome_message"])

    # Exibe todas as mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------------------------------------------
    # INPUT DO USUÁRIO
    # -----------------------------------------------------------------
    if prompt := st.chat_input("Digite sua mensagem aqui..."):
        # Verifica se o agente está configurado
        if not agent:
            st.error("❌ Configure a API Key primeiro!")
            return

        # Adiciona mensagem do usuário ao histórico
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Exibe mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa e exibe resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("🤔 Pensando..."):
                response = agent.process_message(prompt)
                st.markdown(response)

        # Adiciona resposta ao histórico
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Sincroniza o chat ativo com a configuração atual
        sync_active_chat(config)


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    main()
