"""
=============================================================================
API - FastAPI para Exposição dos Agentes
=============================================================================

Esta API REST permite que outros aplicativos utilizem os agentes de IA
desenvolvidos neste projeto.

CARACTERÍSTICAS:
- Descoberta dinâmica de agentes
- Sessões de chat persistentes
- Streaming de respostas
- Documentação automática (Swagger/OpenAPI)
- Autenticação via API Key
- Rate limiting
- CORS configurável

ENDPOINTS PRINCIPAIS:
- GET  /agents                  - Lista todos os agentes disponíveis
- POST /agents/{agent}/chat     - Envia mensagem para um agente
- GET  /agents/{agent}/info     - Informações detalhadas do agente
- GET  /sessions                - Lista sessões ativas
- DELETE /sessions/{session_id} - Encerra uma sessão

COMO EXECUTAR:
    uvicorn api:app --reload --port 8000

DOCUMENTAÇÃO:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)

Autor: Curso Master GenAI
Data: 2026
=============================================================================
"""

import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


# =============================================================================
# REGISTRO DINÂMICO DE AGENTES
# =============================================================================

class AgentRegistry:
    """
    Registro central de agentes disponíveis.

    Esta classe descobre automaticamente os agentes disponíveis e
    mantém um registro de suas configurações.
    """

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Any] = {}
        self._discover_agents()

    def _discover_agents(self):
        """
        Descobre automaticamente os agentes disponíveis.

        Esta função importa os agentes do módulo agents e registra
        aqueles que estão disponíveis para uso via API.
        """
        # Agentes base
        try:
            from agents import OpenAIAgent, GeminiAgent, SimpleAgent

            self._agents["openai"] = {
                "class": OpenAIAgent,
                "name": "OpenAI Agent",
                "description": "Agente com GPT-4 e ferramentas (calculadora, busca web, RAG)",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": True,
                "has_rag": True
            }

            self._agents["gemini"] = {
                "class": GeminiAgent,
                "name": "Gemini Agent",
                "description": "Agente com Google Gemini e ferramentas",
                "provider": "google",
                "requires_api_key": "GOOGLE_API_KEY",
                "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"],
                "default_model": "gemini-2.0-flash",
                "has_tools": True,
                "has_rag": True
            }

            self._agents["simple-openai"] = {
                "class": SimpleAgent,
                "name": "Simple Agent (OpenAI)",
                "description": "Agente simples sem ferramentas - apenas conversação",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": False,
                "has_rag": False,
                "extra_params": {"provider": "openai"}
            }

            self._agents["simple-gemini"] = {
                "class": SimpleAgent,
                "name": "Simple Agent (Gemini)",
                "description": "Agente simples sem ferramentas - apenas conversação",
                "provider": "google",
                "requires_api_key": "GOOGLE_API_KEY",
                "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"],
                "default_model": "gemini-2.0-flash",
                "has_tools": False,
                "has_rag": False,
                "extra_params": {"provider": "google"}
            }
        except ImportError as e:
            print(f"⚠️ Erro ao importar agentes base: {e}")

        # Agentes especialistas
        try:
            from agents import FinanceAgent

            self._agents["finance-openai"] = {
                "class": FinanceAgent,
                "name": "Finance Agent (OpenAI)",
                "description": "Especialista em finanças: ações, criptomoedas e câmbio",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": True,
                "has_rag": False,
                "specialization": "finance",
                "extra_params": {"provider": "openai"}
            }

            self._agents["finance-gemini"] = {
                "class": FinanceAgent,
                "name": "Finance Agent (Gemini)",
                "description": "Especialista em finanças: ações, criptomoedas e câmbio",
                "provider": "google",
                "requires_api_key": "GOOGLE_API_KEY",
                "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"],
                "default_model": "gemini-2.0-flash",
                "has_tools": True,
                "has_rag": False,
                "specialization": "finance",
                "extra_params": {"provider": "google"}
            }
        except ImportError as e:
            print(f"⚠️ Erro ao importar FinanceAgent: {e}")

        try:
            from agents import KnowledgeAgent

            self._agents["knowledge-openai"] = {
                "class": KnowledgeAgent,
                "name": "Knowledge Agent (OpenAI)",
                "description": "Especialista em conhecimento: Wikipedia e informações enciclopédicas",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": True,
                "has_rag": False,
                "specialization": "knowledge",
                "extra_params": {"provider": "openai"}
            }

            self._agents["knowledge-gemini"] = {
                "class": KnowledgeAgent,
                "name": "Knowledge Agent (Gemini)",
                "description": "Especialista em conhecimento: Wikipedia e informações enciclopédicas",
                "provider": "google",
                "requires_api_key": "GOOGLE_API_KEY",
                "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"],
                "default_model": "gemini-2.0-flash",
                "has_tools": True,
                "has_rag": False,
                "specialization": "knowledge",
                "extra_params": {"provider": "google"}
            }
        except ImportError as e:
            print(f"⚠️ Erro ao importar KnowledgeAgent: {e}")

        try:
            from agents import WebSearchAgent

            self._agents["websearch-openai"] = {
                "class": WebSearchAgent,
                "name": "Web Search Agent (OpenAI)",
                "description": "Especialista em pesquisa web usando DuckDuckGo",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": True,
                "has_rag": False,
                "specialization": "websearch",
                "extra_params": {"provider": "openai"}
            }

            self._agents["websearch-gemini"] = {
                "class": WebSearchAgent,
                "name": "Web Search Agent (Gemini)",
                "description": "Especialista em pesquisa web usando DuckDuckGo",
                "provider": "google",
                "requires_api_key": "GOOGLE_API_KEY",
                "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro"],
                "default_model": "gemini-2.0-flash",
                "has_tools": True,
                "has_rag": False,
                "specialization": "websearch",
                "extra_params": {"provider": "google"}
            }
        except ImportError as e:
            print(f"⚠️ Erro ao importar WebSearchAgent: {e}")

        # MCP Agents
        try:
            from agents import MCPAgent, MCPAgentDemo
            from agents.mcp_agent import MCP_AVAILABLE, MCP_SERVERS

            # Demo sempre disponível
            self._agents["mcp-demo"] = {
                "class": MCPAgentDemo,
                "name": "MCP Demo Agent",
                "description": "Demonstração do Model Context Protocol (sem conexão real)",
                "provider": "openai",
                "requires_api_key": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                "default_model": "gpt-4o-mini",
                "has_tools": False,
                "has_rag": False,
                "specialization": "mcp",
                "extra_params": {"provider": "openai", "mcp_server_name": "fetch"}
            }

            # MCP Real (se disponível)
            if MCP_AVAILABLE and MCPAgent is not None:
                for server_name, server_info in MCP_SERVERS.items():
                    # Só adiciona se não precisa de API key
                    if not server_info.get("env_required"):
                        self._agents[f"mcp-{server_name}"] = {
                            "class": MCPAgent,
                            "name": f"MCP {server_info['name']} Agent",
                            "description": f"MCP Real: {server_info['description']}",
                            "provider": "openai",
                            "requires_api_key": "OPENAI_API_KEY",
                            "models": ["gpt-4o-mini", "gpt-4o", "gpt-4"],
                            "default_model": "gpt-4o-mini",
                            "has_tools": True,
                            "has_rag": False,
                            "specialization": "mcp",
                            "mcp_server": server_name,
                            "extra_params": {"provider": "openai", "mcp_server_name": server_name}
                        }
        except ImportError as e:
            print(f"⚠️ Erro ao importar MCP agents: {e}")

    def list_agents(self) -> List[Dict[str, Any]]:
        """Lista todos os agentes disponíveis."""
        agents_list = []
        for agent_id, config in self._agents.items():
            # Verifica se a API key está configurada
            api_key_env = config.get("requires_api_key", "")
            api_key_available = bool(os.getenv(api_key_env)) if api_key_env else True

            agents_list.append({
                "id": agent_id,
                "name": config["name"],
                "description": config["description"],
                "provider": config["provider"],
                "models": config.get("models", []),
                "default_model": config.get("default_model"),
                "has_tools": config.get("has_tools", False),
                "has_rag": config.get("has_rag", False),
                "specialization": config.get("specialization"),
                "available": api_key_available
            })

        return agents_list

    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retorna a configuração de um agente específico."""
        return self._agents.get(agent_id)

    def create_agent_instance(
        self,
        agent_id: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Cria uma instância de um agente.

        Args:
            agent_id: ID do agente
            model: Modelo a usar (usa default se None)
            temperature: Temperatura do modelo
            system_prompt: Prompt de sistema customizado
            **kwargs: Parâmetros adicionais

        Returns:
            Instância do agente
        """
        config = self.get_agent_config(agent_id)
        if not config:
            raise ValueError(f"Agente '{agent_id}' não encontrado")

        agent_class = config["class"]

        # Parâmetros base
        params = {
            "model": model or config.get("default_model"),
            "temperature": temperature,
        }

        if system_prompt:
            params["system_prompt"] = system_prompt

        # Adiciona parâmetros extras do config
        extra_params = config.get("extra_params", {})
        params.update(extra_params)

        # Adiciona kwargs adicionais
        params.update(kwargs)

        return agent_class(**params)

    def create_session(self, agent_id: str, **agent_params) -> str:
        """
        Cria uma nova sessão de chat.

        Returns:
            ID da sessão criada
        """
        session_id = str(uuid.uuid4())
        agent = self.create_agent_instance(agent_id, **agent_params)

        self._sessions[session_id] = {
            "id": session_id,
            "agent_id": agent_id,
            "agent": agent,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retorna uma sessão existente."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Remove uma sessão."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lista todas as sessões ativas."""
        return [
            {
                "id": s["id"],
                "agent_id": s["agent_id"],
                "created_at": s["created_at"],
                "last_activity": s["last_activity"],
                "message_count": s["message_count"]
            }
            for s in self._sessions.values()
        ]


# Instância global do registro
agent_registry = AgentRegistry()


# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class ChatMessage(BaseModel):
    """Mensagem de chat."""
    message: str = Field(..., description="Mensagem do usuário", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Olá, como você pode me ajudar?"},
                {"message": "Qual o preço do Bitcoin?"}
            ]
        }
    }


class ChatResponse(BaseModel):
    """Resposta do chat."""
    response: str = Field(..., description="Resposta do agente")
    session_id: str = Field(..., description="ID da sessão")
    agent_id: str = Field(..., description="ID do agente")
    model: str = Field(..., description="Modelo utilizado")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")


class SessionCreate(BaseModel):
    """Criação de sessão."""
    agent_id: str = Field(..., description="ID do agente a usar")
    model: Optional[str] = Field(None, description="Modelo específico (usa default se não informado)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperatura do modelo")
    system_prompt: Optional[str] = Field(None, description="Prompt de sistema customizado")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "agent_id": "finance-openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.5
                }
            ]
        }
    }


class SessionResponse(BaseModel):
    """Resposta de criação de sessão."""
    session_id: str
    agent_id: str
    model: str
    created_at: str


class AgentInfo(BaseModel):
    """Informações de um agente."""
    id: str
    name: str
    description: str
    provider: str
    models: List[str]
    default_model: str
    has_tools: bool
    has_rag: bool
    specialization: Optional[str]
    available: bool


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# AUTENTICAÇÃO
# =============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verifica a API key do cliente.

    Se API_AUTH_REQUIRED=true, exige X-API-Key header.
    A chave deve corresponder a API_AUTH_KEY no .env
    """
    auth_required = os.getenv("API_AUTH_REQUIRED", "false").lower() == "true"

    if not auth_required:
        return True

    expected_key = os.getenv("API_AUTH_KEY")
    if not expected_key:
        # Se auth é requerido mas não há chave configurada, permite acesso
        return True

    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida ou não fornecida. Use o header X-API-Key."
        )

    return True


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # Startup
    print("🚀 API iniciada!")
    print(f"📋 {len(agent_registry.list_agents())} agentes disponíveis")
    yield
    # Shutdown
    print("👋 API encerrada!")


app = FastAPI(
    title="GenAI Agents API",
    description="""
## 🤖 API de Agentes de IA

Esta API permite integrar agentes de IA em suas aplicações.

### Agentes Disponíveis

- **OpenAI/Gemini**: Agentes genéricos com ferramentas
- **Finance**: Especialista em ações, crypto e câmbio
- **Knowledge**: Especialista em Wikipedia
- **Web Search**: Especialista em pesquisa web
- **MCP**: Model Context Protocol

### Como Usar

1. Liste os agentes disponíveis: `GET /agents`
2. Crie uma sessão: `POST /sessions`
3. Envie mensagens: `POST /chat/{session_id}`

### Autenticação

Se configurado, use o header `X-API-Key` com sua chave de API.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": "GenAI Agents API",
        "version": "1.0.0",
        "docs": "/docs",
        "demo": "/demo",
        "agents_count": len(agent_registry.list_agents()),
        "endpoints": {
            "agents": "/agents",
            "sessions": "/sessions",
            "chat": "/chat/{session_id}",
            "chat_stream": "/chat/{session_id}/stream",
            "quick_chat": "/chat/quick/{agent_id}",
            "quick_chat_stream": "/chat/quick/{agent_id}/stream",
            "health": "/health",
            "demo": "/demo"
        },
        "features": {
            "streaming": "SSE (Server-Sent Events)",
            "auth": "Optional API Key",
            "cors": "Configurable"
        }
    }


@app.get("/demo", tags=["Demo"], response_class=HTMLResponse)
async def demo_page():
    """
    Página de demonstração do chat com SSE.

    Acesse http://localhost:8000/demo para ver o chat em ação!
    """
    try:
        demo_path = os.path.join(os.path.dirname(__file__), "static", "chat_sse_demo.html")
        with open(demo_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
            <head><title>Demo não encontrado</title></head>
            <body style="font-family: sans-serif; padding: 50px; text-align: center;">
                <h1>❌ Demo não encontrado</h1>
                <p>O arquivo static/chat_sse_demo.html não foi encontrado.</p>
                <p><a href="/docs">Ir para a documentação da API</a></p>
            </body>
            </html>
            """,
            status_code=404
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Verifica a saúde da API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_available": len(agent_registry.list_agents()),
        "active_sessions": len(agent_registry.list_sessions()),
        "features": {
            "streaming_sse": True,
            "sessions": True,
            "quick_chat": True
        }
    }


# ----- AGENTS -----

@app.get(
    "/agents",
    response_model=List[AgentInfo],
    tags=["Agents"],
    summary="Lista todos os agentes disponíveis"
)
async def list_agents(
    available_only: bool = Query(False, description="Mostrar apenas agentes com API key configurada"),
    _: bool = Depends(verify_api_key)
):
    """
    Retorna a lista de todos os agentes disponíveis.

    Cada agente tem informações sobre:
    - Provider (OpenAI ou Google)
    - Modelos suportados
    - Se tem ferramentas (tools)
    - Se suporta RAG
    - Especialização (se houver)
    """
    agents = agent_registry.list_agents()

    if available_only:
        agents = [a for a in agents if a["available"]]

    return agents


@app.get(
    "/agents/{agent_id}",
    response_model=AgentInfo,
    tags=["Agents"],
    summary="Informações de um agente específico"
)
async def get_agent_info(
    agent_id: str,
    _: bool = Depends(verify_api_key)
):
    """Retorna informações detalhadas de um agente específico."""
    agents = agent_registry.list_agents()
    agent = next((a for a in agents if a["id"] == agent_id), None)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' não encontrado")

    return agent


# ----- SESSIONS -----

@app.post(
    "/sessions",
    response_model=SessionResponse,
    tags=["Sessions"],
    summary="Cria uma nova sessão de chat"
)
async def create_session(
    session_data: SessionCreate,
    _: bool = Depends(verify_api_key)
):
    """
    Cria uma nova sessão de chat com um agente.

    A sessão mantém o histórico da conversa e permite
    múltiplas mensagens com o mesmo contexto.
    """
    # Verifica se o agente existe
    config = agent_registry.get_agent_config(session_data.agent_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Agente '{session_data.agent_id}' não encontrado"
        )

    # Verifica se a API key está disponível
    api_key_env = config.get("requires_api_key", "")
    if api_key_env and not os.getenv(api_key_env):
        raise HTTPException(
            status_code=400,
            detail=f"API key '{api_key_env}' não configurada no servidor"
        )

    try:
        session_id = agent_registry.create_session(
            agent_id=session_data.agent_id,
            model=session_data.model,
            temperature=session_data.temperature,
            system_prompt=session_data.system_prompt
        )

        session = agent_registry.get_session(session_id)

        return SessionResponse(
            session_id=session_id,
            agent_id=session_data.agent_id,
            model=session_data.model or config.get("default_model", "unknown"),
            created_at=session["created_at"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/sessions",
    tags=["Sessions"],
    summary="Lista todas as sessões ativas"
)
async def list_sessions(_: bool = Depends(verify_api_key)):
    """Retorna todas as sessões de chat ativas."""
    return agent_registry.list_sessions()


@app.get(
    "/sessions/{session_id}",
    tags=["Sessions"],
    summary="Informações de uma sessão"
)
async def get_session(session_id: str, _: bool = Depends(verify_api_key)):
    """Retorna informações de uma sessão específica."""
    session = agent_registry.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    return {
        "id": session["id"],
        "agent_id": session["agent_id"],
        "created_at": session["created_at"],
        "last_activity": session["last_activity"],
        "message_count": session["message_count"]
    }


@app.delete(
    "/sessions/{session_id}",
    tags=["Sessions"],
    summary="Encerra uma sessão"
)
async def delete_session(session_id: str, _: bool = Depends(verify_api_key)):
    """Remove uma sessão de chat."""
    if agent_registry.delete_session(session_id):
        return {"message": "Sessão encerrada com sucesso"}

    raise HTTPException(status_code=404, detail="Sessão não encontrada")


# ----- CHAT -----

@app.post(
    "/chat/{session_id}",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Envia mensagem para uma sessão existente"
)
async def chat_with_session(
    session_id: str,
    chat_message: ChatMessage,
    _: bool = Depends(verify_api_key)
):
    """
    Envia uma mensagem para um agente em uma sessão existente.

    O agente processará a mensagem e retornará uma resposta.
    O histórico da conversa é mantido na sessão.
    """
    session = agent_registry.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    agent = session["agent"]

    try:
        start_time = datetime.now()

        # Processa a mensagem
        response = agent.process_message(chat_message.message)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000

        # Atualiza estatísticas da sessão
        session["last_activity"] = datetime.now().isoformat()
        session["message_count"] += 1

        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_id=session["agent_id"],
            model=agent.model if hasattr(agent, 'model') else "unknown",
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/chat/quick/{agent_id}",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat rápido sem sessão"
)
async def quick_chat(
    agent_id: str,
    chat_message: ChatMessage,
    model: Optional[str] = Query(None, description="Modelo a usar"),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
    _: bool = Depends(verify_api_key)
):
    """
    Envia uma mensagem rápida para um agente sem criar sessão.

    ⚠️ Não mantém histórico de conversa.
    Use sessões para conversas com múltiplas mensagens.
    """
    config = agent_registry.get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' não encontrado")

    try:
        agent = agent_registry.create_agent_instance(
            agent_id=agent_id,
            model=model,
            temperature=temperature
        )

        start_time = datetime.now()
        response = agent.process_message(chat_message.message)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000

        return ChatResponse(
            response=response,
            session_id="quick-chat",
            agent_id=agent_id,
            model=model or config.get("default_model", "unknown"),
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SSE (Server-Sent Events) - STREAMING
# =============================================================================

class StreamingChatMessage(BaseModel):
    """Mensagem para chat com streaming."""
    message: str = Field(..., description="Mensagem do usuário", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Explique o que é inteligência artificial em detalhes"}
            ]
        }
    }


async def create_streaming_response(
    agent,
    message: str,
    session_id: str,
    agent_id: str
) -> AsyncGenerator[str, None]:
    """
    Gera respostas em streaming usando SSE.

    Formato SSE:
    - data: {"type": "start", ...}
    - data: {"type": "token", "content": "..."}
    - data: {"type": "end", ...}
    - data: [DONE]
    """
    start_time = datetime.now()

    # Calcula tokens de entrada (estimativa ~4 chars por token)
    input_char_count = len(message)
    # Adiciona system prompt se existir
    if hasattr(agent, 'system_prompt') and agent.system_prompt:
        input_char_count += len(agent.system_prompt)
    tokens_in = input_char_count // 4

    # Evento de início com tokens de entrada
    yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'agent_id': agent_id, 'timestamp': start_time.isoformat(), 'tokens_in': tokens_in})}\n\n"

    try:
        # Verifica se o agente suporta streaming nativo
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'stream'):
            # Streaming nativo do LLM
            full_response = ""

            # Prepara as mensagens
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []
            if hasattr(agent, 'system_prompt') and agent.system_prompt:
                messages.append(SystemMessage(content=agent.system_prompt))

            # Adiciona histórico se disponível
            if hasattr(agent, 'chat_history'):
                messages.extend(agent.chat_history)

            messages.append(HumanMessage(content=message))

            # Stream tokens
            async for chunk in agent.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)  # Permite que o event loop processe

            # Atualiza o histórico do agente
            if hasattr(agent, 'add_to_history'):
                agent.add_to_history(message, full_response)

            response_text = full_response
        else:
            # Fallback: Processa normalmente e simula streaming
            full_response = agent.process_message(message)

            # Simula streaming dividindo em chunks
            words = full_response.split(' ')
            response_text = ""

            for i, word in enumerate(words):
                token = word + (' ' if i < len(words) - 1 else '')
                response_text += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.02)  # Pequeno delay para efeito visual

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000

        # Calcula tokens de saída (estimativa ~4 chars por token)
        output_char_count = len(response_text)
        tokens_out = output_char_count // 4

        # Evento de fim com estatísticas completas
        yield f"data: {json.dumps({'type': 'end', 'processing_time_ms': processing_time, 'tokens_in': tokens_in, 'tokens_out': tokens_out, 'total_tokens': tokens_in + tokens_out})}\n\n"

    except Exception as e:
        # Evento de erro
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    # Sinal de fim do stream (padrão OpenAI)
    yield "data: [DONE]\n\n"


@app.post(
    "/chat/{session_id}/stream",
    tags=["Chat Streaming"],
    summary="Chat com streaming SSE (sessão existente)",
    response_class=StreamingResponse
)
async def chat_stream_session(
    session_id: str,
    chat_message: StreamingChatMessage,
    _: bool = Depends(verify_api_key)
):
    """
    Envia mensagem com resposta em streaming via SSE.

    ## Como usar com JavaScript:

    ```javascript
    const eventSource = new EventSource('/chat/session-id/stream');

    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();
            return;
        }
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            // Adiciona token à resposta
            console.log(data.content);
        }
    };
    ```

    ## Eventos SSE:

    - `start`: Início do streaming
    - `token`: Token/chunk da resposta
    - `end`: Fim do streaming com estatísticas
    - `error`: Erro durante processamento
    - `[DONE]`: Sinal de fim (padrão OpenAI)
    """
    session = agent_registry.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    agent = session["agent"]

    # Atualiza estatísticas
    session["last_activity"] = datetime.now().isoformat()
    session["message_count"] += 1

    return StreamingResponse(
        create_streaming_response(
            agent=agent,
            message=chat_message.message,
            session_id=session_id,
            agent_id=session["agent_id"]
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Desabilita buffering no nginx
        }
    )


@app.post(
    "/chat/quick/{agent_id}/stream",
    tags=["Chat Streaming"],
    summary="Chat rápido com streaming SSE (sem sessão)",
    response_class=StreamingResponse
)
async def quick_chat_stream(
    agent_id: str,
    chat_message: StreamingChatMessage,
    model: Optional[str] = Query(None, description="Modelo a usar"),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
    _: bool = Depends(verify_api_key)
):
    """
    Chat rápido com streaming sem criar sessão.

    ⚠️ Não mantém histórico. Use sessões para conversas longas.

    ## Formato da resposta (SSE):

    ```
    data: {"type": "start", "agent_id": "...", "timestamp": "..."}
    data: {"type": "token", "content": "Olá"}
    data: {"type": "token", "content": ", "}
    data: {"type": "token", "content": "como"}
    ...
    data: {"type": "end", "processing_time_ms": 1234}
    data: [DONE]
    ```
    """
    config = agent_registry.get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' não encontrado")

    try:
        agent = agent_registry.create_agent_instance(
            agent_id=agent_id,
            model=model,
            temperature=temperature
        )

        return StreamingResponse(
            create_streaming_response(
                agent=agent,
                message=chat_message.message,
                session_id="quick-stream",
                agent_id=agent_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/chat/test-stream",
    tags=["Chat Streaming"],
    summary="Teste de streaming SSE",
    response_class=StreamingResponse
)
async def test_stream():
    """
    Endpoint de teste para verificar se SSE está funcionando.

    Envia 10 mensagens de teste com intervalo de 500ms.
    """
    async def generate_test():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Iniciando teste de streaming...'})}\n\n"

        for i in range(10):
            yield f"data: {json.dumps({'type': 'token', 'content': f'Token {i+1} ', 'index': i+1})}\n\n"
            await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'type': 'end', 'message': 'Teste concluído!'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_test(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ----- TOOLS INFO -----

@app.get(
    "/tools",
    tags=["Tools"],
    summary="Lista todas as ferramentas disponíveis"
)
async def list_tools(_: bool = Depends(verify_api_key)):
    """Retorna informações sobre as ferramentas (tools) disponíveis."""
    try:
        from tools import (
            calculator_tool,
            get_current_datetime,
            web_search_tool,
            geocode_address_tool,
            reverse_geocode_tool,
            crypto_price_tool,
            top_cryptos_tool,
            stock_quote_tool,
            forex_rate_tool,
            wikipedia_summary_tool,
            wikipedia_search_tool
        )

        tools_list = [
            {"name": "calculator", "description": "Cálculos matemáticos"},
            {"name": "get_current_datetime", "description": "Data e hora atual"},
            {"name": "web_search", "description": "Busca na web (DuckDuckGo)"},
            {"name": "geocode_address", "description": "Converte endereço em coordenadas"},
            {"name": "reverse_geocode", "description": "Converte coordenadas em endereço"},
            {"name": "crypto_price", "description": "Preço de criptomoedas"},
            {"name": "top_cryptos", "description": "Ranking de criptomoedas"},
            {"name": "stock_quote", "description": "Cotação de ações"},
            {"name": "forex_rate", "description": "Taxa de câmbio"},
            {"name": "wikipedia_summary", "description": "Resumo de artigo da Wikipedia"},
            {"name": "wikipedia_search", "description": "Pesquisa na Wikipedia"}
        ]

        return {"tools": tools_list, "count": len(tools_list)}

    except ImportError as e:
        return {"error": f"Erro ao carregar tools: {e}", "tools": [], "count": 0}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    print(f"🚀 Iniciando API em http://{host}:{port}")
    print(f"📚 Documentação em http://{host}:{port}/docs")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True
    )

