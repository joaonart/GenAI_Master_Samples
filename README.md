# 🤖 GenAI Master Samples

> **Projeto educacional completo** para aprender a criar **Agentes de IA** com LangChain, FastAPI e Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Índice

- [🎯 Sobre o Projeto](#-sobre-o-projeto)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🤖 Agentes Disponíveis](#-agentes-disponíveis)
- [🔧 Tools (Ferramentas)](#-tools-ferramentas)
- [🌐 API REST](#-api-rest)
- [🎮 Demo Interativo](#-demo-interativo)
- [📚 Conceitos Importantes](#-conceitos-importantes)
- [🛠️ Criando Seus Próprios Componentes](#️-criando-seus-próprios-componentes)
- [🔑 Configuração](#-configuração)
- [📖 Exemplos de Uso](#-exemplos-de-uso)
- [🤝 Contribuindo](#-contribuindo)

---

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido para ensinar os conceitos fundamentais de **Agentes de IA**:

| Conceito | O que você vai aprender |
|----------|------------------------|
| 🤖 **Agentes** | Programas que usam LLMs para "pensar" e agir autonomamente |
| 🔧 **Tools** | Como permitir que o agente execute ações reais (cálculos, buscas, APIs) |
| 📚 **RAG** | Como dar conhecimento específico ao agente com documentos |
| 🧠 **Memória** | Como manter contexto entre conversas (curto e longo prazo) |
| 🔌 **MCP** | Model Context Protocol para conectar a servidores externos |
| 🌐 **API** | Como expor agentes via REST API com streaming |

---

## ✨ Features

### 🖥️ Interfaces
- ✅ **Streamlit App** - Interface completa estilo ChatGPT
- ✅ **API REST** - FastAPI com documentação automática
- ✅ **Demo Web** - Chat interativo com SSE streaming
- ✅ **3 Temas** - Default, ChatGPT e Gemini

### 🤖 Agentes
- ✅ **OpenAI** - GPT-4, GPT-4o, GPT-4o-mini
- ✅ **Google Gemini** - Gemini 2.5 Flash, 2.0 Flash, 1.5 Pro
- ✅ **Especializados** - Finance, Knowledge, Web Search
- ✅ **MCP** - Conexão com servidores externos

### 🔧 Tools
- ✅ Calculadora, Data/Hora, Busca Web
- ✅ Geocoding, Criptomoedas, Ações/Forex
- ✅ Wikipedia, RAG Search

### 📚 RAG
- ✅ Upload de PDF, DOCX, CSV, TXT, MD, JSON
- ✅ Vector Store com FAISS
- ✅ Chunking configurável

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Poetry (recomendado) ou pip
- API Key da OpenAI e/ou Google

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/seu-usuario/GenAI_Master_Samples.git
cd GenAI_Master_Samples
```

### 2️⃣ Instale as dependências

```bash
# Com Poetry (recomendado)
poetry install

# Ou com pip
pip install -r requirements.txt
```

### 3️⃣ Configure as API Keys

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite e adicione suas chaves
nano .env
```

```env
# .env
OPENAI_API_KEY=sk-sua-chave-aqui
GOOGLE_API_KEY=sua-chave-aqui
```

### 4️⃣ Execute!

```bash
# Usando Makefile (recomendado)
make dev          # Inicia API + Streamlit

# Ou manualmente
make api          # Apenas API (porta 8000)
make app          # Apenas Streamlit (porta 8501)
```

### 5️⃣ Acesse

| Interface | URL |
|-----------|-----|
| 🎮 **Demo Chat** | http://localhost:8000/demo |
| 📚 **API Docs** | http://localhost:8000/docs |
| 🎨 **Streamlit** | http://localhost:8501 |

---

## 📁 Estrutura do Projeto

```
GenAI_Master_Samples/
│
├── 📄 app.py                    # Interface Streamlit
├── 📄 api.py                    # API REST FastAPI
├── 📄 Makefile                  # Comandos úteis
├── 📄 pyproject.toml            # Configuração Poetry
├── 📄 requirements.txt          # Dependências pip
├── 📄 .env                      # Variáveis de ambiente
│
├── 📁 agents/                   # 🤖 AGENTES DE IA
│   ├── base_agent.py            # Classe base abstrata
│   ├── simple_agent.py          # Agente simples (sem tools)
│   ├── openai_agent.py          # Agente OpenAI completo
│   ├── gemini_agent.py          # Agente Gemini completo
│   ├── finance_agent.py         # 💰 Especialista em finanças
│   ├── knowledge_agent.py       # 📚 Especialista em conhecimento
│   ├── websearch_agent.py       # 🔍 Especialista em pesquisa
│   └── mcp_agent.py             # 🔌 Agente MCP
│
├── 📁 tools/                    # 🔧 FERRAMENTAS
│   ├── calculator.py            # Calculadora matemática
│   ├── datetime_tool.py         # Data e hora
│   ├── web_search.py            # Busca web (DuckDuckGo)
│   ├── rag_tool.py              # Busca no RAG
│   ├── geocoding.py             # Geocoding (Nominatim)
│   ├── crypto.py                # Criptomoedas (CoinGecko)
│   ├── stocks.py                # Ações/Forex (Alpha Vantage)
│   └── wikipedia.py             # Wikipedia API
│
├── 📁 knowledge_base/           # 📚 RAG
│   ├── document_loader.py       # Carregador de documentos
│   └── vector_store.py          # Vector Store (FAISS)
│
├── 📁 core/                     # 🧠 CORE
│   └── memory.py                # Sistema de memória
│
├── 📁 static/                   # 🎨 ARQUIVOS ESTÁTICOS
│   └── chat_sse_demo.html       # Demo interativo
│
└── 📁 logs/                     # 📋 LOGS
    └── .gitkeep
```

---

## 🤖 Agentes Disponíveis

| ID | Nome | Provider | Especialização | Tools |
|----|------|----------|----------------|-------|
| `simple-openai` | Simple Agent | OpenAI | Geral | ❌ |
| `simple-gemini` | Simple Agent | Google | Geral | ❌ |
| `openai` | OpenAI Agent | OpenAI | Geral | ✅ |
| `gemini` | Gemini Agent | Google | Geral | ✅ |
| `finance-openai` | Finance Expert | OpenAI | 💰 Finanças | ✅ |
| `finance-gemini` | Finance Expert | Google | 💰 Finanças | ✅ |
| `knowledge-openai` | Knowledge Expert | OpenAI | 📚 Conhecimento | ✅ |
| `knowledge-gemini` | Knowledge Expert | Google | 📚 Conhecimento | ✅ |
| `websearch-openai` | Web Search Expert | OpenAI | 🔍 Pesquisa | ✅ |
| `websearch-gemini` | Web Search Expert | Google | 🔍 Pesquisa | ✅ |
| `mcp-fetch` | MCP Fetch | OpenAI | 🔌 URLs | MCP |
| `mcp-time` | MCP Time | OpenAI | 🔌 Data/Hora | MCP |

---

## 🔧 Tools (Ferramentas)

### Tools Disponíveis

| Tool | Função | Exemplo de Pergunta |
|------|--------|---------------------|
| 🧮 `calculator` | Cálculos matemáticos | "Quanto é 15% de 230?" |
| 📅 `get_current_datetime` | Data e hora atual | "Que dia é hoje?" |
| 🔍 `web_search` | Busca na web | "Pesquise sobre LangChain" |
| 📚 `knowledge_base_search` | Busca no RAG | "O que diz o documento?" |
| 🌍 `geocode_address` | Endereço → Coordenadas | "Coordenadas da Av. Paulista?" |
| 📍 `reverse_geocode` | Coordenadas → Endereço | "Que lugar é -23.55, -46.63?" |
| 🪙 `crypto_price` | Preço de criptomoeda | "Preço do Bitcoin?" |
| 🏆 `top_cryptos` | Ranking de cryptos | "Top 10 criptomoedas?" |
| 📊 `stock_quote` | Cotação de ações | "Preço da Apple?" |
| 💱 `forex_rate` | Taxa de câmbio | "Cotação do dólar?" |
| 📖 `wikipedia_summary` | Resumo da Wikipedia | "Quem foi Einstein?" |
| 🔎 `wikipedia_search` | Busca na Wikipedia | "Artigos sobre física quântica" |

### APIs Utilizadas (Gratuitas)

| Tool | API | Precisa de Key? |
|------|-----|-----------------|
| Busca Web | DuckDuckGo | ❌ Não |
| Geocoding | Nominatim/OSM | ❌ Não |
| Criptomoedas | CoinGecko | ❌ Não |
| Ações/Forex | Alpha Vantage | ⚠️ Gratuita |
| Wikipedia | Wikipedia API | ❌ Não |

---

## 🌐 API REST

### Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/agents` | Lista agentes disponíveis |
| `GET` | `/agents/{id}` | Detalhes de um agente |
| `POST` | `/sessions` | Cria sessão de chat |
| `GET` | `/sessions` | Lista sessões ativas |
| `POST` | `/chat/{session_id}` | Envia mensagem |
| `POST` | `/chat/{session_id}/stream` | **Chat com streaming** |
| `POST` | `/chat/quick/{agent_id}` | Chat rápido (sem sessão) |
| `POST` | `/chat/quick/{agent_id}/stream` | **Chat rápido com streaming** |
| `GET` | `/tools` | Lista ferramentas |
| `GET` | `/health` | Status da API |
| `GET` | `/demo` | **Página de demonstração** |

### Exemplo: Chat com Streaming (JavaScript)

```javascript
async function chat(message) {
    const response = await fetch('/chat/quick/openai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                
                const parsed = JSON.parse(data);
                if (parsed.type === 'token') {
                    console.log(parsed.content); // Token recebido!
                }
            }
        }
    }
}
```

### Exemplo: Chat com Streaming (Python)

```python
import requests

def chat(message: str):
    response = requests.post(
        "http://localhost:8000/chat/quick/openai/stream",
        json={"message": message},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                
                import json
                parsed = json.loads(data)
                if parsed['type'] == 'token':
                    print(parsed['content'], end='', flush=True)
    print()

chat("Explique o que é machine learning")
```

### Exemplo: Usando cURL

```bash
# Listar agentes
curl http://localhost:8000/agents

# Criar sessão
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "openai"}'

# Chat rápido
curl -X POST http://localhost:8000/chat/quick/openai \
  -H "Content-Type: application/json" \
  -d '{"message": "Olá, tudo bem?"}'
```

---

## 🎮 Demo Interativo

Acesse **http://localhost:8000/demo** para usar o chat interativo com:

### ✨ Features do Demo

- 💬 **Chat em tempo real** com streaming SSE
- 🎨 **3 Temas**: Default, ChatGPT, Gemini
- 📝 **Histórico de conversas** persistente
- 📊 **Contagem de tokens** (input/output)
- ⚙️ **Configurações** de agente, modelo e temperatura
- 📱 **Responsivo** para mobile

### 🎨 Temas Disponíveis

| Tema | Descrição |
|------|-----------|
| 🌙 Default | Tema escuro com gradiente roxo/ciano |
| 💚 ChatGPT | Idêntico ao ChatGPT da OpenAI |
| 💙 Gemini | Idêntico ao Google Gemini |

---

## 📚 Conceitos Importantes

### 🤖 O que é um Agente?

Um **Agente de IA** é um programa que:

```
┌─────────────────────────────────────────────────────────┐
│                      AGENTE DE IA                       │
├─────────────────────────────────────────────────────────┤
│  1. ENTENDE → Analisa a mensagem do usuário            │
│  2. DECIDE  → Escolhe qual ação tomar                  │
│  3. EXECUTA → Usa tools, RAG ou responde diretamente   │
│  4. FORMULA → Gera resposta baseada no resultado       │
└─────────────────────────────────────────────────────────┘
```

### 🔧 O que são Tools?

**Tools** são funções que o agente pode chamar quando necessário:

```python
@tool("calculator")
def calculator(expression: str) -> str:
    """Calcula expressões matemáticas."""
    return str(eval(expression))

# O LLM decide QUANDO usar:
# "Quanto é 10 + 20?" → Usa calculator
# "Olá, tudo bem?"    → Não usa (responde direto)
```

### 📚 O que é RAG?

**RAG** (Retrieval Augmented Generation) dá conhecimento específico ao LLM:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DOCUMENTOS  │ →  │   VETORES    │ →  │    BUSCA     │
│  PDF, DOCX   │    │   FAISS      │    │  Relevantes  │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RESPOSTA   │ ←  │     LLM      │ ←  │   CONTEXTO   │
│   Precisa    │    │  GPT/Gemini  │    │  + Pergunta  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 🧠 Tipos de Memória

| Tipo | Descrição | Persistência |
|------|-----------|--------------|
| **Sem Memória** | Cada mensagem é independente | ❌ |
| **Curto Prazo** | Últimas N mensagens | Sessão |
| **Longo Prazo** | Fatos importantes | Disco |
| **Combinada** | Curto + Longo prazo | Ambos |

---

## 🛠️ Criando Seus Próprios Componentes

### Criando um Novo Agente

```python
# agents/meu_agente.py
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI

class MeuAgente(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Meu Agente",
            description="Um agente personalizado"
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    def process_message(self, message: str) -> str:
        response = self.llm.invoke(message)
        return response.content
```

### Criando uma Nova Tool

```python
# tools/minha_tool.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MinhaToolInput(BaseModel):
    query: str = Field(description="O que buscar")

@tool("minha_tool", args_schema=MinhaToolInput)
def minha_tool(query: str) -> str:
    """
    Descrição da tool para o LLM saber quando usar.
    
    Use quando o usuário perguntar sobre X.
    """
    # Sua lógica aqui
    resultado = fazer_algo(query)
    return resultado
```

### Registrando no Sistema

```python
# api.py - Adicione no agent_registry
agent_registry.register(
    agent_id="meu-agente",
    config={
        "name": "Meu Agente",
        "class": "MeuAgente",
        "provider": "openai",
        # ...
    }
)
```

---

## 🔑 Configuração

### Variáveis de Ambiente

```env
# === LLM APIs (pelo menos uma obrigatória) ===
OPENAI_API_KEY=sk-sua-chave-aqui
GOOGLE_API_KEY=sua-chave-aqui

# === APIs de Tools (opcionais) ===
ALPHA_VANTAGE_API_KEY=sua-chave  # Para ações/forex

# === API Config (opcionais) ===
API_PORT=8000
API_HOST=0.0.0.0
API_AUTH_REQUIRED=false
API_AUTH_KEY=sua-chave-secreta
```

### Onde Obter as API Keys

| API | URL | Custo |
|-----|-----|-------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Pago |
| Google AI | [aistudio.google.com](https://aistudio.google.com/apikey) | Gratuito |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Gratuito |

---

## 📖 Exemplos de Uso

### Exemplo 1: Chat Simples

```python
from agents import SimpleOpenAIAgent

agent = SimpleOpenAIAgent()
response = agent.process_message("Olá, tudo bem?")
print(response)
```

### Exemplo 2: Agente com Tools

```python
from agents import OpenAIAgent

agent = OpenAIAgent()
response = agent.process_message("Quanto é 15% de 350?")
print(response)  # Usa a calculadora automaticamente
```

### Exemplo 3: Agente de Finanças

```python
from agents import FinanceOpenAIAgent

agent = FinanceOpenAIAgent()
response = agent.process_message("Qual o preço do Bitcoin?")
print(response)  # Usa a API CoinGecko
```

### Exemplo 4: Usando a API

```python
import requests

# Criar sessão
session = requests.post("http://localhost:8000/sessions", json={
    "agent_id": "finance-openai"
}).json()

# Enviar mensagens
response = requests.post(
    f"http://localhost:8000/chat/{session['session_id']}",
    json={"message": "Cotação do dólar hoje?"}
).json()

print(response['response'])
```

---

## 🧰 Comandos Úteis (Makefile)

```bash
# Instalação
make install          # Instala dependências
make install-dev      # Instala com deps de desenvolvimento

# Execução
make dev              # Inicia API + Streamlit (desenvolvimento)
make api              # Inicia apenas a API
make app              # Inicia apenas o Streamlit

# Background
make start            # Inicia tudo em background
make stop             # Para todos os serviços
make restart          # Reinicia tudo
make status           # Verifica status

# Logs
make logs             # Mostra logs recentes
make logs-api         # Logs da API em tempo real
make logs-app         # Logs do Streamlit em tempo real

# Qualidade
make test             # Executa testes
make lint             # Verifica código
make format           # Formata código

# Utilidades
make clean            # Limpa arquivos temporários
make check-env        # Verifica variáveis de ambiente
make info             # Informações do projeto
make help             # Lista todos os comandos
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Add MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- [LangChain](https://langchain.com) - Framework de LLM
- [FastAPI](https://fastapi.tiangolo.com) - API Framework
- [Streamlit](https://streamlit.io) - Interface Web
- [OpenAI](https://openai.com) - GPT Models
- [Google AI](https://ai.google.dev) - Gemini Models

---

<div align="center">

**Feito com ❤️ para o Curso Master de GenAI**

🎓 2026

</div>

