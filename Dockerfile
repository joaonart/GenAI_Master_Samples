# ============================================
# GenAI Master Samples - Dockerfile
# ============================================
# Imagem otimizada para produção da API
#
# Build:
#   docker build -t genai-api .
#
# Run:
#   docker run -d -p 8000:8000 --env-file .env genai-api
# ============================================

# Estágio 1: Builder
FROM python:3.11-slim as builder

# Variáveis de ambiente para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instala Poetry
RUN pip install poetry==1.7.1

# Diretório de trabalho
WORKDIR /app

# Copia apenas arquivos de dependências primeiro (cache de layers)
COPY pyproject.toml poetry.lock* ./

# Exporta dependências para requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Estágio 2: Runtime
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="GenAI Master Team" \
      version="1.0.0" \
      description="GenAI Master Samples - API de Agentes de IA"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Configuração da API
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Desabilita telemetria
    ANONYMIZED_TELEMETRY=false

# Cria usuário não-root para segurança
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Diretório de trabalho
WORKDIR /app

# Copia requirements do builder
COPY --from=builder /app/requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY --chown=appuser:appgroup . .

# Cria diretórios necessários
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appgroup /app/logs /app/data

# Muda para usuário não-root
USER appuser

# Expõe a porta da API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão: inicia a API
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

