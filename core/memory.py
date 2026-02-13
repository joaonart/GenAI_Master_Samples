"""
=============================================================================
MEMORY - Sistema de Memória para Agentes de IA
=============================================================================

Este módulo implementa diferentes tipos de memória para os agentes:

1. MEMÓRIA DE CURTO PRAZO (Short-term / Buffer):
   - Armazena as últimas N mensagens da conversa atual
   - Rápida e eficiente
   - Perde informações quando o limite é atingido
   - Ideal para: Conversas simples, contexto imediato

2. MEMÓRIA DE LONGO PRAZO (Long-term / Summary):
   - Resume conversas anteriores
   - Persiste informações importantes
   - Usa o próprio LLM para criar resumos
   - Ideal para: Lembrar preferências, fatos sobre o usuário

3. MEMÓRIA COMBINADA (Combined):
   - Usa ambas: curto prazo para contexto imediato
   - Longo prazo para informações persistentes
   - Melhor dos dois mundos

Analogia:
- Curto prazo = Memória de trabalho (o que você está fazendo agora)
- Longo prazo = Memória episódica (lembranças de eventos passados)

=============================================================================
"""

import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory


class ShortTermMemory(BaseChatMessageHistory):
    """
    Memória de Curto Prazo - Buffer de mensagens recentes.

    Mantém apenas as últimas N mensagens da conversa.
    Quando o limite é atingido, as mensagens mais antigas são removidas.

    Attributes:
        max_messages: Número máximo de mensagens a manter
        messages: Lista de mensagens armazenadas
    """

    def __init__(self, max_messages: int = 20):
        """
        Inicializa a memória de curto prazo.

        Args:
            max_messages: Número máximo de mensagens a manter (padrão: 20)
        """
        self.max_messages = max_messages
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Retorna as mensagens armazenadas."""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """
        Adiciona uma mensagem à memória.

        Se o limite for atingido, remove a mensagem mais antiga.
        """
        self._messages.append(message)

        # Remove mensagens antigas se exceder o limite
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)

    def add_user_message(self, message: str) -> None:
        """Adiciona uma mensagem do usuário."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Adiciona uma mensagem do assistente."""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Limpa todas as mensagens."""
        self._messages = []

    def get_messages_as_text(self) -> str:
        """Retorna as mensagens como texto formatado."""
        lines = []
        for msg in self._messages:
            role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


class LongTermMemory:
    """
    Memória de Longo Prazo - Armazena resumos e fatos importantes.

    Esta memória persiste informações entre sessões, salvando em disco.
    Pode usar o LLM para criar resumos automáticos das conversas.

    Attributes:
        storage_path: Caminho do arquivo de memória
        memories: Lista de memórias armazenadas
        max_memories: Número máximo de memórias a manter
    """

    def __init__(
        self,
        storage_path: str = "./memory_data",
        session_id: str = "default",
        max_memories: int = 100
    ):
        """
        Inicializa a memória de longo prazo.

        Args:
            storage_path: Diretório para salvar as memórias
            session_id: ID da sessão/usuário
            max_memories: Número máximo de memórias a manter
        """
        self.storage_path = Path(storage_path)
        self.session_id = session_id
        self.max_memories = max_memories
        self.memories: List[Dict[str, Any]] = []

        # Cria o diretório se não existir
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Carrega memórias existentes
        self._load()

    def _get_file_path(self) -> Path:
        """Retorna o caminho do arquivo de memória."""
        return self.storage_path / f"{self.session_id}_memory.json"

    def _load(self) -> None:
        """Carrega memórias do disco."""
        file_path = self._get_file_path()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memories = data.get("memories", [])
            except Exception as e:
                print(f"⚠️ Erro ao carregar memória: {e}")
                self.memories = []

    def _save(self) -> None:
        """Salva memórias em disco."""
        file_path = self._get_file_path()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": self.session_id,
                    "last_updated": datetime.now().isoformat(),
                    "memories": self.memories
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar memória: {e}")

    def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Adiciona uma nova memória.

        Args:
            content: Conteúdo da memória
            memory_type: Tipo da memória (fact, preference, summary, event)
            importance: Importância de 1 a 10 (10 = mais importante)
            metadata: Metadados adicionais
        """
        memory = {
            "id": len(self.memories) + 1,
            "content": content,
            "type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.memories.append(memory)

        # Remove memórias antigas se exceder o limite (mantém as mais importantes)
        if len(self.memories) > self.max_memories:
            # Ordena por importância (menor primeiro) e remove os menos importantes
            self.memories.sort(key=lambda x: x.get("importance", 0))
            self.memories = self.memories[-self.max_memories:]

        self._save()

    def add_conversation_summary(self, summary: str) -> None:
        """Adiciona um resumo de conversa."""
        self.add_memory(
            content=summary,
            memory_type="summary",
            importance=7
        )

    def add_user_preference(self, preference: str) -> None:
        """Adiciona uma preferência do usuário."""
        self.add_memory(
            content=preference,
            memory_type="preference",
            importance=8
        )

    def add_important_fact(self, fact: str) -> None:
        """Adiciona um fato importante."""
        self.add_memory(
            content=fact,
            memory_type="fact",
            importance=9
        )

    def get_memories(
        self,
        memory_type: str = None,
        limit: int = 10,
        min_importance: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retorna memórias filtradas.

        Args:
            memory_type: Filtrar por tipo (None = todos)
            limit: Número máximo de memórias a retornar
            min_importance: Importância mínima

        Returns:
            Lista de memórias ordenadas por importância
        """
        filtered = self.memories

        if memory_type:
            filtered = [m for m in filtered if m.get("type") == memory_type]

        if min_importance > 0:
            filtered = [m for m in filtered if m.get("importance", 0) >= min_importance]

        # Ordena por importância (maior primeiro) e timestamp
        filtered.sort(key=lambda x: (-x.get("importance", 0), x.get("timestamp", "")))

        return filtered[:limit]

    def get_memories_as_text(self, limit: int = 10) -> str:
        """Retorna memórias formatadas como texto."""
        memories = self.get_memories(limit=limit)

        if not memories:
            return ""

        lines = ["MEMÓRIAS DE LONGO PRAZO:"]
        for mem in memories:
            mem_type = mem.get("type", "unknown")
            content = mem.get("content", "")
            lines.append(f"- [{mem_type}] {content}")

        return "\n".join(lines)

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Busca memórias que contenham a query.

        Args:
            query: Texto a buscar

        Returns:
            Lista de memórias que contêm a query
        """
        query_lower = query.lower()
        return [
            m for m in self.memories
            if query_lower in m.get("content", "").lower()
        ]

    def clear(self) -> None:
        """Limpa todas as memórias."""
        self.memories = []
        self._save()


class CombinedMemory:
    """
    Memória Combinada - Une curto e longo prazo.

    Usa memória de curto prazo para contexto imediato da conversa
    e memória de longo prazo para informações persistentes.

    Attributes:
        short_term: Memória de curto prazo
        long_term: Memória de longo prazo
    """

    def __init__(
        self,
        max_short_term_messages: int = 20,
        storage_path: str = "./memory_data",
        session_id: str = "default",
        max_long_term_memories: int = 100
    ):
        """
        Inicializa a memória combinada.

        Args:
            max_short_term_messages: Limite de mensagens no curto prazo
            storage_path: Caminho para salvar memórias de longo prazo
            session_id: ID da sessão/usuário
            max_long_term_memories: Limite de memórias de longo prazo
        """
        self.short_term = ShortTermMemory(max_messages=max_short_term_messages)
        self.long_term = LongTermMemory(
            storage_path=storage_path,
            session_id=session_id,
            max_memories=max_long_term_memories
        )

    def add_user_message(self, message: str) -> None:
        """Adiciona mensagem do usuário ao curto prazo."""
        self.short_term.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """Adiciona mensagem do assistente ao curto prazo."""
        self.short_term.add_ai_message(message)

    def add_to_long_term(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 5
    ) -> None:
        """Adiciona uma memória ao longo prazo."""
        self.long_term.add_memory(content, memory_type, importance)

    def get_context(self, include_long_term: bool = True) -> str:
        """
        Retorna o contexto completo para o agente.

        Args:
            include_long_term: Se deve incluir memórias de longo prazo

        Returns:
            Texto com o contexto formatado
        """
        parts = []

        # Adiciona memórias de longo prazo
        if include_long_term:
            long_term_text = self.long_term.get_memories_as_text(limit=5)
            if long_term_text:
                parts.append(long_term_text)

        # Adiciona conversa recente
        short_term_text = self.short_term.get_messages_as_text()
        if short_term_text:
            parts.append("CONVERSA RECENTE:\n" + short_term_text)

        return "\n\n".join(parts)

    def get_short_term_messages(self) -> List[BaseMessage]:
        """Retorna as mensagens do curto prazo."""
        return self.short_term.messages

    def clear_short_term(self) -> None:
        """Limpa apenas a memória de curto prazo."""
        self.short_term.clear()

    def clear_long_term(self) -> None:
        """Limpa apenas a memória de longo prazo."""
        self.long_term.clear()

    def clear_all(self) -> None:
        """Limpa todas as memórias."""
        self.short_term.clear()
        self.long_term.clear()


# =============================================================================
# TIPOS DE MEMÓRIA DISPONÍVEIS
# =============================================================================

MEMORY_TYPES = {
    "none": {
        "name": "Sem Memória",
        "description": "Não mantém histórico entre mensagens",
        "icon": "🚫"
    },
    "short_term": {
        "name": "Curto Prazo",
        "description": "Mantém as últimas N mensagens da conversa",
        "icon": "⏱️"
    },
    "long_term": {
        "name": "Longo Prazo",
        "description": "Persiste informações importantes entre sessões",
        "icon": "💾"
    },
    "combined": {
        "name": "Combinada",
        "description": "Usa curto e longo prazo juntos",
        "icon": "🧠"
    }
}


def get_memory_types() -> Dict[str, Dict[str, str]]:
    """Retorna os tipos de memória disponíveis."""
    return MEMORY_TYPES


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    print("🧠 Testando Sistema de Memória")
    print("=" * 50)

    # Teste memória de curto prazo
    print("\n📝 Memória de Curto Prazo:")
    short_mem = ShortTermMemory(max_messages=5)
    short_mem.add_user_message("Olá!")
    short_mem.add_ai_message("Oi! Como posso ajudar?")
    short_mem.add_user_message("Qual é a capital do Brasil?")
    short_mem.add_ai_message("A capital do Brasil é Brasília.")
    print(short_mem.get_messages_as_text())

    # Teste memória de longo prazo
    print("\n💾 Memória de Longo Prazo:")
    long_mem = LongTermMemory(storage_path="./test_memory", session_id="test")
    long_mem.add_user_preference("O usuário prefere respostas em português")
    long_mem.add_important_fact("O usuário se chama João")
    print(long_mem.get_memories_as_text())

    # Teste memória combinada
    print("\n🧠 Memória Combinada:")
    combined = CombinedMemory(
        max_short_term_messages=10,
        storage_path="./test_memory",
        session_id="test"
    )
    combined.add_user_message("Meu nome é Maria")
    combined.add_ai_message("Prazer em conhecê-la, Maria!")
    combined.add_to_long_term("A usuária se chama Maria", "fact", importance=10)
    print(combined.get_context())

    # Limpa dados de teste
    import shutil
    shutil.rmtree("./test_memory", ignore_errors=True)
    print("\n✅ Testes concluídos!")

