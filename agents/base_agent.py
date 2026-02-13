"""
=============================================================================
BASE AGENT - Classe Base Abstrata para Todos os Agentes
=============================================================================

Este módulo define a classe base que todos os agentes devem herdar.
Utiliza o padrão de projeto "Template Method" para garantir consistência.

Conceitos importantes:
- ABC (Abstract Base Class): Classe que não pode ser instanciada diretamente
- @abstractmethod: Método que DEVE ser implementado pelas classes filhas
- Chat History: Histórico de mensagens para manter contexto da conversa

Autor: Seu Nome
Data: 2024
=============================================================================
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class BaseAgent(ABC):
    """
    Classe base abstrata para todos os agentes de IA.

    Esta classe define a interface comum que todos os agentes devem seguir.
    Ela gerencia o histórico de conversas e define métodos que devem ser
    implementados por cada agente específico.

    Attributes:
        name (str): Nome do agente (ex: "Assistente", "Tutor")
        description (str): Descrição do que o agente faz
        chat_history (List[BaseMessage]): Lista de mensagens da conversa

    Example:
        >>> class MeuAgente(BaseAgent):
        ...     def process_message(self, message: str) -> str:
        ...         return f"Você disse: {message}"
        ...
        >>> agente = MeuAgente("Eco", "Repete mensagens")
        >>> agente.process_message("Olá!")
        'Você disse: Olá!'
    """

    def __init__(self, name: str, description: str = ""):
        """
        Inicializa o agente base.

        Args:
            name: O nome do agente (será exibido no chat)
            description: Descrição do propósito do agente

        Note:
            O chat_history é inicializado vazio e será preenchido
            conforme as mensagens são trocadas.
        """
        self.name = name
        self.description = description

        # Histórico de mensagens usando tipos do LangChain
        # HumanMessage: mensagens do usuário
        # AIMessage: respostas do agente
        self.chat_history: List[BaseMessage] = []

    @abstractmethod
    def process_message(self, message: str) -> str:
        """
        Processa uma mensagem do usuário e retorna uma resposta.

        Este método DEVE ser implementado por todas as classes filhas.
        É aqui que a "mágica" acontece - cada agente implementa sua
        própria lógica de processamento.

        Args:
            message: A mensagem enviada pelo usuário

        Returns:
            A resposta gerada pelo agente

        Raises:
            NotImplementedError: Se a classe filha não implementar este método
        """
        pass

    def add_to_history(self, human_message: str, ai_response: str) -> None:
        """
        Adiciona uma troca de mensagens ao histórico.

        O histórico é importante para manter o contexto da conversa.
        O modelo de IA usa esse histórico para entender o contexto
        e dar respostas mais relevantes.

        Args:
            human_message: A mensagem enviada pelo usuário
            ai_response: A resposta gerada pelo agente

        Example:
            >>> agente.add_to_history("Olá!", "Olá! Como posso ajudar?")
            >>> len(agente.chat_history)
            2
        """
        # HumanMessage representa mensagens do usuário
        self.chat_history.append(HumanMessage(content=human_message))

        # AIMessage representa respostas do assistente/agente
        self.chat_history.append(AIMessage(content=ai_response))

    def clear_history(self) -> None:
        """
        Limpa o histórico de conversas.

        Útil quando o usuário quer "começar do zero" ou
        quando a conversa fica muito longa e consome muitos tokens.
        """
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retorna o histórico em formato de dicionário.

        Formato compatível com a API do OpenAI/ChatGPT:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            Lista de dicionários com 'role' e 'content'
        """
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
