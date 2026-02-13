"""
=============================================================================
VECTOR STORE - Armazenamento Vetorial para RAG
=============================================================================

O que são Embeddings?
- São representações numéricas (vetores) de texto
- Textos com significado similar têm vetores próximos
- Permite busca por SIMILARIDADE SEMÂNTICA

Exemplo:
- "cachorro" e "cão" terão vetores muito próximos
- "cachorro" e "avião" terão vetores distantes

Vector Store (Banco de Vetores):
- Armazena os embeddings dos documentos
- Permite busca rápida por similaridade
- Exemplos: FAISS, Chroma, Pinecone, Weaviate

Neste exemplo usamos FAISS (gratuito e local)

=============================================================================
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    Gerenciador de Vector Store para RAG.

    Esta classe encapsula a lógica de:
    - Criar embeddings dos documentos
    - Armazenar em um vector store
    - Buscar documentos relevantes

    Attributes:
        vector_store: O banco de vetores (FAISS)
        embeddings: O modelo de embeddings usado
    """

    def __init__(self, embeddings: Optional[Embeddings] = None):
        """
        Inicializa o gerenciador.

        Args:
            embeddings: Modelo de embeddings a usar.
                       Se None, usa OpenAI por padrão.
        """
        self.vector_store = None
        self.embeddings = embeddings

        # Se não foi passado um modelo, tenta criar um
        if self.embeddings is None:
            self._initialize_default_embeddings()

    def _initialize_default_embeddings(self):
        """
        Inicializa o modelo de embeddings padrão.

        Tenta usar OpenAI primeiro, depois alternativas gratuitas.
        """
        # Tenta OpenAI (melhor qualidade)
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings()
                print("✅ Usando OpenAI Embeddings")
                return
            except ImportError:
                pass

        # Tenta HuggingFace (gratuito, local)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✅ Usando HuggingFace Embeddings (local e gratuito)")
            return
        except ImportError:
            pass

        raise ValueError(
            "Nenhum modelo de embeddings disponível. "
            "Instale langchain-openai ou sentence-transformers."
        )

    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Cria o vector store a partir de documentos.

        Este é o processo de INDEXAÇÃO:
        1. Cada documento é convertido em embedding
        2. Os embeddings são armazenados no FAISS

        Args:
            documents: Lista de documentos a indexar

        Example:
            >>> manager = VectorStoreManager()
            >>> manager.create_from_documents(my_docs)
        """
        try:
            from langchain_community.vectorstores import FAISS

            print(f"📊 Indexando {len(documents)} documentos...")

            # Cria o vector store
            # Isso pode demorar dependendo da quantidade de documentos
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            print(f"✅ Vector store criado com sucesso!")

        except ImportError:
            raise ImportError(
                "FAISS não está instalado. "
                "Execute: pip install faiss-cpu"
            )

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Busca documentos similares a uma query.

        Este é o processo de RETRIEVAL (recuperação):
        1. A query é convertida em embedding
        2. Buscamos os k embeddings mais próximos
        3. Retornamos os documentos correspondentes

        Args:
            query: Texto de busca
            k: Número de resultados a retornar

        Returns:
            Lista dos k documentos mais relevantes

        Example:
            >>> results = manager.similarity_search("O que é Python?")
            >>> for doc in results:
            ...     print(doc.page_content[:100])
        """
        if self.vector_store is None:
            raise ValueError("Vector store não foi inicializado. Use create_from_documents primeiro.")

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Busca documentos com score de similaridade.

        Útil para entender o quão relevante cada resultado é.
        Score menor = mais similar.

        Args:
            query: Texto de busca
            k: Número de resultados

        Returns:
            Lista de tuplas (Document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store não inicializado.")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def save(self, path: str) -> None:
        """
        Salva o vector store em disco.

        Útil para não precisar reindexar toda vez.

        Args:
            path: Caminho do diretório para salvar
        """
        if self.vector_store is None:
            raise ValueError("Vector store não inicializado.")

        self.vector_store.save_local(path)
        print(f"💾 Vector store salvo em: {path}")

    def load(self, path: str) -> None:
        """
        Carrega um vector store salvo.

        Args:
            path: Caminho do diretório
        """
        from langchain_community.vectorstores import FAISS

        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 Vector store carregado de: {path}")


def create_simple_knowledge_base(texts: List[str], metadatas: List[dict] = None) -> VectorStoreManager:
    """
    Cria uma base de conhecimento simples a partir de textos.

    Função de conveniência para criar rapidamente uma KB.

    Args:
        texts: Lista de textos a indexar
        metadatas: Metadados opcionais para cada texto

    Returns:
        VectorStoreManager configurado

    Example:
        >>> texts = [
        ...     "Python é uma linguagem de programação",
        ...     "LangChain é um framework para LLMs"
        ... ]
        >>> kb = create_simple_knowledge_base(texts)
        >>> results = kb.similarity_search("O que é Python?")
    """
    # Cria documentos a partir dos textos
    if metadatas is None:
        metadatas = [{"source": f"text_{i}"} for i in range(len(texts))]

    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(texts, metadatas)
    ]

    # Cria e retorna o manager
    manager = VectorStoreManager()
    manager.create_from_documents(documents)

    return manager


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Exemplo: criar uma base de conhecimento simples

    sample_texts = [
        """
        Python é uma linguagem de programação de alto nível, interpretada 
        e de propósito geral. Foi criada por Guido van Rossum e lançada 
        em 1991. Python enfatiza legibilidade de código.
        """,
        """
        LangChain é um framework para desenvolvimento de aplicações 
        alimentadas por modelos de linguagem. Permite criar agentes, 
        chains, e integrar com diversas fontes de dados.
        """,
        """
        RAG (Retrieval Augmented Generation) é uma técnica que combina 
        busca de informação com geração de texto. Permite que LLMs 
        respondam perguntas usando documentos específicos como contexto.
        """,
        """
        FAISS (Facebook AI Similarity Search) é uma biblioteca para 
        busca eficiente de vetores similares. É muito usada em 
        aplicações de RAG para armazenar embeddings.
        """
    ]

    print("🚀 Criando base de conhecimento de exemplo...")

    try:
        kb = create_simple_knowledge_base(sample_texts)

        # Teste de busca
        query = "O que é RAG?"
        print(f"\n🔍 Buscando: '{query}'")

        results = kb.similarity_search(query, k=2)

        for i, doc in enumerate(results, 1):
            print(f"\n📄 Resultado {i}:")
            print(doc.page_content.strip()[:200] + "...")

    except Exception as e:
        print(f"❌ Erro: {e}")
        print("💡 Certifique-se de ter configurado OPENAI_API_KEY ou instalado sentence-transformers")

