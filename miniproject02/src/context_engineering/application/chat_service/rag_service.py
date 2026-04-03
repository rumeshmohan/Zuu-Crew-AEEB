from typing import Any, Dict, List
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from context_engineering.config import TOP_K_RESULTS
from context_engineering.domain.prompts.rag_templates import RAG_TEMPLATE
from context_engineering.domain.utils import format_docs


def build_rag_chain(
    retriever: VectorStoreRetriever,
    llm: Any,
    k: int = TOP_K_RESULTS,
    template: str = RAG_TEMPLATE
) -> Runnable:
    """
    Build a RAG chain using LangChain Expression Language (LCEL).

    Args:
        retriever: VectorStore retriever instance.
        llm: LangChain LLM instance.
        k: Number of documents to retrieve.
        template: Prompt template string.

    Returns:
        Runnable LCEL chain invocable with a query string.
    """
    if k != TOP_K_RESULTS:
        retriever.search_kwargs["k"] = k

    rag_prompt = ChatPromptTemplate.from_template(template)

    return (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )


class RAGService:
    """
    High-level RAG service for question answering with evidence tracking.

    Args:
        retriever: Vector store retriever instance.
        llm: LangChain LLM instance.
        k: Number of documents to retrieve.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Any,
        k: int = TOP_K_RESULTS
    ):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.chain = build_rag_chain(retriever, llm, k)

    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer for a query using RAG.

        Args:
            query: User question.

        Returns:
            Dict with keys: 'answer', 'evidence', 'evidence_urls',
            'generation_time', 'num_docs'.
        """
        start = time.time()
        evidence = self.retriever.invoke(query)
        answer = self.chain.invoke(query)
        elapsed = time.time() - start

        return {
            'answer': answer,
            'evidence': evidence,
            'evidence_urls': list(set([doc.metadata['url'] for doc in evidence if 'url' in doc.metadata])),
            'generation_time': elapsed,
            'num_docs': len(evidence)
        }

    def stream(self, query: str):
        """
        Stream answer generation for real-time output.

        Args:
            query: User question.

        Yields:
            String chunks as they are generated.
        """
        for chunk in self.chain.stream(query):
            yield chunk

    def batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of user questions.

        Returns:
            List of result dicts in the same format as generate().
        """
        return [self.generate(query) for query in queries]


__all__ = ['build_rag_chain', 'RAGService']