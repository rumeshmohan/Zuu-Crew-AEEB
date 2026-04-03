from typing import Any, Dict, List
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from context_engineering.config import (
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    TOP_K_RESULTS
)
from context_engineering.domain.prompts.rag_templates import RAG_TEMPLATE
from context_engineering.domain.utils import format_docs, calculate_confidence


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.

    Performs initial retrieval and scores confidence. If the score falls
    below the threshold, a second expanded retrieval is triggered before
    generation.

    Args:
        retriever: Vector store retriever instance.
        llm: LangChain LLM instance.
        initial_k: Number of documents for initial retrieval.
        expanded_k: Number of documents for corrective retrieval.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        self.confidence_threshold = CRAG_CONFIDENCE_THRESHOLD
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def generate(
        self,
        query: str,
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer with corrective retrieval.

        Retrieves initial documents and scores confidence. If confidence
        is below the threshold, repeats retrieval with expanded_k and
        recalculates before generating.

        Args:
            query: User question.
            confidence_threshold: Minimum confidence score (0.0–1.0).
            verbose: If True, prints progress logs to stdout.

        Returns:
            Dict with keys: 'answer', 'confidence_initial', 'confidence_final',
            'correction_applied', 'docs_used', 'generation_time', 'evidence_urls',
            'evidence'.
        """
        if verbose:
            print(f"🔍 Query: {query}")
            print(f"🎯 Confidence threshold: {confidence_threshold}\n")

        if verbose:
            print(f"1️⃣  Initial retrieval (k={self.initial_k})...")

        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)

        if verbose:
            print(f"   📊 Confidence: {confidence_initial:.2f}")

        if confidence_initial >= confidence_threshold:
            if verbose:
                print(f"   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                print(f"   ⚠️  Low confidence - applying corrective retrieval...\n")
                print(f"2️⃣  Corrective retrieval (k={self.expanded_k}, expanded)...")

            self.retriever.search_kwargs["k"] = self.expanded_k
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)

            if verbose:
                print(f"   📊 Corrected confidence: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                print(f"   📈 Confidence improved by {improvement:.1f}%")

            final_docs = docs_corrected  
            correction_applied = True

        if verbose:
            print(f"\n3️⃣  Generating answer...")

        start = time.time()
        context = format_docs(final_docs)
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": query}
        )
        elapsed = time.time() - start

        evidence_urls = list(set([doc.metadata.get('url', '') for doc in final_docs if doc.metadata.get('url')]))

        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'evidence_urls': evidence_urls,
            'evidence': final_docs
        }

    def batch_generate(
        self,
        queries: List[str],
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of user questions.
            confidence_threshold: Minimum confidence score (0.0–1.0).

        Returns:
            List of result dicts in the same format as generate().
        """
        return [
            self.generate(query, confidence_threshold, verbose=False)
            for query in queries
        ]

    def analyze_confidence(self, query: str) -> Dict[str, Any]:
        """
        Analyse retrieval confidence without generating an answer.

        Args:
            query: User question.

        Returns:
            Dict with 'confidence_initial', 'confidence_expanded', 'improvement',
            'docs_initial', and 'docs_expanded'.
        """
        self.retriever.search_kwargs["k"] = self.initial_k
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)

        self.retriever.search_kwargs["k"] = self.expanded_k
        docs_expanded = self.retriever.invoke(query)
        confidence_expanded = calculate_confidence(docs_expanded, query)

        return {
            'query': query,
            'confidence_initial': confidence_initial,
            'confidence_expanded': confidence_expanded,
            'improvement': confidence_expanded - confidence_initial,
            'docs_initial': len(docs_initial),
            'docs_expanded': len(docs_expanded)
        }


__all__ = ['CRAGService']