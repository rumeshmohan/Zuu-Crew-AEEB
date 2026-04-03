"""
Benchmarking sub-package for the Prime Lands Real Estate Intelligence Platform.

Exposes retrieval benchmarking and evaluation utilities used in Part 2
(Chunking Lab) and Part 4 (Performance Arena) to measure and compare the five
chunking strategies across Precision, Recall, answer relevance, latency,
and index size metrics.

Modules:
    benchmarking_utils: Core benchmarking functions —
        ``benchmark_retrieval`` and ``benchmark_all_strategies`` for timed
        retrieval runs; ``get_collection_size_mb`` and
        ``calculate_all_index_sizes`` for Qdrant index measurement;
        ``create_comparison_table`` and ``print_comparison_table`` for
        summary reporting; ``analyze_trade_offs`` for qualitative trade-off
        analysis; and ``run_full_evaluation`` to execute the complete
        evaluation pipeline in one call.
"""

from .benchmarking_utils import (
    benchmark_retrieval,
    benchmark_all_strategies,
    get_collection_size_mb,
    calculate_all_index_sizes,
    create_comparison_table,
    print_comparison_table,
    analyze_trade_offs,
    run_full_evaluation,
)

__all__ = [
    # Retrieval benchmarking
    "benchmark_retrieval",
    "benchmark_all_strategies",
    # Index size measurement
    "get_collection_size_mb",
    "calculate_all_index_sizes",
    # Comparison reporting
    "create_comparison_table",
    "print_comparison_table",
    # Trade-off analysis
    "analyze_trade_offs",
    "run_full_evaluation",
]