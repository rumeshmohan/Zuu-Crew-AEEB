import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_qdrant import QdrantVectorStore


def benchmark_retrieval(
    vectorstore: QdrantVectorStore,
    test_query: str = "luxury land in Colombo",
    k: int = 5,
    iterations: int = 10,
) -> float:
    """
    Benchmark retrieval time for a single collection.

    Args:
        vectorstore: QdrantVectorStore instance to test.
        test_query: Query string to test with.
        k: Number of results to retrieve per query.
        iterations: Number of iterations to average over.

    Returns:
        Average retrieval time in milliseconds.
    """
    times = []
    for _ in range(iterations):
        start = time.time()
        vectorstore.similarity_search(test_query, k=k)
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)


def benchmark_all_strategies(
    collections: Dict[str, QdrantVectorStore],
    test_query: str = "luxury land in Colombo with highway access",
    k: int = 5,
    iterations: int = 10,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Benchmark retrieval times across all strategy collections.

    Args:
        collections: Dict mapping strategy name to vectorstore.
        test_query: Query to test with.
        k: Number of results to retrieve per query.
        iterations: Number of iterations per strategy.
        verbose: If True, prints progress to stdout.

    Returns:
        Dict mapping strategy name to average retrieval time in milliseconds.
    """
    if verbose:
        print("Benchmarking retrieval times...")
        print(f"   Query      : '{test_query}'")
        print(f"   Iterations : {iterations} per strategy\n")

    retrieval_times = {}

    for strategy, vectorstore in collections.items():
        try:
            avg_time = benchmark_retrieval(vectorstore, test_query, k, iterations)
            retrieval_times[strategy] = avg_time
            if verbose:
                print(f"   {strategy:<16}: {avg_time:.2f} ms")
        except Exception as e:
            if verbose:
                print(f"   {strategy:<16}: Error - {e}")
            retrieval_times[strategy] = 0.0

    if verbose:
        print("\nRetrieval benchmarking complete.")

    return retrieval_times


def get_collection_size_mb(vector_db_path: Path, collection_name: str) -> float:
    """
    Calculate the disk size of a Qdrant collection directory.

    Args:
        vector_db_path: Path to the vector database root.
        collection_name: Name of the collection to measure.

    Returns:
        Size in megabytes. Returns 0.0 if the path does not exist.
    """
    collection_path = vector_db_path / "collection" / collection_name

    if not collection_path.exists():
        return 0.0

    total_bytes = 0
    for filepath in collection_path.rglob("*"):
        if filepath.is_file():
            try:
                total_bytes += filepath.stat().st_size
            except (OSError, PermissionError):
                continue

    return total_bytes / (1024 * 1024)


def calculate_all_index_sizes(
    vector_db_path: Path,
    collection_names: Dict[str, str],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calculate disk sizes for all strategy collections.

    Args:
        vector_db_path: Path to the vector database root.
        collection_names: Dict mapping strategy name to full collection name.
        verbose: If True, prints progress to stdout.

    Returns:
        Dict mapping strategy name to index size in MB.
    """
    if verbose:
        print("Calculating index sizes...")

    index_sizes = {}

    for strategy, collection_name in collection_names.items():
        size_mb = get_collection_size_mb(vector_db_path, collection_name)
        index_sizes[strategy] = size_mb
        if verbose:
            print(f"   {strategy:<16}: {size_mb:.2f} MB")

    if verbose:
        print("\nIndex size calculation complete.")

    return index_sizes


def create_comparison_table(
    strategy_chunks: Dict[str, List[Dict[str, Any]]],
    index_sizes: Dict[str, float],
    retrieval_times: Dict[str, float],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a comparison metrics table across all chunking strategies.

    Args:
        strategy_chunks: Dict mapping strategy name to chunk list.
        index_sizes: Dict mapping strategy name to index size in MB.
        retrieval_times: Dict mapping strategy name to retrieval time in ms.
        output_path: Optional path to save the table as CSV.

    Returns:
        DataFrame with columns: Strategy, Chunk Count, Avg Size (tokens),
        Index Size (MB), Retrieval Time (ms).
    """
    strategy_display_names = {
        'semantic':     'Semantic',
        'fixed':        'Fixed',
        'sliding':      'Sliding',
        'parent_child': 'Parent-Child',
        'late_chunk':   'Late Chunking',
    }

    comparison_data = []

    for strategy, chunks in strategy_chunks.items():
        if not chunks:
            continue

        avg_tokens = sum(c.get('token_count', 0) for c in chunks) / len(chunks)

        comparison_data.append({
            'Strategy':            strategy_display_names.get(strategy, strategy.title()),
            'Chunk Count':         len(chunks),
            'Avg Size (tokens)':   round(avg_tokens, 1),
            'Index Size (MB)':     round(index_sizes.get(strategy, 0), 2),
            'Retrieval Time (ms)': round(retrieval_times.get(strategy, 0), 2),
        })

    df = pd.DataFrame(comparison_data)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Comparison table saved to: {output_path}\n")

    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Pretty-print the comparison table to stdout.

    Args:
        df: Comparison DataFrame from create_comparison_table().
    """
    print("=" * 80)
    print("CHUNKING STRATEGY COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def analyze_trade_offs(df: pd.DataFrame) -> None:
    """
    Print key trade-off insights across chunking strategies.

    Identifies the best performer in each metric and summarises
    qualitative trade-offs between strategies.

    Args:
        df: Comparison DataFrame from create_comparison_table().
    """
    if df.empty:
        print("No data to analyze.")
        return

    print("\nANALYSIS & INSIGHTS\n")

    try:
        fastest       = df.loc[df['Retrieval Time (ms)'].idxmin()]
        smallest      = df.loc[df['Index Size (MB)'].idxmin()]
        most_chunks   = df.loc[df['Chunk Count'].idxmax()]
        largest_chunk = df.loc[df['Avg Size (tokens)'].idxmax()]

        print(f"   Fastest Retrieval  : {fastest['Strategy']} ({fastest['Retrieval Time (ms)']:.2f} ms)")
        print(f"   Smallest Index     : {smallest['Strategy']} ({smallest['Index Size (MB)']:.2f} MB)")
        print(f"   Most Chunks        : {most_chunks['Strategy']} ({most_chunks['Chunk Count']} chunks)")
        print(f"   Largest Avg Chunk  : {largest_chunk['Strategy']} ({largest_chunk['Avg Size (tokens)']:.1f} tokens)")
    except Exception as e:
        print(f"   Could not complete full analysis: {e}")

    print("\nKEY TRADE-OFFS:")
    print("   Sliding Window  : High chunk count -> Better recall but larger index")
    print("   Semantic        : Variable sizes   -> Preserves context but unpredictable")
    print("   Fixed           : Uniform sizes    -> Predictable but breaks semantic boundaries")
    print("   Parent-Child    : Two-tier         -> Best of both worlds but complex retrieval")
    print("   Late Chunking   : Fewer chunks     -> Smaller index, split at query time")


def run_full_evaluation(
    strategy_chunks: Dict[str, List[Dict[str, Any]]],
    collections: Dict[str, QdrantVectorStore],
    vector_db_path: Path,
    collection_prefix: str = "primelands",
    test_query: str = "luxury land in Colombo with highway access",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline for all chunking strategies.

    Calculates index sizes, benchmarks retrieval times, generates the
    comparison table, and prints trade-off analysis.

    Args:
        strategy_chunks: Dict mapping strategy name to chunk list.
        collections: Dict mapping strategy name to vectorstore.
        vector_db_path: Path to the vector database root.
        collection_prefix: Prefix used when naming Qdrant collections.
        test_query: Query used for retrieval benchmarking.
        output_path: Optional path to save the comparison CSV.

    Returns:
        DataFrame containing all comparison metrics.
    """
    collection_names = {
        strategy: f"{collection_prefix}_{strategy}"
        for strategy in strategy_chunks.keys()
    }

    index_sizes     = calculate_all_index_sizes(vector_db_path, collection_names)
    retrieval_times = benchmark_all_strategies(collections, test_query)

    df = create_comparison_table(
        strategy_chunks=strategy_chunks,
        index_sizes=index_sizes,
        retrieval_times=retrieval_times,
        output_path=output_path,
    )

    print_comparison_table(df)
    analyze_trade_offs(df)

    return df


__all__ = [
    'analyze_trade_offs',
    'benchmark_all_strategies',
    'benchmark_retrieval',
    'calculate_all_index_sizes',
    'create_comparison_table',
    'get_collection_size_mb',
    'print_comparison_table',
    'run_full_evaluation',
]