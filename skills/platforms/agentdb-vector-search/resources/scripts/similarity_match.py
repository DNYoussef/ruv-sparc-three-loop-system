#!/usr/bin/env python3
"""
Similarity Matching for AgentDB Vector Search
Advanced similarity computation with multiple metrics and optimizations
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class DistanceMetric(Enum):
    """Supported distance metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    HAMMING = "hamming"


@dataclass
class MatchResult:
    """Similarity match result"""
    index: int
    score: float
    distance: float
    rank: int


class SimilarityMatcher:
    """
    High-performance similarity matching for vector search

    Features:
    - Multiple distance metrics
    - Batch processing
    - SIMD optimizations
    - Sub-millisecond search
    """

    def __init__(
        self,
        metric: DistanceMetric = DistanceMetric.COSINE,
        use_simd: bool = True
    ):
        """
        Initialize similarity matcher

        Args:
            metric: Distance metric to use
            use_simd: Enable SIMD optimizations
        """
        self.metric = metric
        self.use_simd = use_simd

        # Precompute normalized vectors for cosine
        self.normalized_vectors: Optional[np.ndarray] = None

    def preprocess_vectors(self, vectors: np.ndarray) -> None:
        """
        Preprocess vectors for faster search

        Args:
            vectors: Array of shape (n, dim)
        """
        if self.metric == DistanceMetric.COSINE:
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.normalized_vectors = vectors / (norms + 1e-10)
        else:
            self.normalized_vectors = vectors

    def cosine_similarity_fast(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Fast cosine similarity using normalized vectors

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Similarity scores (n,)
        """
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Use preprocessed normalized vectors
        if self.normalized_vectors is not None:
            # Fast dot product (already normalized)
            similarities = np.dot(self.normalized_vectors, query_norm)
        else:
            # Normalize on the fly
            vectors_norm = vectors / (
                np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
            )
            similarities = np.dot(vectors_norm, query_norm)

        return similarities

    def euclidean_distance_fast(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Fast Euclidean distance using vectorized operations

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Distances (n,)
        """
        # Vectorized L2 norm
        diff = vectors - query
        distances = np.linalg.norm(diff, axis=1)
        return distances

    def dot_product_fast(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Fast dot product similarity

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Dot products (n,)
        """
        return np.dot(vectors, query)

    def manhattan_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Manhattan (L1) distance

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Distances (n,)
        """
        return np.sum(np.abs(vectors - query), axis=1)

    def chebyshev_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Chebyshev (L-inf) distance

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Distances (n,)
        """
        return np.max(np.abs(vectors - query), axis=1)

    def hamming_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Hamming distance for binary vectors

        Args:
            query: Query vector (dim,) binary
            vectors: Document vectors (n, dim) binary

        Returns:
            Distances (n,)
        """
        # XOR for binary vectors
        return np.sum(query != vectors, axis=1)

    def compute_similarity(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity/distance based on metric

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)

        Returns:
            Scores (n,) - higher is better
        """
        start = time.perf_counter()

        if self.metric == DistanceMetric.COSINE:
            scores = self.cosine_similarity_fast(query, vectors)

        elif self.metric == DistanceMetric.EUCLIDEAN:
            distances = self.euclidean_distance_fast(query, vectors)
            # Convert to similarity (0 to 1)
            scores = 1.0 / (1.0 + distances)

        elif self.metric == DistanceMetric.DOT_PRODUCT:
            scores = self.dot_product_fast(query, vectors)

        elif self.metric == DistanceMetric.MANHATTAN:
            distances = self.manhattan_distance(query, vectors)
            scores = 1.0 / (1.0 + distances)

        elif self.metric == DistanceMetric.CHEBYSHEV:
            distances = self.chebyshev_distance(query, vectors)
            scores = 1.0 / (1.0 + distances)

        elif self.metric == DistanceMetric.HAMMING:
            distances = self.hamming_distance(query, vectors)
            scores = 1.0 / (1.0 + distances)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        elapsed = (time.perf_counter() - start) * 1000
        print(f"Similarity computation: {elapsed:.3f}ms for {len(vectors)} vectors")

        return scores

    def find_top_k(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        k: int = 10,
        threshold: float = 0.0
    ) -> List[MatchResult]:
        """
        Find top-k most similar vectors

        Args:
            query: Query vector (dim,)
            vectors: Document vectors (n, dim)
            k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of MatchResult objects
        """
        # Compute similarities
        scores = self.compute_similarity(query, vectors)

        # Apply threshold
        valid_mask = scores >= threshold
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        # Get top-k
        if len(valid_scores) > k:
            top_k_indices = np.argpartition(valid_scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(valid_scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(valid_scores)[::-1]

        # Build results
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            actual_idx = valid_indices[idx]
            score = valid_scores[idx]

            # Compute distance
            if self.metric == DistanceMetric.COSINE:
                distance = 1.0 - score
            elif self.metric == DistanceMetric.DOT_PRODUCT:
                distance = -score  # Higher dot product = lower "distance"
            else:
                distance = 1.0 / score - 1.0

            results.append(MatchResult(
                index=int(actual_idx),
                score=float(score),
                distance=float(distance),
                rank=rank
            ))

        return results

    def batch_similarity(
        self,
        queries: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Batch similarity computation for multiple queries

        Args:
            queries: Query vectors (m, dim)
            vectors: Document vectors (n, dim)

        Returns:
            Similarity matrix (m, n)
        """
        if self.metric == DistanceMetric.COSINE:
            # Normalize both
            queries_norm = queries / (
                np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10
            )
            vectors_norm = vectors / (
                np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
            )
            # Matrix multiplication
            return np.dot(queries_norm, vectors_norm.T)

        elif self.metric == DistanceMetric.DOT_PRODUCT:
            return np.dot(queries, vectors.T)

        elif self.metric == DistanceMetric.EUCLIDEAN:
            # Pairwise Euclidean distances
            distances = euclidean_distances(queries, vectors)
            return 1.0 / (1.0 + distances)

        else:
            # Fallback: compute individually
            results = []
            for query in queries:
                scores = self.compute_similarity(query, vectors)
                results.append(scores)
            return np.array(results)

    def cross_similarity(
        self,
        vectors1: np.ndarray,
        vectors2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarity between two sets of vectors

        Args:
            vectors1: First set (m, dim)
            vectors2: Second set (n, dim)

        Returns:
            Similarity matrix (m, n)
        """
        return self.batch_similarity(vectors1, vectors2)


# Example usage and benchmarks
if __name__ == "__main__":
    # Generate random vectors for testing
    n_docs = 10000
    n_queries = 100
    dim = 384

    print(f"Generating {n_docs} documents and {n_queries} queries (dim={dim})")

    np.random.seed(42)
    documents = np.random.randn(n_docs, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    # Test different metrics
    metrics = [
        DistanceMetric.COSINE,
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.DOT_PRODUCT
    ]

    for metric in metrics:
        print(f"\n=== Testing {metric.value} ===")

        matcher = SimilarityMatcher(metric=metric)
        matcher.preprocess_vectors(documents)

        # Single query benchmark
        query = queries[0]
        start = time.perf_counter()
        results = matcher.find_top_k(query, documents, k=10)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Single query: {elapsed:.3f}ms")
        print("Top 3 results:")
        for result in results[:3]:
            print(f"  Rank {result.rank}: idx={result.index}, score={result.score:.4f}")

        # Batch query benchmark
        start = time.perf_counter()
        similarity_matrix = matcher.batch_similarity(queries, documents)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Batch ({n_queries} queries): {elapsed:.3f}ms")
        print(f"Avg per query: {elapsed/n_queries:.3f}ms")

        if elapsed/n_queries < 1.0:
            print("✅ Sub-millisecond performance!")
        else:
            print("⚠️ Performance could be improved")
