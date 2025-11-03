#!/usr/bin/env python3
"""
Custom Distance Metrics for AgentDB Advanced Search
Implements specialized distance functions for domain-specific similarity
"""

import numpy as np
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for custom metrics"""
    name: str
    description: str
    normalize: bool = True
    symmetric: bool = True


class CustomMetrics:
    """Collection of custom distance metrics for AgentDB"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity (standard implementation)
        Range: [-1, 1] where 1 is identical, -1 is opposite
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Euclidean (L2) distance
        Range: [0, ∞] where 0 is identical
        """
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Manhattan (L1) distance
        Range: [0, ∞] where 0 is identical
        """
        return np.sum(np.abs(vec1 - vec2))

    @staticmethod
    def chebyshev_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Chebyshev (L∞) distance - maximum absolute difference
        Range: [0, ∞] where 0 is identical
        """
        return np.max(np.abs(vec1 - vec2))

    @staticmethod
    def minkowski_distance(vec1: np.ndarray, vec2: np.ndarray, p: float = 3) -> float:
        """
        Minkowski distance (generalization of Euclidean and Manhattan)
        p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev
        Range: [0, ∞] where 0 is identical
        """
        return np.power(np.sum(np.power(np.abs(vec1 - vec2), p)), 1/p)

    @staticmethod
    def weighted_euclidean(vec1: np.ndarray, vec2: np.ndarray, weights: np.ndarray) -> float:
        """
        Weighted Euclidean distance
        Applies different importance to each dimension
        """
        if weights.shape != vec1.shape:
            raise ValueError("Weights must have same shape as vectors")

        weighted_diff = weights * (vec1 - vec2)
        return np.linalg.norm(weighted_diff)

    @staticmethod
    def hamming_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Hamming distance (for binary vectors)
        Counts the number of differing elements
        """
        return np.sum(vec1 != vec2)

    @staticmethod
    def jaccard_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Jaccard similarity (for binary/set-based vectors)
        Range: [0, 1] where 1 is identical
        """
        intersection = np.sum(np.minimum(vec1, vec2))
        union = np.sum(np.maximum(vec1, vec2))

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Pearson correlation coefficient
        Range: [-1, 1] where 1 is perfect positive correlation
        """
        if len(vec1) < 2:
            return 0.0

        mean1 = np.mean(vec1)
        mean2 = np.mean(vec2)

        numerator = np.sum((vec1 - mean1) * (vec2 - mean2))
        denominator = np.sqrt(np.sum((vec1 - mean1)**2) * np.sum((vec2 - mean2)**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def spearman_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Spearman rank correlation
        Range: [-1, 1] where 1 is perfect monotonic relationship
        """
        rank1 = np.argsort(np.argsort(vec1))
        rank2 = np.argsort(np.argsort(vec2))

        return CustomMetrics.pearson_correlation(rank1, rank2)

    @staticmethod
    def mahalanobis_distance(vec1: np.ndarray, vec2: np.ndarray,
                            covariance_inv: np.ndarray) -> float:
        """
        Mahalanobis distance (accounts for correlations)
        Requires inverse covariance matrix of the dataset
        """
        diff = vec1 - vec2
        return np.sqrt(diff.T @ covariance_inv @ diff)

    @staticmethod
    def canberra_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Canberra distance (sensitive to small changes near zero)
        Range: [0, ∞]
        """
        numerator = np.abs(vec1 - vec2)
        denominator = np.abs(vec1) + np.abs(vec2)

        # Avoid division by zero
        mask = denominator != 0
        result = np.zeros_like(numerator)
        result[mask] = numerator[mask] / denominator[mask]

        return np.sum(result)

    @staticmethod
    def angular_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Angular distance (angle between vectors)
        Range: [0, π] radians
        """
        cos_sim = CustomMetrics.cosine_similarity(vec1, vec2)
        # Clamp to [-1, 1] to handle numerical errors
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return np.arccos(cos_sim)

    @staticmethod
    def bhattacharyya_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Bhattacharyya distance (for probability distributions)
        Range: [0, ∞]
        """
        # Normalize to probability distributions
        vec1_norm = vec1 / np.sum(vec1) if np.sum(vec1) > 0 else vec1
        vec2_norm = vec2 / np.sum(vec2) if np.sum(vec2) > 0 else vec2

        bc = np.sum(np.sqrt(vec1_norm * vec2_norm))

        if bc == 0:
            return float('inf')

        return -np.log(bc)

    @staticmethod
    def hellinger_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Hellinger distance (for probability distributions)
        Range: [0, 1]
        """
        # Normalize to probability distributions
        vec1_norm = vec1 / np.sum(vec1) if np.sum(vec1) > 0 else vec1
        vec2_norm = vec2 / np.sum(vec2) if np.sum(vec2) > 0 else vec2

        return np.sqrt(0.5 * np.sum((np.sqrt(vec1_norm) - np.sqrt(vec2_norm))**2))


class DomainSpecificMetrics:
    """Domain-specific custom metrics"""

    @staticmethod
    def time_weighted_similarity(vec1: np.ndarray, vec2: np.ndarray,
                                 timestamp1: float, timestamp2: float,
                                 decay_factor: float = 0.1) -> float:
        """
        Time-weighted cosine similarity
        Recent vectors get higher weight
        """
        base_similarity = CustomMetrics.cosine_similarity(vec1, vec2)
        time_diff = abs(timestamp1 - timestamp2)
        time_weight = np.exp(-decay_factor * time_diff)

        return base_similarity * time_weight

    @staticmethod
    def hierarchical_similarity(vec1: np.ndarray, vec2: np.ndarray,
                               hierarchy_levels: List[int]) -> float:
        """
        Hierarchical similarity for structured embeddings
        Different weights for different hierarchy levels

        Args:
            vec1, vec2: Embeddings
            hierarchy_levels: List of dimension indices for each level
        """
        similarities = []
        weights = [1.0 / (i + 1) for i in range(len(hierarchy_levels))]

        for i, level_dims in enumerate(hierarchy_levels):
            level_vec1 = vec1[level_dims]
            level_vec2 = vec2[level_dims]
            sim = CustomMetrics.cosine_similarity(level_vec1, level_vec2)
            similarities.append(sim * weights[i])

        return sum(similarities) / sum(weights)

    @staticmethod
    def semantic_drift_distance(vec1: np.ndarray, vec2: np.ndarray,
                                reference_vec: np.ndarray) -> float:
        """
        Measures how similarly two vectors drift from a reference
        Useful for tracking concept evolution
        """
        drift1 = CustomMetrics.cosine_similarity(vec1, reference_vec)
        drift2 = CustomMetrics.cosine_similarity(vec2, reference_vec)

        return abs(drift1 - drift2)


class MetricFactory:
    """Factory for creating and managing custom metrics"""

    def __init__(self):
        self.metrics: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default metrics"""
        self.register("cosine", CustomMetrics.cosine_similarity)
        self.register("euclidean", CustomMetrics.euclidean_distance)
        self.register("manhattan", CustomMetrics.manhattan_distance)
        self.register("chebyshev", CustomMetrics.chebyshev_distance)
        self.register("jaccard", CustomMetrics.jaccard_similarity)
        self.register("pearson", CustomMetrics.pearson_correlation)
        self.register("angular", CustomMetrics.angular_distance)

    def register(self, name: str, metric_fn: Callable):
        """Register a custom metric"""
        self.metrics[name] = metric_fn
        logger.info(f"Registered metric: {name}")

    def get(self, name: str) -> Optional[Callable]:
        """Get a metric by name"""
        return self.metrics.get(name)

    def list_metrics(self) -> List[str]:
        """List all available metrics"""
        return list(self.metrics.keys())

    def compute(self, name: str, vec1: np.ndarray, vec2: np.ndarray,
                **kwargs) -> float:
        """Compute distance/similarity using named metric"""
        metric_fn = self.get(name)
        if metric_fn is None:
            raise ValueError(f"Unknown metric: {name}")

        return metric_fn(vec1, vec2, **kwargs)


def create_weighted_metric(weights: np.ndarray) -> Callable:
    """
    Factory function to create weighted distance metrics

    Example:
        weights = np.array([1.0, 2.0, 1.5, ...])  # Higher weight = more important
        weighted_dist = create_weighted_metric(weights)
        distance = weighted_dist(vec1, vec2)
    """
    def weighted_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return CustomMetrics.weighted_euclidean(vec1, vec2, weights)

    return weighted_distance


def create_domain_metric(config: Dict) -> Callable:
    """
    Create a domain-specific metric from configuration

    Example config:
    {
        "type": "hierarchical",
        "hierarchy_levels": [[0, 128], [128, 256], [256, 384]],
        "weights": [1.0, 0.5, 0.25]
    }
    """
    metric_type = config.get("type")

    if metric_type == "hierarchical":
        def hierarchical_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
            return DomainSpecificMetrics.hierarchical_similarity(
                vec1, vec2, config["hierarchy_levels"]
            )
        return hierarchical_metric

    elif metric_type == "time_weighted":
        decay_factor = config.get("decay_factor", 0.1)
        def time_weighted_metric(vec1: np.ndarray, vec2: np.ndarray,
                                t1: float, t2: float) -> float:
            return DomainSpecificMetrics.time_weighted_similarity(
                vec1, vec2, t1, t2, decay_factor
            )
        return time_weighted_metric

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


if __name__ == "__main__":
    # Example usage
    vec1 = np.random.rand(384)
    vec2 = np.random.rand(384)

    factory = MetricFactory()

    print("Available metrics:", factory.list_metrics())
    print("\nMetric comparisons:")

    for metric_name in factory.list_metrics():
        distance = factory.compute(metric_name, vec1, vec2)
        print(f"  {metric_name}: {distance:.4f}")
