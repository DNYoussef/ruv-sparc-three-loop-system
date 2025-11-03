#!/usr/bin/env python3
"""
AgentDB Vector Quantization
Reduces memory usage by 4-32x with minimal accuracy loss
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import struct
import json
from dataclasses import dataclass, asdict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'sentence-transformers'])
    from sentence_transformers import SentenceTransformer


@dataclass
class QuantizationMetrics:
    """Metrics for vector quantization"""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    memory_reduction_percent: float
    reconstruction_error: float
    quantization_method: str


class VectorQuantizer:
    """Vector quantization for memory-efficient storage"""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector quantizer

        Args:
            embedding_dim: Dimension of input vectors
        """
        self.embedding_dim = embedding_dim

    def quantize_scalar(self, vectors: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Scalar quantization: Map float32 to int8/int16

        Args:
            vectors: Input vectors (N, D) shape
            bits: Quantization bits (8 or 16)

        Returns:
            Quantized vectors and scale/offset parameters
        """
        if bits == 8:
            dtype = np.int8
            max_val = 127
        elif bits == 16:
            dtype = np.int16
            max_val = 32767
        else:
            raise ValueError("bits must be 8 or 16")

        # Calculate scale and offset
        min_val = vectors.min()
        max_val_data = vectors.max()
        scale = (max_val_data - min_val) / (2 * max_val)
        offset = min_val + scale * max_val

        # Quantize
        quantized = np.clip(
            np.round((vectors - offset) / scale),
            -max_val,
            max_val
        ).astype(dtype)

        params = {
            'scale': float(scale),
            'offset': float(offset),
            'bits': bits
        }

        return quantized, params

    def dequantize_scalar(self, quantized: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Reconstruct vectors from scalar quantization"""
        return quantized.astype(np.float32) * params['scale'] + params['offset']

    def quantize_product(self, vectors: np.ndarray,
                        n_subvectors: int = 8,
                        n_centroids: int = 256) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Product quantization: Split vectors into subvectors and cluster

        Args:
            vectors: Input vectors (N, D)
            n_subvectors: Number of subvector splits
            n_centroids: Centroids per subvector (typically 256 for 8-bit)

        Returns:
            Quantized codes and codebook per subvector
        """
        N, D = vectors.shape

        if D % n_subvectors != 0:
            raise ValueError(f"Embedding dim {D} must be divisible by n_subvectors {n_subvectors}")

        subvector_dim = D // n_subvectors
        codes = np.zeros((N, n_subvectors), dtype=np.uint8)
        codebooks = []

        print(f"Product quantization: {n_subvectors} subvectors × {n_centroids} centroids")

        for i in range(n_subvectors):
            start_idx = i * subvector_dim
            end_idx = start_idx + subvector_dim
            subvectors = vectors[:, start_idx:end_idx]

            # K-means clustering (simplified - use sklearn for production)
            centroids = self._kmeans(subvectors, n_centroids)
            codebooks.append(centroids)

            # Assign codes
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

            if (i + 1) % 2 == 0:
                print(f"  Processed {i+1}/{n_subvectors} subvectors")

        return codes, codebooks

    def dequantize_product(self, codes: np.ndarray, codebooks: List[np.ndarray]) -> np.ndarray:
        """Reconstruct vectors from product quantization"""
        N, n_subvectors = codes.shape
        subvector_dim = codebooks[0].shape[1]

        reconstructed = np.zeros((N, n_subvectors * subvector_dim), dtype=np.float32)

        for i in range(n_subvectors):
            start_idx = i * subvector_dim
            end_idx = start_idx + subvector_dim
            reconstructed[:, start_idx:end_idx] = codebooks[i][codes[:, i]]

        return reconstructed

    def _kmeans(self, data: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
        """Simple k-means implementation"""
        # Initialize centroids randomly
        indices = np.random.choice(len(data), k, replace=False)
        centroids = data[indices].copy()

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                data[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
                for i in range(k)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids

    def calculate_metrics(self, original: np.ndarray,
                         reconstructed: np.ndarray,
                         method: str,
                         compression_ratio: float) -> QuantizationMetrics:
        """Calculate quantization quality metrics"""
        original_size = original.nbytes / (1024 ** 2)
        quantized_size = original_size / compression_ratio

        # Mean squared error
        mse = np.mean((original - reconstructed) ** 2)

        # Cosine similarity (for vector search)
        cosine_sim = np.mean([
            np.dot(original[i], reconstructed[i]) /
            (np.linalg.norm(original[i]) * np.linalg.norm(reconstructed[i]))
            for i in range(min(100, len(original)))
        ])

        return QuantizationMetrics(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            memory_reduction_percent=(1 - 1/compression_ratio) * 100,
            reconstruction_error=float(mse),
            quantization_method=method
        )

    def benchmark_quantization_methods(self, vectors: np.ndarray) -> Dict[str, QuantizationMetrics]:
        """Compare different quantization methods"""
        print(f"\n{'='*60}")
        print("Vector Quantization Benchmark")
        print(f"{'='*60}")
        print(f"Input shape: {vectors.shape}")
        print(f"Original size: {vectors.nbytes / (1024**2):.2f} MB")
        print("")

        results = {}

        # 1. INT8 Scalar Quantization (4x compression)
        print("1️⃣  INT8 Scalar Quantization...")
        quant_int8, params_int8 = self.quantize_scalar(vectors, bits=8)
        recon_int8 = self.dequantize_scalar(quant_int8, params_int8)
        results['int8'] = self.calculate_metrics(vectors, recon_int8, 'int8', 4.0)
        print(f"   Compression: {results['int8'].compression_ratio}x")
        print(f"   Error: {results['int8'].reconstruction_error:.6f}")

        # 2. INT16 Scalar Quantization (2x compression)
        print("\n2️⃣  INT16 Scalar Quantization...")
        quant_int16, params_int16 = self.quantize_scalar(vectors, bits=16)
        recon_int16 = self.dequantize_scalar(quant_int16, params_int16)
        results['int16'] = self.calculate_metrics(vectors, recon_int16, 'int16', 2.0)
        print(f"   Compression: {results['int16'].compression_ratio}x")
        print(f"   Error: {results['int16'].reconstruction_error:.6f}")

        # 3. Product Quantization 8 subvectors (32x compression)
        print("\n3️⃣  Product Quantization (8 subvectors)...")
        codes_pq8, codebooks_pq8 = self.quantize_product(vectors, n_subvectors=8)
        recon_pq8 = self.dequantize_product(codes_pq8, codebooks_pq8)
        # PQ compression: (N * 8 bytes) vs (N * 384 * 4 bytes) ≈ 32x
        results['pq8'] = self.calculate_metrics(vectors, recon_pq8, 'pq8', 32.0)
        print(f"   Compression: {results['pq8'].compression_ratio}x")
        print(f"   Error: {results['pq8'].reconstruction_error:.6f}")

        # 4. Product Quantization 16 subvectors (16x compression)
        print("\n4️⃣  Product Quantization (16 subvectors)...")
        codes_pq16, codebooks_pq16 = self.quantize_product(vectors, n_subvectors=16)
        recon_pq16 = self.dequantize_product(codes_pq16, codebooks_pq16)
        results['pq16'] = self.calculate_metrics(vectors, recon_pq16, 'pq16', 16.0)
        print(f"   Compression: {results['pq16'].compression_ratio}x")
        print(f"   Error: {results['pq16'].reconstruction_error:.6f}")

        return results

    def print_summary(self, results: Dict[str, QuantizationMetrics]):
        """Print quantization comparison summary"""
        print(f"\n{'='*60}")
        print("QUANTIZATION SUMMARY")
        print(f"{'='*60}\n")

        print("┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│ Method      │ Ratio    │ Size     │ Saved    │ Error    │")
        print("├─────────────┼──────────┼──────────┼──────────┼──────────┤")

        for method, metrics in results.items():
            print(f"│ {method:11s} │ {metrics.compression_ratio:5.0f}x   │ " \
                  f"{metrics.quantized_size_mb:5.1f} MB │ " \
                  f"{metrics.memory_reduction_percent:5.1f}%  │ " \
                  f"{metrics.reconstruction_error:8.6f} │")

        print("└─────────────┴──────────┴──────────┴──────────┴──────────┘")

        print("\n📊 Recommendations:")
        print("  • INT8: Fast reconstruction, 4x compression, minimal error")
        print("  • INT16: Highest quality, 2x compression")
        print("  • PQ8: Maximum compression (32x), acceptable for similarity search")
        print("  • PQ16: Balanced compression (16x), better accuracy than PQ8")

        print("\n💡 Use Cases:")
        print("  • High-accuracy: INT16 scalar quantization")
        print("  • Balanced: INT8 scalar quantization")
        print("  • Memory-constrained: PQ8 or PQ16")
        print(f"\n{'='*60}\n")


def main():
    """Run quantization benchmark"""
    # Initialize model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate test vectors
    print("Generating test vectors...")
    test_docs = [
        f"This is test document {i} about various topics in AI and machine learning."
        for i in range(1000)
    ]
    vectors = model.encode(test_docs, show_progress_bar=True)

    # Run quantization benchmark
    quantizer = VectorQuantizer(embedding_dim=vectors.shape[1])
    results = quantizer.benchmark_quantization_methods(vectors)
    quantizer.print_summary(results)

    # Save results
    output = {
        'vector_count': len(vectors),
        'embedding_dim': vectors.shape[1],
        'results': {k: asdict(v) for k, v in results.items()}
    }

    with open('quantization_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Results saved to quantization_results.json")


if __name__ == "__main__":
    main()
