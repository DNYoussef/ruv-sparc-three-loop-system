# Quantization Techniques Reference

## Overview

Quantization compresses vector representations by reducing precision, trading minimal accuracy loss for dramatic memory and speed improvements. This reference covers the mathematical foundations and practical implementation of quantization in AgentDB.

---

## Quantization Methods

### 1. Binary Quantization (32x Compression)

**Principle:** Convert float32 values to single bits (positive → 1, negative → 0)

**Mathematical Formula:**
```
For each dimension d in embedding e:
  binary[d] = 1 if e[d] ≥ 0
  binary[d] = 0 if e[d] < 0
```

**Implementation:**
```typescript
function binaryQuantize(embedding: number[]): Uint8Array {
  const numBytes = Math.ceil(embedding.length / 8);
  const binary = new Uint8Array(numBytes);

  for (let i = 0; i < embedding.length; i++) {
    if (embedding[i] >= 0) {
      const byteIdx = Math.floor(i / 8);
      const bitIdx = i % 8;
      binary[byteIdx] |= (1 << bitIdx);
    }
  }

  return binary;
}
```

**Distance Computation (Hamming Distance):**
```typescript
function binaryDistance(a: Uint8Array, b: Uint8Array): number {
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    // XOR and count set bits (population count)
    let xor = a[i] ^ b[i];
    while (xor) {
      distance += xor & 1;
      xor >>= 1;
    }
  }
  return distance;
}
```

**Characteristics:**
- **Compression:** 32x (768 dims × 4 bytes → 96 bytes)
- **Accuracy:** 95-98% (2-5% loss)
- **Speed:** 10x faster (bit operations)
- **Best for:** Large-scale deployments, mobile/edge

**Trade-offs:**
- ✅ Extreme memory reduction
- ✅ Very fast distance computation
- ✅ Cache-friendly (small vectors)
- ❌ Moderate accuracy loss (5%)
- ❌ Not reversible (lossy)

---

### 2. Scalar Quantization (4x Compression)

**Principle:** Convert float32 to uint8 (256 discrete values)

**Mathematical Formula:**
```
For each dimension d in embedding e:
  min_val = min(e)
  max_val = max(e)
  scale = (max_val - min_val) / 255
  quantized[d] = round((e[d] - min_val) / scale)
```

**Implementation:**
```typescript
function scalarQuantize(embedding: number[]): {
  quantized: Uint8Array;
  min: number;
  scale: number;
} {
  const min = Math.min(...embedding);
  const max = Math.max(...embedding);
  const scale = (max - min) / 255;

  const quantized = new Uint8Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) {
    quantized[i] = Math.round((embedding[i] - min) / scale);
  }

  return { quantized, min, scale };
}

function scalarDequantize(
  quantized: Uint8Array,
  min: number,
  scale: number
): number[] {
  const embedding = new Array(quantized.length);
  for (let i = 0; i < quantized.length; i++) {
    embedding[i] = quantized[i] * scale + min;
  }
  return embedding;
}
```

**Distance Computation (Approximate L2):**
```typescript
function scalarDistance(
  a: Uint8Array,
  b: Uint8Array,
  aScale: number,
  bScale: number
): number {
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] * aScale) - (b[i] * bScale);
    distance += diff * diff;
  }
  return Math.sqrt(distance);
}
```

**Characteristics:**
- **Compression:** 4x (768 dims × 4 bytes → 768 bytes)
- **Accuracy:** 98-99% (1-2% loss)
- **Speed:** 3x faster
- **Best for:** Production apps requiring high accuracy

**Trade-offs:**
- ✅ Good compression ratio
- ✅ Minimal accuracy loss
- ✅ Reversible (can reconstruct approximate original)
- ✅ Fast distance computation
- ❌ Requires storing min/scale per vector

---

### 3. Product Quantization (8-16x Compression)

**Principle:** Split vector into subvectors and quantize each independently

**Mathematical Formula:**
```
Split embedding e into m subvectors of dimension d/m:
  e = [e₁, e₂, ..., eₘ]

For each subvector eᵢ:
  1. Learn k centroids via k-means
  2. Replace eᵢ with nearest centroid index
  3. Store as log₂(k) bits
```

**Implementation (Simplified):**
```typescript
function productQuantize(
  embedding: number[],
  numSubvectors: number = 8,
  numCentroids: number = 256
): {
  codes: Uint8Array;
  codebooks: number[][][];
} {
  const subvectorSize = embedding.length / numSubvectors;
  const codes = new Uint8Array(numSubvectors);
  const codebooks: number[][][] = [];

  for (let i = 0; i < numSubvectors; i++) {
    const subvector = embedding.slice(
      i * subvectorSize,
      (i + 1) * subvectorSize
    );

    // Find nearest centroid (pre-computed codebook)
    const codebook = getCodebook(i); // k-means trained codebook
    const centroidIdx = findNearestCentroid(subvector, codebook);

    codes[i] = centroidIdx;
    codebooks.push(codebook);
  }

  return { codes, codebooks };
}
```

**Distance Computation (Asymmetric Distance):**
```typescript
function productDistance(
  query: number[],
  codes: Uint8Array,
  codebooks: number[][][]
): number {
  let distance = 0;
  const numSubvectors = codes.length;
  const subvectorSize = query.length / numSubvectors;

  for (let i = 0; i < numSubvectors; i++) {
    const querySubvector = query.slice(
      i * subvectorSize,
      (i + 1) * subvectorSize
    );

    const centroid = codebooks[i][codes[i]];

    // Compute L2 distance for this subvector
    let subDistance = 0;
    for (let j = 0; j < subvectorSize; j++) {
      const diff = querySubvector[j] - centroid[j];
      subDistance += diff * diff;
    }

    distance += subDistance;
  }

  return Math.sqrt(distance);
}
```

**Characteristics:**
- **Compression:** 8-16x (768 dims × 4 bytes → 48-96 bytes)
- **Accuracy:** 93-97% (3-7% loss)
- **Speed:** 5x faster
- **Best for:** High-dimensional vectors, image/video embeddings

**Trade-offs:**
- ✅ Excellent compression for high dims
- ✅ Asymmetric distance (fast query, exact query vector)
- ✅ Scalable to billions of vectors
- ❌ Requires codebook training (k-means)
- ❌ Moderate accuracy loss (3-7%)
- ❌ Complex implementation

---

## Quantization Selection Guide

### By Accuracy Requirements

| Accuracy Needed | Method | Compression | Typical Use Case |
|----------------|--------|-------------|------------------|
| >99% | None | 1x | Small datasets, research |
| 98-99% | Scalar | 4x | Production apps |
| 93-97% | Product | 8-16x | Image search, video |
| 95-98% | Binary | 32x | Mobile, large-scale |

### By Memory Constraints

| Available Memory | Vector Count | Recommended Method |
|-----------------|-------------|-------------------|
| >10GB | <1M | None (full precision) |
| 1-10GB | 100K-1M | Scalar (4x) |
| 500MB-1GB | 1M-5M | Product (8-16x) |
| <500MB | >5M | Binary (32x) |

### By Dataset Size

| Vector Count | Full Precision | Scalar (4x) | Product (16x) | Binary (32x) |
|-------------|---------------|------------|---------------|-------------|
| 10K | 30MB | 8MB | 2MB | 1MB |
| 100K | 300MB | 75MB | 19MB | 9MB |
| 1M | 3GB | 768MB | 192MB | 96MB |
| 10M | 30GB | 7.5GB | 1.9GB | 960MB |

---

## Advanced Techniques

### 4. Learned Quantization

**Principle:** Use neural networks to learn optimal quantization

```typescript
// Pseudo-code (requires ML framework)
function learnedQuantize(embeddings: number[][]): QuantizationModel {
  // Train autoencoder to compress embeddings
  const encoder = trainAutoencoder({
    inputDim: 768,
    compressedDim: 96,  // 8x compression
    epochs: 100,
  });

  return encoder;
}

// Apply learned quantization
const compressedEmbedding = encoder.encode(embedding);
const reconstructedEmbedding = encoder.decode(compressedEmbedding);
```

**Characteristics:**
- **Compression:** Variable (8-32x)
- **Accuracy:** 97-99% (better than PQ)
- **Cost:** Requires training on representative data

### 5. Residual Quantization

**Principle:** Iteratively quantize residuals for better accuracy

```typescript
function residualQuantize(
  embedding: number[],
  numLayers: number = 3
): Uint8Array[] {
  let residual = embedding;
  const codes: Uint8Array[] = [];

  for (let layer = 0; layer < numLayers; layer++) {
    const { code, codebook } = productQuantize(residual);
    codes.push(code);

    // Compute residual
    const reconstructed = reconstruct(code, codebook);
    residual = residual.map((v, i) => v - reconstructed[i]);
  }

  return codes;
}
```

**Characteristics:**
- **Compression:** 8-16x
- **Accuracy:** 96-98% (better than single-layer PQ)
- **Cost:** More complex, slower encoding

---

## Performance Optimization Tips

### 1. SIMD Acceleration

```typescript
// Use SIMD instructions for distance computation
function simdBinaryDistance(a: Uint8Array, b: Uint8Array): number {
  // Modern CPUs support 128-bit, 256-bit, or 512-bit SIMD
  // Process 16, 32, or 64 bytes at once
  // Typical speedup: 4-8x

  // Pseudo-code (requires SIMD intrinsics)
  const chunks = Math.floor(a.length / 16);
  let distance = 0;

  for (let i = 0; i < chunks; i++) {
    const aChunk = loadSIMD(a, i * 16);
    const bChunk = loadSIMD(b, i * 16);
    distance += popcountSIMD(xorSIMD(aChunk, bChunk));
  }

  return distance;
}
```

### 2. Cache Optimization

```typescript
// Store quantized vectors contiguously for cache efficiency
class QuantizedVectorStore {
  private vectors: Uint8Array;
  private vectorSize: number;

  constructor(vectorSize: number, numVectors: number) {
    this.vectorSize = vectorSize;
    this.vectors = new Uint8Array(vectorSize * numVectors);
  }

  get(index: number): Uint8Array {
    const start = index * this.vectorSize;
    return this.vectors.subarray(start, start + this.vectorSize);
  }

  // Benefit: Sequential access pattern, cache-friendly
}
```

### 3. Batch Processing

```typescript
async function batchQuantize(
  embeddings: number[][],
  method: 'binary' | 'scalar' | 'product'
): Promise<Uint8Array[]> {
  // Process in parallel batches
  const BATCH_SIZE = 1000;
  const results: Uint8Array[] = [];

  for (let i = 0; i < embeddings.length; i += BATCH_SIZE) {
    const batch = embeddings.slice(i, i + BATCH_SIZE);
    const quantized = await Promise.all(
      batch.map(e => quantize(e, method))
    );
    results.push(...quantized);
  }

  return results;
}
```

---

## Accuracy Evaluation

### Recall Metrics

```typescript
function evaluateRecall(
  queries: number[][],
  originalDB: VectorDB,
  quantizedDB: VectorDB,
  k: number = 10
): {
  recallAt10: number;
  recallAt100: number;
  avgDistance: number;
} {
  let totalRecall10 = 0;
  let totalRecall100 = 0;
  let totalDistance = 0;

  for (const query of queries) {
    const originalResults = originalDB.search(query, 100);
    const quantizedResults = quantizedDB.search(query, 100);

    // Recall@10
    const top10Original = originalResults.slice(0, 10).map(r => r.id);
    const top10Quantized = quantizedResults.slice(0, 10).map(r => r.id);
    const overlap10 = top10Original.filter(id => top10Quantized.includes(id)).length;
    totalRecall10 += overlap10 / 10;

    // Recall@100
    const ids100Original = originalResults.map(r => r.id);
    const ids100Quantized = quantizedResults.map(r => r.id);
    const overlap100 = ids100Original.filter(id => ids100Quantized.includes(id)).length;
    totalRecall100 += overlap100 / 100;

    // Distance correlation
    totalDistance += correlate(
      originalResults.map(r => r.distance),
      quantizedResults.map(r => r.distance)
    );
  }

  return {
    recallAt10: totalRecall10 / queries.length,
    recallAt100: totalRecall100 / queries.length,
    avgDistance: totalDistance / queries.length,
  };
}
```

---

## References

1. **Binary Quantization**
   - "Practical and Optimal LSH for Angular Distance" (Andoni et al., 2015)
   - 32x compression, 95-98% accuracy

2. **Scalar Quantization**
   - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
   - 4x compression, 98-99% accuracy

3. **Product Quantization**
   - "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)
   - 8-16x compression, 93-97% accuracy

4. **Survey Paper**
   - "A Survey on Learning to Hash" (Wang et al., 2018)
   - Comprehensive overview of quantization methods

---

## Related Documentation

- [Main Skill](../skill.md)
- [Quantization Example](../examples/example-1-quantization.md)
- [HNSW Parameters](./hnsw-parameters.md)
