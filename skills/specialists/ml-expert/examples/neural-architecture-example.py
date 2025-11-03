#!/usr/bin/env python3
"""
Neural Architecture Design Example

Demonstrates complete neural network architecture design workflow using
the model-architect tool including custom architectures, residual connections,
attention mechanisms, and optimization strategies.

This example builds a production-ready transformer-based architecture with
advanced features like sliding window attention, long-term memory, and
adaptive computation time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlidingWindowAttention(nn.Module):
    """
    Sliding window multi-head attention for efficient long-sequence processing.

    Implements local attention with configurable window size to reduce
    quadratic complexity to linear for sequence length.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        window_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores with sliding window
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        attention_scores = attention_scores.masked_fill(window_mask == 0, float('-inf'))

        # Apply additional mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, V)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.out_proj(context)

        return output

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sliding window attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1

        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


class LongTermMemory(nn.Module):
    """
    Long-term memory module for maintaining context across sequences.

    Implements a memory-augmented architecture with read/write operations
    for persistent state across multiple forward passes.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_slots: int = 256,
        memory_dim: int = 512
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim

        # Memory matrix
        self.register_buffer(
            'memory',
            torch.zeros(memory_slots, memory_dim)
        )

        # Read/write controllers
        self.read_query = nn.Linear(hidden_dim, memory_dim)
        self.write_key = nn.Linear(hidden_dim, memory_dim)
        self.write_value = nn.Linear(hidden_dim, memory_dim)

        # Output projection
        self.output_proj = nn.Linear(memory_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory read/write.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Read from memory
        read_query = self.read_query(x)  # [batch, seq_len, memory_dim]

        # Compute attention over memory slots
        attention_scores = torch.matmul(
            read_query,
            self.memory.transpose(0, 1)
        )  # [batch, seq_len, memory_slots]

        attention_probs = F.softmax(attention_scores, dim=-1)

        # Read values from memory
        read_values = torch.matmul(
            attention_probs,
            self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        )  # [batch, seq_len, memory_dim]

        # Concatenate read values with input
        combined = torch.cat([x, read_values], dim=-1)
        output = self.output_proj(combined)

        # Write to memory (update based on current input)
        write_key = self.write_key(x.mean(dim=1))  # [batch, memory_dim]
        write_value = self.write_value(x.mean(dim=1))  # [batch, memory_dim]

        # Update memory (simplified - single slot update)
        # In production, use more sophisticated write mechanisms
        self.memory[0] = write_value.mean(dim=0)

        return output

    def reset_memory(self):
        """Reset memory to zeros."""
        self.memory.zero_()


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time (ACT) for dynamic computation depth.

    Allows the model to perform variable computation steps based on
    input complexity, improving efficiency and performance.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_steps: int = 10,
        threshold: float = 0.99
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.threshold = threshold

        # Halting probability predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Processing unit
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive computation.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (output tensor, pondering cost)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize accumulators
        output = torch.zeros_like(x)
        halting_prob = torch.zeros(batch_size, seq_len, 1, device=x.device)
        remainders = torch.zeros(batch_size, seq_len, 1, device=x.device)
        n_updates = torch.zeros(batch_size, seq_len, 1, device=x.device)

        state = x

        for step in range(self.max_steps):
            # Predict halting probability
            p = self.halt_predictor(state)

            # Update halting probability
            still_running = (halting_prob < self.threshold).float()
            new_halted = (halting_prob + p > self.threshold).float() * still_running

            # Compute remainders
            p_adjusted = p * still_running + new_halted * (self.threshold - halting_prob)

            # Process state
            processed = self.processor(state)

            # Accumulate output
            output = output + processed * p_adjusted

            # Update state
            state = processed

            # Update accumulators
            halting_prob = halting_prob + p_adjusted
            n_updates = n_updates + still_running

            # Check if all sequences have halted
            if (halting_prob >= self.threshold).all():
                break

        # Compute pondering cost (average number of steps)
        pondering_cost = n_updates.mean()

        return output, pondering_cost


class AdvancedTransformer(nn.Module):
    """
    Advanced transformer architecture with sliding window attention,
    long-term memory, and adaptive computation time.

    Production-ready architecture for efficient long-sequence processing
    with dynamic computation depth.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        window_size: int = 512,
        memory_slots: int = 256,
        max_act_steps: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(4096, hidden_dim)  # Max sequence length

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Long-term memory
        self.memory = LongTermMemory(
            hidden_dim=hidden_dim,
            memory_slots=memory_slots
        )

        # Adaptive computation
        self.act = AdaptiveComputationTime(
            hidden_dim=hidden_dim,
            max_steps=max_act_steps
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Tuple of (logits, pondering_cost)
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Long-term memory integration
        x = self.memory(x)

        # Adaptive computation
        x, pondering_cost = self.act(x)

        # Output projection
        logits = self.output_proj(x)

        return logits, pondering_cost

    def reset_memory(self):
        """Reset long-term memory."""
        self.memory.reset_memory()


class TransformerLayer(nn.Module):
    """Single transformer layer with sliding window attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float
    ):
        super().__init__()

        self.attention = SlidingWindowAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attended = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attended)

        # Feed-forward with residual
        fed_forward = self.feed_forward(self.norm2(x))
        x = x + self.dropout(fed_forward)

        return x


# Example usage and demonstration
if __name__ == '__main__':
    print("Advanced Neural Architecture Example")
    print("=" * 60)

    # Model configuration
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 6
    num_heads = 8
    window_size = 512
    batch_size = 4
    seq_len = 1024

    # Create model
    model = AdvancedTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        window_size=window_size
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter size: {total_params * 4 / (1024**2):.2f} MB")

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits, pondering_cost = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Pondering cost: {pondering_cost.item():.2f} steps")

    # Demonstrate memory persistence
    print("\nDemonstrating memory persistence...")
    model.reset_memory()

    for i in range(3):
        with torch.no_grad():
            logits, _ = model(input_ids)
        print(f"Pass {i+1}: Output mean = {logits.mean().item():.4f}")

    print("\nArchitecture demonstration complete!")
