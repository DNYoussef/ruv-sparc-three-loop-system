"""
Base classes for modular image generation.
Supports pluggable providers: local models, OpenAI, Replicate, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from enum import Enum



# ============================================================
# LIBRARY-FIRST PROTOCOL
# ============================================================
# Before generating code, check:
#   1. .claude/library/catalog.json
#   2. .claude/docs/inventories/LIBRARY-PATTERNS-GUIDE.md
#   3. D:\Projects\* for existing implementations
#
# Decision: REUSE (>90%) | ADAPT (70-90%) | FOLLOW pattern | BUILD new
# ============================================================

class ImageProvider(Enum):
    """Supported image generation providers."""
    LOCAL_SDXL = "local_sdxl"          # SDXL Lightning local
    LOCAL_SD15 = "local_sd15"          # Stable Diffusion 1.5 local
    OPENAI = "openai"                  # DALL-E 3
    REPLICATE = "replicate"            # Replicate API
    STABILITY = "stability"            # Stability AI API
    ATLAS = "atlas"                    # Atlas Cloud API (explicit opt-in)
    CUSTOM = "custom"                  # User-defined


@dataclass
class ImageConfig:
    """Configuration for image generation."""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4       # Low for SDXL Lightning
    guidance_scale: float = 0.0        # 0 for Lightning
    num_images: int = 1
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    confirm_paid: bool = False           # Explicit confirmation for guarded APIs


@dataclass
class GeneratedImage:
    """Result of image generation."""
    path: Path
    prompt: str
    provider: ImageProvider
    config: ImageConfig
    generation_time_seconds: float


class ImageGeneratorBase(ABC):
    """Abstract base class for image generators."""

    provider: ImageProvider

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (models downloaded, API key set, etc.)"""
        pass

    @abstractmethod
    def setup(self) -> bool:
        """Download models or configure API. Returns True if successful."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: Path,
        config: Optional[ImageConfig] = None
    ) -> GeneratedImage:
        """Generate an image from prompt."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Path,
        config: Optional[ImageConfig] = None
    ) -> List[GeneratedImage]:
        """Generate multiple images."""
        pass


class ProviderRegistry:
    """Registry of available image generation providers."""

    _providers: dict = {}

    @classmethod
    def register(cls, provider_type: ImageProvider, provider_class: type):
        """Register a provider implementation."""
        cls._providers[provider_type] = provider_class

    @classmethod
    def get(cls, provider_type: ImageProvider) -> Optional[ImageGeneratorBase]:
        """Get a provider instance."""
        if provider_type in cls._providers:
            return cls._providers[provider_type]()
        return None

    @classmethod
    def list_available(cls) -> List[ImageProvider]:
        """List providers that are currently available."""
        available = []
        for provider_type, provider_class in cls._providers.items():
            try:
                instance = provider_class()
                if instance.is_available():
                    available.append(provider_type)
            except Exception:
                pass
        return available

    @classmethod
    def get_best_available(cls) -> Optional[ImageGeneratorBase]:
        """Get the best available provider (prefers local over API)."""
        # Priority order
        priority = [
            ImageProvider.LOCAL_SDXL,
            ImageProvider.LOCAL_SD15,
            ImageProvider.OPENAI,
            ImageProvider.REPLICATE,
            ImageProvider.STABILITY,
        ]

        for provider_type in priority:
            if provider_type in cls._providers:
                instance = cls._providers[provider_type]()
                if instance.is_available():
                    return instance
        return None
