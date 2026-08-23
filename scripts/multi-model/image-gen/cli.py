#!/usr/bin/env python3
"""
Modular Image Generation CLI
Supports: Local SDXL Lightning, OpenAI DALL-E, Replicate, etc.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports

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

sys.path.insert(0, str(Path(__file__).parent))

try:
    from .base import ImageProvider, ImageConfig, ProviderRegistry
    from . import local_sdxl
    from . import api_providers
except ImportError:
    from base import ImageProvider, ImageConfig, ProviderRegistry
    import local_sdxl
    import api_providers


def list_providers():
    """List all available providers."""
    print("Available Image Generation Providers:")
    print("=" * 50)

    available = ProviderRegistry.list_available()

    all_providers = [
        (ImageProvider.LOCAL_SDXL, "SDXL Lightning (local)", "Free, ~7GB, 8GB VRAM"),
        (ImageProvider.OPENAI, "OpenAI DALL-E 3", "Paid API, OPENAI_API_KEY"),
        (ImageProvider.REPLICATE, "Replicate API", "Paid API, REPLICATE_API_TOKEN"),
        (ImageProvider.ATLAS, "Atlas Cloud", "Paid API, ATLASCLOUD_API_KEY; explicit --yes"),
    ]

    for provider, name, note in all_providers:
        status = "[OK]" if provider in available else "[NOT CONFIGURED]"
        print(f"  {status} {name}")
        print(f"         {note}")
        print()


def setup_provider(provider_name: str):
    """Set up a specific provider."""
    provider_map = {
        "local": ImageProvider.LOCAL_SDXL,
        "sdxl": ImageProvider.LOCAL_SDXL,
        "openai": ImageProvider.OPENAI,
        "dalle": ImageProvider.OPENAI,
        "replicate": ImageProvider.REPLICATE,
        "atlas": ImageProvider.ATLAS,
    }

    provider_type = provider_map.get(provider_name.lower())
    if not provider_type:
        print(f"Unknown provider: {provider_name}")
        print(f"Available: {list(provider_map.keys())}")
        return False

    provider = ProviderRegistry.get(provider_type)
    if provider:
        print(f"Setting up {provider_type.value}...")
        return provider.setup()
    return False


def generate_image(
    prompt: str,
    output: str,
    provider_name: str = "auto",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    confirm_paid: bool = False,
):
    """Generate an image."""

    # Get provider
    if provider_name == "auto":
        provider = ProviderRegistry.get_best_available()
        if not provider:
            print("No image generation provider available!")
            print("Run: python cli.py --setup local")
            return False
    else:
        provider_map = {
            "local": ImageProvider.LOCAL_SDXL,
            "sdxl": ImageProvider.LOCAL_SDXL,
            "openai": ImageProvider.OPENAI,
            "replicate": ImageProvider.REPLICATE,
            "atlas": ImageProvider.ATLAS,
        }
        provider_type = provider_map.get(provider_name.lower())
        provider = ProviderRegistry.get(provider_type)

    if not provider:
        print(f"Provider not available: {provider_name}")
        return False

    if not provider.is_available():
        print(f"Provider not configured. Run: python cli.py --setup {provider_name}")
        return False

    # Configure
    config = ImageConfig(
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=0.0 if provider.provider == ImageProvider.LOCAL_SDXL else 7.5,
        confirm_paid=confirm_paid,
    )

    # Generate
    print(f"Generating with {provider.provider.value}...")
    print(f"Prompt: {prompt[:100]}...")

    try:
        result = provider.generate(prompt, Path(output), config)
    except api_providers.AtlasConfirmationRequired as exc:
        print(str(exc))
        return False

    print(f"Generated: {result.path}")
    print(f"Time: {result.generation_time_seconds:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Modular Image Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available providers
  python cli.py --list

  # Setup local SDXL Lightning
  python cli.py --setup local

  # Generate image (auto-selects best provider)
  python cli.py "A beautiful sunset" output.png

  # Generate with specific provider
  python cli.py "A cat" cat.png --provider openai

  # Atlas performs a live catalog/schema/price preflight before one paid submit
  python cli.py "A cat" cat.png --provider atlas --yes

  # Generate LinkedIn banner size
  python cli.py "Tech concept" banner.png --width 1200 --height 630
"""
    )

    parser.add_argument("prompt", nargs="?", help="Image prompt")
    parser.add_argument("output", nargs="?", help="Output file path")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument(
        "--setup",
        metavar="PROVIDER",
        help="Setup a provider (local, openai, replicate, atlas)",
    )
    parser.add_argument("--provider", "-p", default="auto", help="Provider to use (default: auto)")
    parser.add_argument("--width", "-W", type=int, default=1024, help="Image width")
    parser.add_argument("--height", "-H", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", "-s", type=int, default=4, help="Inference steps")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the live Atlas price plan and submit once",
    )

    args = parser.parse_args()

    if args.list:
        list_providers()
        return 0

    if args.setup:
        success = setup_provider(args.setup)
        return 0 if success else 1

    if not args.prompt or not args.output:
        parser.print_help()
        return 1

    success = generate_image(
        args.prompt,
        args.output,
        provider_name=args.provider,
        width=args.width,
        height=args.height,
        steps=args.steps,
        confirm_paid=args.yes,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
