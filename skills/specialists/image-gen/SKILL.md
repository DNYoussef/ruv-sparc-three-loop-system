---
name: image-gen
description: Modular image generation - supports local SDXL Lightning, OpenAI DALL-E, Replicate, optional Atlas Cloud, or custom providers
allowed-tools: Bash, Read, Write, Task, TodoWrite, Glob, Grep
---

# Modular Image Generation Skill



---

## LIBRARY-FIRST PROTOCOL (MANDATORY)

**Before writing ANY code, you MUST check:**

### Step 1: Library Catalog
- Location: `.claude/library/catalog.json`
- If match >70%: REUSE or ADAPT

### Step 2: Patterns Guide
- Location: `.claude/docs/inventories/LIBRARY-PATTERNS-GUIDE.md`
- If pattern exists: FOLLOW documented approach

### Step 3: Existing Projects
- Location: `D:\Projects\*`
- If found: EXTRACT and adapt

### Decision Matrix
| Match | Action |
|-------|--------|
| Library >90% | REUSE directly |
| Library 70-90% | ADAPT minimally |
| Pattern exists | FOLLOW pattern |
| In project | EXTRACT |
| No match | BUILD (add to library after) |

---

## Purpose

Generate images using the best available provider - local models (free, private) or cloud APIs (fast, paid). Fully modular architecture allows plugging in any image generation backend.

## Providers

| Provider | Type | Cost | Requirements | Quality |
|----------|------|------|--------------|---------|
| **SDXL Lightning** | Local | Free | 8GB VRAM, ~7GB disk | Excellent |
| **OpenAI DALL-E 3** | API | ~$0.04/image | OPENAI_API_KEY | Excellent |
| **Replicate** | API | ~$0.01/image | REPLICATE_API_TOKEN | Good |
| **Atlas Cloud** | API | Live price preflight | ATLASCLOUD_API_KEY | Excellent |
| **Custom** | Any | Varies | User-defined | Varies |

## When to Use

### Perfect For:
- Blog banners and social media images (LinkedIn: 1200x630)
- Documentation diagrams and illustrations
- UI mockups and wireframes
- Concept visualization
- Any image generation need

### Provider Selection:
- **Privacy required?** -> Use local SDXL
- **No GPU?** -> Use OpenAI or Replicate API
- **Batch generation?** -> Use local (no API costs)
- **Highest quality?** -> DALL-E 3 or SDXL Lightning

## Quick Start

### 1. Check Available Providers
```bash
python scripts/multi-model/image-gen/cli.py --list
```

### 2. Setup Local SDXL (Recommended)
```bash
# First-time setup (downloads ~7GB)
python scripts/multi-model/image-gen/cli.py --setup local
```

### 3. Generate Images
```bash
# Auto-selects best available provider
python scripts/multi-model/image-gen/cli.py "A sunset over mountains" output.png

# LinkedIn banner size
python scripts/multi-model/image-gen/cli.py "Tech concept" banner.png --width 1200 --height 630

# Specific provider
python scripts/multi-model/image-gen/cli.py "A cat" cat.png --provider openai

# Atlas is explicit-only: preflight the live catalog/schema/price, then submit once
python scripts/multi-model/image-gen/cli.py "A cat" cat.png --provider atlas --yes
```

## Integration with Visual Art Composition

For professional-quality images, combine with `visual-art-composition`:

```
Step 1: visual-art-composition (Structure the prompt)
    |
    +---> 13-dimension aesthetic framework
    +---> Cross-cultural synthesis
    +---> Productive tension resolution
    |
    v
Step 2: image-gen (Generate the image)
    |
    +---> Select best provider (local or API)
    +---> Generate high-quality image
    +---> Save to specified path
```

### Example Pipeline
```bash
# 1. Get structured prompt from visual-art-composition
/visual-art-composition "tech dashboard for productivity app"

# 2. Generate with structured prompt
python scripts/multi-model/image-gen/cli.py \
  "Dashboard UI with linear perspective depth, composed blues and warm golds,
   focal hierarchy with clear primary metric, notan two-value contrast.
   Modern professional aesthetic, clean geometric forms." \
  docs/images/dashboard.png --width 1200 --height 630
```

## Provider Setup

### Local SDXL Lightning (Recommended)

**Requirements:**
- GPU with 8GB+ VRAM (or CPU with 16GB+ RAM, slower)
- ~7GB disk space on D: drive
- Python with diffusers, torch

**Setup:**
```bash
python scripts/multi-model/image-gen/cli.py --setup local
```

**Environment Variables (optional):**
```bash
export SDXL_MODEL_DIR="D:/AI-Models/sdxl-lightning"
```

### OpenAI DALL-E 3

**Requirements:**
- OpenAI API key
- ~$0.04 per image

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
python scripts/multi-model/image-gen/cli.py --setup openai
```

### Replicate

**Requirements:**
- Replicate API token
- ~$0.01 per image

**Setup:**
```bash
export REPLICATE_API_TOKEN="r8_..."
python scripts/multi-model/image-gen/cli.py --setup replicate
```

### Atlas Cloud (Explicit Opt-In)

Atlas never participates in automatic provider selection. It fetches the live
model catalog and referenced schema, validates the request, and prints the
current unit price before submission. `--yes` is required for the single paid
generation POST; that POST is never retried. Only prediction GETs use bounded
retries; catalog, schema, and output download requests are single-attempt.

```bash
export ATLASCLOUD_API_KEY="..."
python scripts/multi-model/image-gen/cli.py --setup atlas
python scripts/multi-model/image-gen/cli.py \
  "A cinematic mountain landscape" output.png --provider atlas --yes
```

The default model is `openai/gpt-image-2/text-to-image`. Override it with
`ATLASCLOUD_IMAGE_MODEL`; the override must still be uniquely available in the
live catalog. Atlas output paths must end in `.png`, `.jpg`, or `.jpeg`.

## Adding Custom Providers

Create a new provider by implementing `ImageGeneratorBase`:

```python
from base import ImageGeneratorBase, ImageProvider, ProviderRegistry

class MyCustomGenerator(ImageGeneratorBase):
    provider = ImageProvider.CUSTOM

    def is_available(self) -> bool:
        # Check if provider is configured
        return True

    def setup(self) -> bool:
        # Download models, verify API keys, etc.
        return True

    def generate(self, prompt, output_path, config=None):
        # Generate image
        # Return GeneratedImage
        pass

# Register
ProviderRegistry.register(ImageProvider.CUSTOM, MyCustomGenerator)
```

## Python API

```python
from scripts.multi_model.image_gen.base import ProviderRegistry, ImageConfig

# Get best available provider
provider = ProviderRegistry.get_best_available()

# Configure
config = ImageConfig(
    width=1200,
    height=630,
    num_inference_steps=4
)

# Generate
result = provider.generate(
    prompt="A beautiful sunset",
    output_path="output.png",
    config=config
)

print(f"Generated: {result.path} in {result.generation_time_seconds}s")
```

## Batch Generation

```python
prompts = [
    "Sunset over mountains",
    "City skyline at night",
    "Forest in autumn"
]

results = provider.generate_batch(
    prompts=prompts,
    output_dir="./images/",
    config=config
)
```

## Best Practices

### Prompt Engineering
1. Be specific about composition, colors, style
2. Include negative prompts for local models
3. Use visual-art-composition for professional quality
4. Specify aspect ratio in prompt when needed

### Performance
1. Local models: First generation is slow (model loading), subsequent are fast
2. API models: Consistent speed, watch for rate limits
3. Batch generation: More efficient than individual calls
4. Atlas: Review the live price plan before adding `--yes`; generation POSTs are not retried

### Quality
1. SDXL Lightning: 4 steps is optimal (more steps = minimal improvement)
2. DALL-E 3: No step control, always high quality
3. Always validate output matches intent

## Related Skills

- `visual-art-composition`: 13-dimension aesthetic framework for structured prompts
- `prompt-architect`: General prompt optimization
- `pptx-generation`: Uses images for presentation slides

## Troubleshooting

### "No provider available"
- Run `--list` to see what's configured
- Run `--setup local` to download SDXL Lightning
- Or set API keys for cloud providers

### Out of VRAM
- Use CPU mode (slower): Set `SDXL_DEVICE=cpu`
- Use API provider instead
- Reduce image size

### Slow First Generation
- Normal for local models (loading ~7GB model)
- Subsequent generations are fast (~2-5 seconds)

### Poor Quality
- Use more descriptive prompts
- Apply visual-art-composition framework
- Try different provider

## Files

- CLI: `scripts/multi-model/image-gen/cli.py`
- Base classes: `scripts/multi-model/image-gen/base.py`
- Local SDXL: `scripts/multi-model/image-gen/local_sdxl.py`
- API providers: `scripts/multi-model/image-gen/api_providers.py`
