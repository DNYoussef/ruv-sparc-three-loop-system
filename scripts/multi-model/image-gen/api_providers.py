"""
API-based image generation providers.
Supports OpenAI DALL-E, Replicate, Stability AI, etc.
"""

import os
import time
import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import urlparse


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

try:
    from .base import (
        ImageGeneratorBase,
        ImageProvider,
        ImageConfig,
        GeneratedImage,
        ProviderRegistry
    )
except ImportError:
    from base import (
        ImageGeneratorBase,
        ImageProvider,
        ImageConfig,
        GeneratedImage,
        ProviderRegistry
    )


class OpenAIGenerator(ImageGeneratorBase):
    """Generate images using OpenAI DALL-E 3."""

    provider = ImageProvider.OPENAI

    def __init__(self):
        self._client = None
        self._api_key = os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        """Check if OpenAI API key is set."""
        return self._api_key is not None

    def setup(self) -> bool:
        """Verify OpenAI API access."""
        if not self._api_key:
            print("OPENAI_API_KEY environment variable not set")
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            return True
        except ImportError:
            print("openai package not installed. Run: pip install openai")
            return False

    def generate(
        self,
        prompt: str,
        output_path: Path,
        config: Optional[ImageConfig] = None
    ) -> GeneratedImage:
        """Generate image using DALL-E 3."""
        config = config or ImageConfig()

        if self._client is None:
            self.setup()

        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)

        start_time = time.time()

        # Map size to DALL-E supported sizes
        size = "1024x1024"
        if config.width == 1792 and config.height == 1024:
            size = "1792x1024"
        elif config.width == 1024 and config.height == 1792:
            size = "1024x1792"

        response = self._client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
            response_format="b64_json"
        )

        generation_time = time.time() - start_time

        # Save image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_data = base64.b64decode(response.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return GeneratedImage(
            path=output_path,
            prompt=prompt,
            provider=self.provider,
            config=config,
            generation_time_seconds=generation_time
        )

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Path,
        config: Optional[ImageConfig] = None
    ) -> List[GeneratedImage]:
        """Generate multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, prompt in enumerate(prompts):
            output_path = output_dir / f"image_{i+1}.png"
            result = self.generate(prompt, output_path, config)
            results.append(result)
            print(f"Generated {i+1}/{len(prompts)}: {output_path}")

        return results


class ReplicateGenerator(ImageGeneratorBase):
    """Generate images using Replicate API."""

    provider = ImageProvider.REPLICATE

    # Default to SDXL Lightning on Replicate
    MODEL = "bytedance/sdxl-lightning-4step:727e49a643e999d602a896c774a0658ffefea21465756a6ce24b7ea4165ber0c"

    def __init__(self):
        self._client = None
        self._api_key = os.environ.get("REPLICATE_API_TOKEN")

    def is_available(self) -> bool:
        """Check if Replicate API key is set."""
        return self._api_key is not None

    def setup(self) -> bool:
        """Verify Replicate API access."""
        if not self._api_key:
            print("REPLICATE_API_TOKEN environment variable not set")
            return False

        try:
            import replicate
            return True
        except ImportError:
            print("replicate package not installed. Run: pip install replicate")
            return False

    def generate(
        self,
        prompt: str,
        output_path: Path,
        config: Optional[ImageConfig] = None
    ) -> GeneratedImage:
        """Generate image using Replicate."""
        config = config or ImageConfig()

        import replicate
        import urllib.request

        start_time = time.time()

        output = replicate.run(
            self.MODEL,
            input={
                "prompt": prompt,
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.num_inference_steps,
            }
        )

        generation_time = time.time() - start_time

        # Download and save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(output, list) and len(output) > 0:
            urllib.request.urlretrieve(output[0], output_path)

        return GeneratedImage(
            path=output_path,
            prompt=prompt,
            provider=self.provider,
            config=config,
            generation_time_seconds=generation_time
        )

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Path,
        config: Optional[ImageConfig] = None
    ) -> List[GeneratedImage]:
        """Generate multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, prompt in enumerate(prompts):
            output_path = output_dir / f"image_{i+1}.png"
            result = self.generate(prompt, output_path, config)
            results.append(result)
            print(f"Generated {i+1}/{len(prompts)}: {output_path}")

        return results


ATLAS_API_BASE_URL = "https://api.atlascloud.ai/api/v1"
ATLAS_CATALOG_URL = f"{ATLAS_API_BASE_URL}/models"
ATLAS_DEFAULT_MODEL = "openai/gpt-image-2/text-to-image"
ATLAS_SUCCESS_STATUSES = {"completed", "succeeded", "success"}
ATLAS_FAILURE_STATUSES = {"failed", "canceled", "cancelled", "error"}


class AtlasConfirmationRequired(RuntimeError):
    """Raised after preflight when a paid Atlas submission is not confirmed."""


def _atlas_json_objects(value: Any) -> List[Dict[str, Any]]:
    """Return all objects nested in a JSON-compatible response."""
    objects: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        objects.append(value)
        for child in value.values():
            objects.extend(_atlas_json_objects(child))
    elif isinstance(value, list):
        for child in value:
            objects.extend(_atlas_json_objects(child))
    return objects


def _atlas_response_ok(payload: Dict[str, Any]) -> bool:
    return str(payload.get("code")) == "200"


def _atlas_get_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    attempts: int = 1,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Fetch JSON with bounded retries. This helper is never used for POSTs."""
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "context-cascade/atlas-image-provider",
        **(headers or {}),
    }
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            req = url_request.Request(url, headers=request_headers, method="GET")
            with url_request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload
        except (OSError, ValueError, url_error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                sleep_fn(float(2**attempt))
    raise RuntimeError(f"Atlas GET failed after {attempts} attempts: {last_error}")


def _atlas_post_json_once(
    url: str,
    *,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Submit exactly one generation request without automatic retries."""
    req = url_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "context-cascade/atlas-image-provider",
        },
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Atlas submit failed ({exc.code}): {detail}") from exc
    if not isinstance(result, dict):
        raise ValueError("Atlas submit returned a non-object response")
    return result


def _atlas_download(
    url: str,
) -> bytes:
    """Download an HTTPS output exactly once."""
    if urlparse(url).scheme != "https":
        raise ValueError("Atlas output URL must use HTTPS")
    req = url_request.Request(
        url,
        headers={"User-Agent": "context-cascade/atlas-image-provider"},
        method="GET",
    )
    with url_request.urlopen(req, timeout=300) as response:
        return response.read()


class AtlasGenerator(ImageGeneratorBase):
    """Generate one image through an explicit, guarded Atlas Cloud request."""

    provider = ImageProvider.ATLAS

    def __init__(self):
        self._api_key = os.environ.get("ATLASCLOUD_API_KEY", "").strip()
        self._model = os.environ.get("ATLASCLOUD_IMAGE_MODEL", "").strip()
        self._model = self._model or ATLAS_DEFAULT_MODEL

    def is_available(self) -> bool:
        """Check whether Atlas credentials are configured."""
        return bool(self._api_key)

    def setup(self) -> bool:
        """Validate local configuration without making a paid request."""
        if not self._api_key:
            print("ATLASCLOUD_API_KEY environment variable not set")
            return False
        return True

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _preflight(
        self,
        prompt: str,
        output_path: Path,
        config: ImageConfig,
    ) -> Dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("ATLASCLOUD_API_KEY environment variable not set")

        catalog = _atlas_get_json(ATLAS_CATALOG_URL, headers=self._auth_headers())
        if not _atlas_response_ok(catalog):
            raise RuntimeError(f"Atlas catalog returned code {catalog.get('code')}")
        matches = [
            item
            for item in _atlas_json_objects(catalog)
            if item.get("model") == self._model and item.get("display_console") is True
        ]
        if len(matches) != 1:
            raise ValueError(f"Atlas model is unavailable or ambiguous: {self._model}")
        model_entry = matches[0]

        schema_url = model_entry.get("schema") or model_entry.get("schema_url")
        if not isinstance(schema_url, str) or urlparse(schema_url).scheme != "https":
            raise ValueError("Atlas catalog is missing an HTTPS model schema URL")
        schema = _atlas_get_json(schema_url)

        suffix = output_path.suffix.lower()
        if suffix == ".png":
            output_format = "png"
        elif suffix in {".jpg", ".jpeg"}:
            output_format = "jpeg"
        else:
            raise ValueError("Atlas output path must end in .png, .jpg, or .jpeg")

        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": f"{config.width}x{config.height}",
            "quality": "medium",
            "output_format": output_format,
            "moderation": "low",
        }
        input_schema = schema.get("components", {}).get("schemas", {}).get("Input", {})
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        if not isinstance(properties, dict):
            raise ValueError("Atlas model schema is missing Input.properties")
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Atlas request is missing fields: {', '.join(missing)}")
        unsupported = sorted(set(payload).difference(set(properties).union({"model"})))
        if unsupported:
            raise ValueError(f"Atlas request has unsupported fields: {', '.join(unsupported)}")
        for key, value in payload.items():
            choices = properties.get(key, {}).get("enum")
            if isinstance(choices, list) and choices and value not in choices:
                rendered = ", ".join(str(choice) for choice in choices)
                raise ValueError(f"Atlas {key} must be one of: {rendered}")

        price_data = model_entry.get("price")
        actual_price = price_data.get("actual") if isinstance(price_data, dict) else None
        price = actual_price.get("base_price", "unknown") if isinstance(actual_price, dict) else "unknown"
        print(
            f"Atlas preflight: model={self._model} size={payload['size']} "
            f"format={output_format} unit_price={price}"
        )
        return payload

    def generate(
        self,
        prompt: str,
        output_path: Path,
        config: Optional[ImageConfig] = None,
    ) -> GeneratedImage:
        """Preflight, submit once, poll with GETs, and save one image."""
        config = config or ImageConfig()
        output_path = Path(output_path)
        start_time = time.time()
        payload = self._preflight(prompt, output_path, config)
        if not config.confirm_paid:
            raise AtlasConfirmationRequired(
                "Atlas generation is paid. Review the plan and rerun with --yes."
            )

        submitted = _atlas_post_json_once(
            f"{ATLAS_API_BASE_URL}/model/generateImage",
            api_key=self._api_key,
            payload=payload,
        )
        if not _atlas_response_ok(submitted):
            raise RuntimeError(f"Atlas submit returned code {submitted.get('code')}")
        data = submitted.get("data")
        prediction_id = data.get("id") if isinstance(data, dict) else None
        if not prediction_id:
            raise ValueError("Atlas submit response is missing a prediction id")

        completed: Optional[Dict[str, Any]] = None
        for attempt in range(20):
            prediction = _atlas_get_json(
                f"{ATLAS_API_BASE_URL}/model/prediction/{prediction_id}",
                headers=self._auth_headers(),
                attempts=4,
            )
            if not _atlas_response_ok(prediction):
                raise RuntimeError(f"Atlas prediction returned code {prediction.get('code')}")
            prediction_data = prediction.get("data")
            status = (
                str(prediction_data.get("status", "")).lower()
                if isinstance(prediction_data, dict)
                else ""
            )
            if status in ATLAS_SUCCESS_STATUSES:
                completed = prediction
                break
            if status in ATLAS_FAILURE_STATUSES:
                detail = prediction_data.get("error") if isinstance(prediction_data, dict) else None
                raise RuntimeError(f"Atlas prediction {status}: {detail or 'no detail'}")
            if attempt < 19:
                time.sleep(float(min(2**attempt, 8)))
        if completed is None:
            raise TimeoutError("Atlas prediction did not complete after 20 polls")

        completed_data = completed.get("data")
        raw_outputs = completed_data.get("outputs") if isinstance(completed_data, dict) else None
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ValueError("Atlas prediction completed without output URLs")
        output_url = raw_outputs[0]
        if isinstance(output_url, dict):
            output_url = output_url.get("url")
        if not isinstance(output_url, str) or not output_url:
            raise ValueError("Atlas prediction returned an invalid output URL")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_atlas_download(output_url))
        return GeneratedImage(
            path=output_path,
            prompt=prompt,
            provider=self.provider,
            config=config,
            generation_time_seconds=time.time() - start_time,
        )

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: Path,
        config: Optional[ImageConfig] = None,
    ) -> List[GeneratedImage]:
        """Generate prompts sequentially; each prompt has one non-retried POST."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.generate(prompt, output_dir / f"image_{index + 1}.png", config)
            for index, prompt in enumerate(prompts)
        ]


# Register providers
ProviderRegistry.register(ImageProvider.OPENAI, OpenAIGenerator)
ProviderRegistry.register(ImageProvider.REPLICATE, ReplicateGenerator)
ProviderRegistry.register(ImageProvider.ATLAS, AtlasGenerator)
