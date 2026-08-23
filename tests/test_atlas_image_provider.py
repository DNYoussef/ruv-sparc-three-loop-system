import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


IMAGE_GEN_DIR = Path(__file__).parents[1] / "scripts" / "multi-model" / "image-gen"
sys.path.insert(0, str(IMAGE_GEN_DIR))

import api_providers  # noqa: E402
from base import ImageConfig, ImageProvider, ProviderRegistry  # noqa: E402


MODEL = api_providers.ATLAS_DEFAULT_MODEL
SCHEMA_URL = "https://static.example.test/atlas-schema.json"


def catalog():
    return {
        "code": 200,
        "data": {
            "groups": [
                {
                    "models": [
                        {
                            "model": MODEL,
                            "display_console": True,
                            "schema": SCHEMA_URL,
                            "price": {"actual": {"base_price": "0.009"}},
                        }
                    ]
                }
            ]
        },
    }


def schema():
    return {
        "components": {
            "schemas": {
                "Input": {
                    "required": ["model", "prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "size": {"enum": ["1024x1024"]},
                        "quality": {"enum": ["medium"]},
                        "output_format": {"enum": ["png", "jpeg"]},
                        "moderation": {"type": "string"},
                    },
                }
            }
        }
    }


class AtlasImageProviderTests(unittest.TestCase):
    def generator(self):
        with mock.patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
            return api_providers.AtlasGenerator()

    def test_atlas_is_explicit_and_not_auto_selected(self):
        class AvailableAtlas:
            def is_available(self):
                return True

        with mock.patch.object(
            ProviderRegistry,
            "_providers",
            {ImageProvider.ATLAS: AvailableAtlas},
        ):
            self.assertIsNone(ProviderRegistry.get_best_available())
            self.assertIsInstance(ProviderRegistry.get(ImageProvider.ATLAS), AvailableAtlas)

    @mock.patch.object(api_providers, "_atlas_post_json_once")
    @mock.patch.object(api_providers, "_atlas_get_json")
    def test_unconfirmed_preflight_never_posts(self, get_json, post_json):
        get_json.side_effect = [catalog(), schema()]

        with self.assertRaises(api_providers.AtlasConfirmationRequired):
            self.generator().generate(
                "test prompt",
                Path("result.png"),
                ImageConfig(confirm_paid=False),
            )

        post_json.assert_not_called()

    @mock.patch.object(api_providers, "_atlas_post_json_once")
    @mock.patch.object(api_providers, "_atlas_get_json")
    def test_schema_rejects_unsupported_size_before_post(self, get_json, post_json):
        get_json.side_effect = [catalog(), schema()]

        with self.assertRaisesRegex(ValueError, "Atlas size must be one of"):
            self.generator().generate(
                "test prompt",
                Path("result.png"),
                ImageConfig(width=1200, height=630, confirm_paid=True),
            )

        post_json.assert_not_called()

    @mock.patch.object(api_providers.time, "sleep", return_value=None)
    @mock.patch.object(api_providers, "_atlas_download", return_value=b"image-bytes")
    @mock.patch.object(api_providers, "_atlas_post_json_once")
    @mock.patch.object(api_providers, "_atlas_get_json")
    def test_confirmed_generation_posts_once_then_polls_get(
        self,
        get_json,
        post_json,
        download,
        _sleep,
    ):
        get_json.side_effect = [
            catalog(),
            schema(),
            {"code": 200, "data": {"status": "processing"}},
            {
                "code": 200,
                "data": {
                    "status": "completed",
                    "outputs": ["https://cdn.example.test/result.png"],
                },
            },
        ]
        post_json.return_value = {"code": 200, "data": {"id": "prediction-1"}}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.png"
            result = self.generator().generate(
                "test prompt",
                output,
                ImageConfig(confirm_paid=True),
            )

            self.assertEqual(result.path, output)
            self.assertEqual(output.read_bytes(), b"image-bytes")

        post_json.assert_called_once()
        self.assertEqual(get_json.call_count, 4)
        self.assertNotIn("attempts", get_json.call_args_list[0].kwargs)
        self.assertNotIn("attempts", get_json.call_args_list[1].kwargs)
        self.assertEqual(get_json.call_args_list[2].kwargs["attempts"], 4)
        self.assertEqual(get_json.call_args_list[3].kwargs["attempts"], 4)
        download.assert_called_once_with("https://cdn.example.test/result.png")


if __name__ == "__main__":
    unittest.main()
