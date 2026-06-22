from src.config import settings
from src.gemini_client import get_gemini_api_keys


def test_get_gemini_api_keys_deduplicates_rotation_pool():
    original_key = settings.gemini_api_key
    original_keys = settings.gemini_api_keys
    try:
        settings.gemini_api_key = "key-a"
        settings.gemini_api_keys = "key-b,key-a\nkey-c"

        assert get_gemini_api_keys() == ["key-b", "key-a", "key-c"]
    finally:
        settings.gemini_api_key = original_key
        settings.gemini_api_keys = original_keys


def test_get_gemini_api_keys_supports_single_key():
    original_key = settings.gemini_api_key
    original_keys = settings.gemini_api_keys
    try:
        settings.gemini_api_key = "key-a"
        settings.gemini_api_keys = ""

        assert get_gemini_api_keys() == ["key-a"]
    finally:
        settings.gemini_api_key = original_key
        settings.gemini_api_keys = original_keys
