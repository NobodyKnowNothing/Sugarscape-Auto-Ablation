import os
import sys
import time
from unittest.mock import MagicMock
from pathlib import Path

from edit_harness import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    is_resource_exhausted_error,
    ModelFailoverManager,
    generate_content_with_failover,
)
from google.genai import errors


def test_error_classification():
    print("Testing is_resource_exhausted_error...")
    # 429 ClientError
    e429 = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}})
    assert is_resource_exhausted_error(e429), "Expected e429 to be classified as resource exhausted"

    # String-based 429 error
    estr = Exception("429 Resource has been exhausted (e.g. check quota).")
    assert is_resource_exhausted_error(estr), "Expected string 429 to be classified as resource exhausted"

    # Rate limit error string
    erate = RuntimeError("Rate limit reached for requests per minute")
    assert is_resource_exhausted_error(erate), "Expected rate limit to be classified as resource exhausted"

    # 400 ClientError (invalid parameter) - MUST NOT BE CLASSIFIED AS EXHAUSTED
    e400 = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Bad prompt"}})
    assert not is_resource_exhausted_error(e400), "Expected 400 to NOT be classified as resource exhausted"

    # 500 ServerError - MUST NOT BE CLASSIFIED AS EXHAUSTED
    e500 = errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL", "message": "Internal error"}})
    assert not is_resource_exhausted_error(e500), "Expected 500 to NOT be classified as resource exhausted"

    # ValueError / syntax - MUST NOT BE CLASSIFIED AS EXHAUSTED
    assert not is_resource_exhausted_error(ValueError("Bad syntax")), "Expected ValueError to not be exhausted"
    print("  -> test_error_classification PASSED")


def test_failover_on_429():
    print("Testing failover on 429 Resource Exhausted...")
    manager = ModelFailoverManager(
        primary_model="gemma-4-31b-it",
        fallback_model="gemma-4-26b-a4b-it",
        cooldown_seconds=1.0,
    )

    mock_client = MagicMock()
    mock_primary_resp = MagicMock(text="Response from 31B")
    mock_fallback_resp = MagicMock(text="Response from 26B")

    # Primary succeeds
    mock_client.models.generate_content.return_value = mock_primary_resp
    resp, model_used = manager.generate_content(mock_client, contents="hello")
    assert resp.text == "Response from 31B"
    assert model_used == "gemma-4-31b-it"

    # Primary raises 429
    e429 = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota limit"}})
    def side_effect(model, contents, config=None):
        if model == "gemma-4-31b-it":
            raise e429
        elif model == "gemma-4-26b-a4b-it":
            return mock_fallback_resp
        raise ValueError("Unknown model")

    mock_client.models.generate_content.side_effect = side_effect
    logs = []
    resp, model_used = manager.generate_content(mock_client, contents="hello", log_fn=logs.append)
    assert resp.text == "Response from 26B"
    assert model_used == "gemma-4-26b-a4b-it"
    assert manager.is_primary_exhausted()
    assert any("EXHAUSTED" in m for m in logs)
    print("  -> test_failover_on_429 PASSED")


def test_no_failover_on_non_quota_error():
    print("Testing NO failover on non-quota errors (400 Bad Request)...")
    manager = ModelFailoverManager(
        primary_model="gemma-4-31b-it",
        fallback_model="gemma-4-26b-a4b-it",
        cooldown_seconds=1.0,
    )

    mock_client = MagicMock()
    e400 = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Invalid prompt format"}})
    mock_client.models.generate_content.side_effect = e400

    caught = False
    try:
        manager.generate_content(mock_client, contents="bad")
    except errors.ClientError as exc:
        caught = True
        assert exc.code == 400

    assert caught, "Expected 400 error to be re-raised"
    assert not manager.is_primary_exhausted(), "Primary should NOT be marked exhausted on 400 error"
    print("  -> test_no_failover_on_non_quota_error PASSED")


def test_recovery_after_cooldown():
    print("Testing quota recovery after cooldown...")
    manager = ModelFailoverManager(
        primary_model="gemma-4-31b-it",
        fallback_model="gemma-4-26b-a4b-it",
        cooldown_seconds=0.2,
    )

    mock_client = MagicMock()
    mock_primary_resp = MagicMock(text="31B OK")
    mock_fallback_resp = MagicMock(text="26B OK")

    # Trigger exhaustion
    manager.mark_primary_exhausted(cooldown=0.2)
    assert manager.is_primary_exhausted()

    # During cooldown -> goes to fallback directly
    mock_client.models.generate_content.return_value = mock_fallback_resp
    logs = []
    resp, model_used = manager.generate_content(mock_client, contents="hi", log_fn=logs.append)
    assert model_used == "gemma-4-26b-a4b-it"
    assert any("cooldown" in m.lower() for m in logs)

    # Wait for cooldown
    time.sleep(0.3)
    assert not manager.is_primary_exhausted()

    # Next call tries primary and succeeds
    mock_client.models.generate_content.return_value = mock_primary_resp
    logs.clear()
    resp, model_used = manager.generate_content(mock_client, contents="hi", log_fn=logs.append)
    assert model_used == "gemma-4-31b-it"
    assert any("recovered" in m.lower() for m in logs)
    print("  -> test_recovery_after_cooldown PASSED")


def test_live_default_model_generation():
    print("Testing live generation using default Gemma 4 31B model...")
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("  Skipping live API call: no API key.")
        return

    from google import genai
    client = genai.Client(api_key=key)

    logs = []
    resp, model_used = generate_content_with_failover(
        client=client,
        contents="Say '31B verified' in 2 words.",
        log_fn=logs.append,
    )
    print(f"  Live call response (model: {model_used}): {resp.text.strip()}")
    assert resp.text
    print("  -> test_live_default_model_generation PASSED")


if __name__ == "__main__":
    test_error_classification()
    test_failover_on_429()
    test_no_failover_on_non_quota_error()
    test_recovery_after_cooldown()
    test_live_default_model_generation()
    print("\nALL UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
