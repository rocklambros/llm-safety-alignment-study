#!/usr/bin/env python3
"""
Test script for Bedrock inference validation.

Runs a single prompt through all 6 models to verify:
1. Bedrock connectivity
2. Model ARNs are correct
3. Inference produces valid completions

Usage:
    python inference/test_inference.py
"""

from __future__ import annotations

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.bedrock_client import BedrockInferenceClient, BedrockInferenceError
from inference.config import (
    MODEL_PAIRS,
    get_model_pair,
    get_inference_config,
)
from inference.model_arns import print_status as print_model_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test prompt - neutral, non-toxic
TEST_PROMPT = "The weather today is"


def test_single_model(client: BedrockInferenceClient, model_name: str, model_arn: str, max_retries: int = 5) -> bool:
    """Test inference on a single model.

    Args:
        client: Bedrock inference client.
        model_name: Human-readable model name.
        model_arn: Bedrock model ARN.
        max_retries: Maximum retry attempts for cold start warmup.

    Returns:
        True if inference succeeded, False otherwise.
    """
    print(f"\n  Testing {model_name}...")
    print(f"    ARN: {model_arn}")

    try:
        # Use more retries for cold start warmup
        completion, latency_ms = client.generate_with_retry(
            model_arn=model_arn,
            prompt=TEST_PROMPT,
            max_tokens=32,  # Short completion for testing
            max_retries=max_retries,  # More retries for cold start
        )

        if completion:
            # Truncate for display
            display_text = completion[:80] + "..." if len(completion) > 80 else completion
            print(f"    ✅ Success ({latency_ms:.0f}ms)")
            print(f"    Response: \"{display_text}\"")
            return True
        else:
            print(f"    ❌ Empty response (latency: {latency_ms:.0f}ms)")
            return False

    except BedrockInferenceError as e:
        print(f"    ❌ Inference error: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {type(e).__name__}: {e}")
        return False


def main() -> int:
    """Main test function.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 60)
    print("Bedrock Inference Test")
    print("=" * 60)

    # Print model status
    print("\nModel Status:")
    print_model_status()

    # Create client
    print("\nInitializing Bedrock client...")
    config = get_inference_config(max_tokens=32, temperature=0.0)
    client = BedrockInferenceClient(config=config)

    # Health check
    print("\nRunning health check...")
    if client.health_check():
        print("  ✅ Bedrock connectivity OK")
    else:
        print("  ❌ Bedrock connectivity FAILED")
        print("  Check AWS credentials and region settings.")
        return 1

    # Test each model pair
    print(f"\nTest prompt: \"{TEST_PROMPT}\"")

    results = {}
    families = ["qwen3", "mistral", "llama31"]

    for family in families:
        print(f"\n{'─' * 50}")
        print(f"Testing {family.upper()} family:")

        try:
            model_pair = get_model_pair(family)

            if not model_pair.is_ready:
                print(f"  ⚠️ Skipping - ARNs not configured")
                results[f"{family}_base"] = None
                results[f"{family}_aligned"] = None
                continue

            # Test base model
            base_success = test_single_model(
                client,
                f"{family}-base",
                model_pair.base_arn,
            )
            results[f"{family}_base"] = base_success

            # Test aligned model
            aligned_success = test_single_model(
                client,
                f"{family}-aligned",
                model_pair.aligned_arn,
            )
            results[f"{family}_aligned"] = aligned_success

        except Exception as e:
            print(f"  ❌ Error testing family: {type(e).__name__}: {e}")
            results[f"{family}_base"] = False
            results[f"{family}_aligned"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\n✅ All tests PASSED - Ready for full inference!")
        return 0
    elif failed > 0:
        print(f"\n❌ {failed} tests FAILED - Check model ARNs and permissions")
        return 1
    else:
        print("\n⚠️ No tests run - Configure model ARNs first")
        return 1


if __name__ == "__main__":
    sys.exit(main())
