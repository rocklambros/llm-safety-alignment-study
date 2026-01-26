"""
Bedrock Model ARNs for LLM Safety Alignment Study.

This file contains the actual ARNs for imported Bedrock models.
All 6 models successfully imported on 2026-01-24.

Last updated: 2026-01-24
"""

from __future__ import annotations

# AWS Account and Region
AWS_ACCOUNT_ID = "685384916687"
AWS_REGION = "us-east-1"

# =============================================================================
# Imported Model ARNs - ALL COMPLETE
# =============================================================================

# Qwen 3 8B
QWEN3_BASE_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/rjej1fiapf3w"
QWEN3_INSTRUCT_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/1gxx8kwco0dv"

# Mistral 7B
MISTRAL_BASE_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/m8dg0atc7g4k"
MISTRAL_INSTRUCT_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/ezfalgh7llyk"

# Llama 3.1 8B
LLAMA31_BASE_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/jppl6xp0gd9s"
LLAMA31_INSTRUCT_ARN = "arn:aws:bedrock:us-east-1:685384916687:imported-model/7jocmdvq7qjk"


# =============================================================================
# Model Pairs for Inference
# =============================================================================

MODEL_PAIRS = {
    "qwen3": {
        "base_arn": QWEN3_BASE_ARN,
        "instruct_arn": QWEN3_INSTRUCT_ARN,
        "base_name": "qwen3-8b-base",
        "instruct_name": "qwen3-8b-instruct",
    },
    "mistral": {
        "base_arn": MISTRAL_BASE_ARN,
        "instruct_arn": MISTRAL_INSTRUCT_ARN,
        "base_name": "mistral-7b-base",
        "instruct_name": "mistral-7b-instruct",
    },
    "llama31": {
        "base_arn": LLAMA31_BASE_ARN,
        "instruct_arn": LLAMA31_INSTRUCT_ARN,
        "base_name": "llama31-8b-base",
        "instruct_name": "llama31-8b-instruct",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_ready_pairs() -> dict[str, tuple[str, str]]:
    """Get all model pairs (base_arn, aligned_arn).

    Returns:
        Dict mapping family name to (base_arn, aligned_arn) tuple.
    """
    return {
        family: (info["base_arn"], info["instruct_arn"])
        for family, info in MODEL_PAIRS.items()
    }


def get_model_status() -> dict[str, dict]:
    """Get status of all models.

    Returns:
        Dict with status for each model family.
    """
    return {
        "qwen3": {
            "base": "ready",
            "instruct": "ready",
            "pair_ready": True,
        },
        "mistral": {
            "base": "ready",
            "instruct": "ready",
            "pair_ready": True,
        },
        "llama31": {
            "base": "ready",
            "instruct": "ready",
            "pair_ready": True,
        },
    }


def print_status() -> None:
    """Print current model import status."""
    print("=" * 60)
    print("Bedrock Model Import Status - ALL COMPLETE")
    print("=" * 60)

    for family, info in MODEL_PAIRS.items():
        print(f"\n{family.upper()}: ✅ READY")
        print(f"  Base:     ✅ {info['base_name']}")
        print(f"  Instruct: ✅ {info['instruct_name']}")

    print(f"\n{'-' * 60}")
    print(f"Ready for inference: 3 of 3 model pairs")
    print("=" * 60)


if __name__ == "__main__":
    print_status()
