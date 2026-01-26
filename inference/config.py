"""Inference configuration for LLM Safety Alignment Study.

This module defines model pairs and inference parameters for Bedrock inference.
All six models (3 base + 3 aligned) are configured with their HuggingFace IDs
and placeholders for Bedrock ARNs.

Security: No hardcoded credentials. Region loaded from environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelFamily(str, Enum):
    """Supported model families for the study."""

    QWEN3 = "qwen3"
    LLAMA31 = "llama31"
    MISTRAL = "mistral"


@dataclass(frozen=True)
class ModelPair:
    """A pair of base and aligned models from the same family.

    Attributes:
        family: Model family identifier (qwen3, llama31, mistral).
        base_model_id: HuggingFace model ID for the base (unaligned) model.
        aligned_model_id: HuggingFace model ID for the aligned model.
        base_arn: Optional AWS Bedrock ARN for the imported base model.
        aligned_arn: Optional AWS Bedrock ARN for the imported aligned model.
    """

    family: str
    base_model_id: str
    aligned_model_id: str
    base_arn: Optional[str] = None
    aligned_arn: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate model family."""
        valid_families = {f.value for f in ModelFamily}
        if self.family not in valid_families:
            raise ValueError(
                f"Invalid family '{self.family}'. "
                f"Must be one of: {', '.join(sorted(valid_families))}"
            )

    @property
    def has_base_arn(self) -> bool:
        """Check if base model ARN is configured."""
        return self.base_arn is not None and len(self.base_arn) > 0

    @property
    def has_aligned_arn(self) -> bool:
        """Check if aligned model ARN is configured."""
        return self.aligned_arn is not None and len(self.aligned_arn) > 0

    @property
    def is_ready(self) -> bool:
        """Check if both ARNs are configured for inference."""
        return self.has_base_arn and self.has_aligned_arn

    def with_arns(
        self,
        base_arn: Optional[str] = None,
        aligned_arn: Optional[str] = None
    ) -> ModelPair:
        """Create a new ModelPair with updated ARNs.

        Args:
            base_arn: New base model ARN (uses existing if None).
            aligned_arn: New aligned model ARN (uses existing if None).

        Returns:
            New ModelPair instance with updated ARNs.
        """
        return ModelPair(
            family=self.family,
            base_model_id=self.base_model_id,
            aligned_model_id=self.aligned_model_id,
            base_arn=base_arn if base_arn is not None else self.base_arn,
            aligned_arn=aligned_arn if aligned_arn is not None else self.aligned_arn,
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for Bedrock inference operations.

    Attributes:
        max_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature (0.0 for deterministic).
        checkpoint_interval: Number of prompts between checkpoint saves.
        region: AWS region for Bedrock API calls.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    max_tokens: int = 128
    temperature: float = 0.0
    checkpoint_interval: int = 500
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_base_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_tokens > 4096:
            raise ValueError("max_tokens must not exceed 4096")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_base_delay <= 0:
            raise ValueError("retry_base_delay must be positive")


# =============================================================================
# Model Pair Definitions
# =============================================================================

# Import ARNs from model_arns.py (populated after Bedrock import completes)
try:
    from inference.model_arns import (
        QWEN3_BASE_ARN,
        QWEN3_INSTRUCT_ARN,
        MISTRAL_BASE_ARN,
        MISTRAL_INSTRUCT_ARN,
        LLAMA31_BASE_ARN,
        LLAMA31_INSTRUCT_ARN,
    )
except ImportError:
    # Fallback to None if model_arns.py doesn't exist yet
    QWEN3_BASE_ARN = None
    QWEN3_INSTRUCT_ARN = None
    MISTRAL_BASE_ARN = None
    MISTRAL_INSTRUCT_ARN = None
    LLAMA31_BASE_ARN = None
    LLAMA31_INSTRUCT_ARN = None


# Qwen 3 model pair (8B parameters)
QWEN3_PAIR = ModelPair(
    family=ModelFamily.QWEN3.value,
    base_model_id="Qwen/Qwen3-8B-Base",
    aligned_model_id="Qwen/Qwen3-8B",
    base_arn=QWEN3_BASE_ARN,
    aligned_arn=QWEN3_INSTRUCT_ARN,
)

# Llama 3.1 model pair (8B parameters)
LLAMA31_PAIR = ModelPair(
    family=ModelFamily.LLAMA31.value,
    base_model_id="meta-llama/Llama-3.1-8B",
    aligned_model_id="meta-llama/Llama-3.1-8B-Instruct",
    base_arn=LLAMA31_BASE_ARN,
    aligned_arn=LLAMA31_INSTRUCT_ARN,
)

# Mistral model pair (7B parameters)
MISTRAL_PAIR = ModelPair(
    family=ModelFamily.MISTRAL.value,
    base_model_id="mistralai/Mistral-7B-v0.3",
    aligned_model_id="mistralai/Mistral-7B-Instruct-v0.3",
    base_arn=MISTRAL_BASE_ARN,
    aligned_arn=MISTRAL_INSTRUCT_ARN,
)

# All model pairs indexed by family
MODEL_PAIRS: dict[str, ModelPair] = {
    ModelFamily.QWEN3.value: QWEN3_PAIR,
    ModelFamily.LLAMA31.value: LLAMA31_PAIR,
    ModelFamily.MISTRAL.value: MISTRAL_PAIR,
}

# Default inference configuration
DEFAULT_CONFIG = InferenceConfig()


def get_model_pair(family: str) -> ModelPair:
    """Get model pair by family name.

    Args:
        family: Model family identifier (qwen3, llama31, mistral).

    Returns:
        ModelPair for the specified family.

    Raises:
        KeyError: If family is not found.
    """
    family_lower = family.lower()
    if family_lower not in MODEL_PAIRS:
        valid = ", ".join(sorted(MODEL_PAIRS.keys()))
        raise KeyError(f"Unknown model family '{family}'. Valid families: {valid}")
    return MODEL_PAIRS[family_lower]


def get_all_model_pairs() -> list[ModelPair]:
    """Get all configured model pairs.

    Returns:
        List of all ModelPair instances.
    """
    return list(MODEL_PAIRS.values())


def get_inference_config(
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    checkpoint_interval: Optional[int] = None,
    region: Optional[str] = None,
) -> InferenceConfig:
    """Create inference configuration with optional overrides.

    Args:
        max_tokens: Override max_tokens (default: 128).
        temperature: Override temperature (default: 0.0).
        checkpoint_interval: Override checkpoint_interval (default: 500).
        region: Override region (default: from AWS_REGION env var or us-east-1).

    Returns:
        InferenceConfig with specified or default values.
    """
    return InferenceConfig(
        max_tokens=max_tokens if max_tokens is not None else 128,
        temperature=temperature if temperature is not None else 0.0,
        checkpoint_interval=checkpoint_interval if checkpoint_interval is not None else 500,
        region=region if region is not None else os.environ.get("AWS_REGION", "us-east-1"),
    )
