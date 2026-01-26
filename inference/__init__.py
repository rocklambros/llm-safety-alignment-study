"""Inference module for LLM Safety Alignment Study.

This module provides configuration, client, and orchestration classes for running
inference against AWS Bedrock-hosted language models.

Classes:
    ModelPair: Configuration for a base/aligned model pair.
    InferenceConfig: Parameters for inference operations.
    BedrockInferenceClient: AWS Bedrock API wrapper with retry logic.
    InferenceRunner: Orchestrates inference with checkpointing and parallel execution.
    CompletionRecord: Data class for inference output records.
    InferenceSummary: Summary statistics for inference runs.

Constants:
    MODEL_PAIRS: Dict mapping family names to ModelPair instances.
    DEFAULT_CONFIG: Default InferenceConfig instance.
    QWEN3_PAIR: Qwen 3 model pair (8B).
    LLAMA31_PAIR: Llama 3.1 model pair (8B).
    MISTRAL_PAIR: Mistral model pair (7B).

Functions:
    get_model_pair: Get ModelPair by family name.
    get_all_model_pairs: Get all configured model pairs.
    get_inference_config: Create InferenceConfig with optional overrides.
    load_prompts: Load prompts from CSV or JSON file.

Example:
    >>> from inference import BedrockInferenceClient, get_model_pair, DEFAULT_CONFIG
    >>>
    >>> # Get model pair configuration
    >>> qwen = get_model_pair("qwen3")
    >>> print(f"Base model: {qwen.base_model_id}")
    >>>
    >>> # Create client and generate completion
    >>> client = BedrockInferenceClient()
    >>> if qwen.has_base_arn:
    ...     text, latency = client.generate_with_retry(qwen.base_arn, "Hello")
    ...     print(f"Generated: {text}")

    >>> # Run inference with checkpointing
    >>> from inference import InferenceRunner, load_prompts
    >>> prompts = load_prompts("data/processed/prompt_sample_25k.csv")
    >>> runner = InferenceRunner(DEFAULT_CONFIG, client, dry_run=True)
    >>> summary = runner.run_all(prompts, "output/completions/")
"""

from inference.config import (
    ModelFamily,
    ModelPair,
    InferenceConfig,
    MODEL_PAIRS,
    DEFAULT_CONFIG,
    QWEN3_PAIR,
    LLAMA31_PAIR,
    MISTRAL_PAIR,
    get_model_pair,
    get_all_model_pairs,
    get_inference_config,
)

from inference.bedrock_client import (
    BedrockInferenceClient,
    BedrockInferenceError,
    BedrockThrottlingError,
    BedrockModelError,
)

from inference.inference_runner import (
    InferenceRunner,
    CompletionRecord,
    InferenceSummary,
    load_prompts,
    load_prompts_csv,
    load_prompts_json,
)


__all__ = [
    # Configuration classes
    "ModelFamily",
    "ModelPair",
    "InferenceConfig",
    # Model pair constants
    "MODEL_PAIRS",
    "DEFAULT_CONFIG",
    "QWEN3_PAIR",
    "LLAMA31_PAIR",
    "MISTRAL_PAIR",
    # Configuration functions
    "get_model_pair",
    "get_all_model_pairs",
    "get_inference_config",
    # Client classes
    "BedrockInferenceClient",
    # Exceptions
    "BedrockInferenceError",
    "BedrockThrottlingError",
    "BedrockModelError",
    # Inference runner
    "InferenceRunner",
    "CompletionRecord",
    "InferenceSummary",
    # Data loading
    "load_prompts",
    "load_prompts_csv",
    "load_prompts_json",
]

__version__ = "1.0.0"
