"""Bedrock inference client for LLM Safety Alignment Study.

This module provides a wrapper around AWS Bedrock's invoke_model API
with retry logic, error handling, and latency tracking.

Security:
- No hardcoded credentials (uses boto3 credential chain)
- Region loaded from environment variable
- Error messages sanitized (no stack traces returned)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from inference.config import DEFAULT_CONFIG, InferenceConfig


logger = logging.getLogger(__name__)


class BedrockInferenceError(Exception):
    """Base exception for Bedrock inference errors."""

    pass


class BedrockThrottlingError(BedrockInferenceError):
    """Raised when request is throttled by AWS."""

    pass


class BedrockModelError(BedrockInferenceError):
    """Raised when model returns an error."""

    pass


class BedrockInferenceClient:
    """Client for AWS Bedrock model inference.

    Provides methods for generating text completions from Bedrock-hosted
    models with automatic retry logic and latency tracking.

    Attributes:
        region: AWS region for API calls.
        config: Inference configuration parameters.

    Example:
        >>> client = BedrockInferenceClient()
        >>> text, latency = client.generate(model_arn, "Hello, world!")
        >>> print(f"Generated: {text} in {latency:.2f}ms")
    """

    def __init__(
        self,
        region: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        """Initialize Bedrock client.

        Args:
            region: AWS region. If None, uses config.region or AWS_REGION env var.
            config: Inference configuration. If None, uses DEFAULT_CONFIG.

        Note:
            Credentials are loaded from the boto3 credential chain:
            1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            2. Shared credential file (~/.aws/credentials)
            3. AWS config file (~/.aws/config)
            4. IAM role (for EC2/Lambda/ECS)
        """
        self._config = config or DEFAULT_CONFIG
        self._region = region or self._config.region

        # Configure boto3 with retries disabled (we handle retries ourselves)
        boto_config = Config(
            region_name=self._region,
            connect_timeout=self._config.timeout_seconds,
            read_timeout=self._config.timeout_seconds,
            retries={"max_attempts": 0},
        )

        self._client = boto3.client(
            "bedrock-runtime",
            config=boto_config,
        )

        logger.info(
            "BedrockInferenceClient initialized for region=%s",
            self._region,
        )

    @property
    def region(self) -> str:
        """Get the configured AWS region."""
        return self._region

    @property
    def config(self) -> InferenceConfig:
        """Get the inference configuration."""
        return self._config

    def generate(
        self,
        model_arn: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Generate completion from a Bedrock model.

        Args:
            model_arn: ARN of the imported Bedrock model.
            prompt: Input text to complete.
            max_tokens: Override max tokens (default: from config).
            temperature: Override temperature (default: from config).

        Returns:
            Tuple of (completion_text, latency_ms).
            Returns ("", latency_ms) on failure.

        Raises:
            BedrockThrottlingError: If request is throttled (for retry handling).
            BedrockModelError: If model returns an error response.
            BedrockInferenceError: For other inference failures.
        """
        if not model_arn:
            logger.error("Empty model ARN provided")
            return "", 0.0

        if not prompt:
            logger.warning("Empty prompt provided, returning empty completion")
            return "", 0.0

        effective_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        effective_temperature = temperature if temperature is not None else self._config.temperature

        # Build request body for Bedrock invoke_model
        # Using generic inference parameters compatible with imported models
        request_body = {
            "prompt": prompt,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temperature,
        }

        start_time = time.perf_counter()

        try:
            response = self._client.invoke_model(
                modelId=model_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse response body
            response_body = json.loads(response["body"].read())

            # Extract completion text (handle different response formats)
            completion = self._extract_completion(response_body)

            logger.debug(
                "Generated completion for model=%s, latency=%.2fms, tokens=%d",
                model_arn,
                latency_ms,
                len(completion.split()),
            )

            return completion, latency_ms

        except ClientError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", "Unknown error")

            # Check for throttling errors (including cold start)
            if error_code in ("ThrottlingException", "TooManyRequestsException", "ModelNotReadyException"):
                logger.warning(
                    "Request throttled/not ready for model=%s: %s",
                    model_arn,
                    error_message,
                )
                raise BedrockThrottlingError(f"Throttled: {error_code}") from e

            # Check for model errors
            if error_code in ("ModelErrorException", "ValidationException"):
                logger.error(
                    "Model error for model=%s: %s",
                    model_arn,
                    error_message,
                )
                raise BedrockModelError(f"Model error: {error_code}") from e

            # Log other errors and return empty
            logger.error(
                "Bedrock API error for model=%s: %s - %s",
                model_arn,
                error_code,
                error_message,
            )
            return "", latency_ms

        except BotoCoreError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Boto core error for model=%s: %s",
                model_arn,
                type(e).__name__,
            )
            return "", latency_ms

        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Failed to parse response for model=%s: JSON decode error",
                model_arn,
            )
            return "", latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            # Log error type but not full details (security)
            logger.exception(
                "Unexpected error for model=%s: %s",
                model_arn,
                type(e).__name__,
            )
            return "", latency_ms

    def generate_with_retry(
        self,
        model_arn: str,
        prompt: str,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Generate completion with exponential backoff retry logic.

        Retries on throttling errors with exponential backoff.
        Does not retry on model errors or other failures.

        Args:
            model_arn: ARN of the imported Bedrock model.
            prompt: Input text to complete.
            max_retries: Override max retries (default: from config).
            max_tokens: Override max tokens (default: from config).
            temperature: Override temperature (default: from config).

        Returns:
            Tuple of (completion_text, total_latency_ms).
            Returns ("", latency_ms) if all retries exhausted.
        """
        effective_max_retries = max_retries if max_retries is not None else self._config.max_retries

        total_latency_ms = 0.0
        last_error: Optional[Exception] = None

        for attempt in range(effective_max_retries + 1):
            try:
                completion, latency_ms = self.generate(
                    model_arn=model_arn,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                total_latency_ms += latency_ms

                # Success - return result
                if attempt > 0:
                    logger.info(
                        "Succeeded after %d retries for model=%s",
                        attempt,
                        model_arn,
                    )

                return completion, total_latency_ms

            except BedrockThrottlingError as e:
                last_error = e
                total_latency_ms += 0  # Latency already captured in generate()

                if attempt < effective_max_retries:
                    # Calculate exponential backoff delay
                    delay = self._config.retry_base_delay * (2 ** attempt)

                    logger.info(
                        "Retry %d/%d for model=%s after %.2fs delay",
                        attempt + 1,
                        effective_max_retries,
                        model_arn,
                        delay,
                    )

                    time.sleep(delay)
                else:
                    logger.warning(
                        "Max retries (%d) exhausted for model=%s",
                        effective_max_retries,
                        model_arn,
                    )

            except (BedrockModelError, BedrockInferenceError) as e:
                # Don't retry on model errors or other inference errors
                logger.error(
                    "Non-retryable error for model=%s: %s",
                    model_arn,
                    type(e).__name__,
                )
                return "", total_latency_ms

        # All retries exhausted
        return "", total_latency_ms

    def _extract_completion(self, response_body: dict) -> str:
        """Extract completion text from response body.

        Handles different response formats from various model types.

        Args:
            response_body: Parsed JSON response from Bedrock.

        Returns:
            Extracted completion text, or empty string if not found.
        """
        # Try common response formats

        # Format 1: OpenAI-compatible "choices" format (Bedrock imported models)
        if "choices" in response_body and isinstance(response_body["choices"], list):
            choices = response_body["choices"]
            if choices:
                choice = choices[0]
                # Try message.content first (chat completion format)
                if "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content", "")
                    if content:
                        return str(content).strip()
                # Try text field (completion format)
                if "text" in choice:
                    return str(choice["text"]).strip()
                # Try delta.content (streaming format)
                if "delta" in choice and isinstance(choice["delta"], dict):
                    content = choice["delta"].get("content", "")
                    if content:
                        return str(content).strip()

        # Format 2: Direct "completion" field
        if "completion" in response_body:
            return str(response_body["completion"]).strip()

        # Format 3: "generated_text" field
        if "generated_text" in response_body:
            return str(response_body["generated_text"]).strip()

        # Format 4: "outputs" list with "text" field
        if "outputs" in response_body and isinstance(response_body["outputs"], list):
            outputs = response_body["outputs"]
            if outputs and "text" in outputs[0]:
                return str(outputs[0]["text"]).strip()

        # Format 5: "generation" field (some models)
        if "generation" in response_body:
            return str(response_body["generation"]).strip()

        # Format 6: "results" list
        if "results" in response_body and isinstance(response_body["results"], list):
            results = response_body["results"]
            if results and "outputText" in results[0]:
                return str(results[0]["outputText"]).strip()

        # Format 7: "text" field directly
        if "text" in response_body:
            return str(response_body["text"]).strip()

        logger.warning(
            "Could not extract completion from response. Keys: %s",
            list(response_body.keys()),
        )
        return ""

    def close(self) -> None:
        """Close the client and release resources.

        Note: boto3 clients don't require explicit cleanup,
        but this method is provided for consistency.
        """
        logger.debug("BedrockInferenceClient closed")

    def __enter__(self) -> BedrockInferenceClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def health_check(self) -> bool:
        """Check if the client can connect to Bedrock.

        Attempts to list foundation models to verify connectivity.

        Returns:
            True if connection is healthy, False otherwise.
        """
        try:
            # Use bedrock client (not bedrock-runtime) for listing models
            bedrock_client = boto3.client(
                "bedrock",
                region_name=self._region,
            )

            # Simple API call to verify connectivity
            bedrock_client.list_imported_models(maxResults=1)

            logger.debug("Health check passed for region=%s", self._region)
            return True

        except Exception as e:
            logger.warning(
                "Health check failed for region=%s: %s",
                self._region,
                type(e).__name__,
            )
            return False
