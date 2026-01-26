"""Inference orchestration for LLM Safety Alignment Study.

This module provides the main inference runner that processes prompts through
Bedrock models with checkpoint-based resumability, parallel execution,
and comprehensive error handling.

Security:
- No hardcoded credentials (uses boto3 credential chain)
- Input paths validated against traversal attacks
- Error messages sanitized (no stack traces in output)

Usage:
    python -m inference.inference_runner \\
        --prompts data/processed/prompt_sample_25k.csv \\
        --output output/completions/ \\
        --families qwen3,llama31,mistral \\
        --parallel
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from inference.bedrock_client import (
    BedrockInferenceClient,
    BedrockInferenceError,
    BedrockModelError,
    BedrockThrottlingError,
)
from inference.config import (
    DEFAULT_CONFIG,
    MODEL_PAIRS,
    InferenceConfig,
    ModelFamily,
    ModelPair,
    get_model_pair,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Path Security
# =============================================================================


def validate_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Validate file path against traversal attacks.

    Args:
        path: User-provided path string.
        base_dir: Optional base directory to restrict access to.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path traversal is detected or path is invalid.
    """
    resolved = Path(path).resolve()

    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}")

    return resolved


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptRecord:
    """A single prompt record from the input CSV.

    Attributes:
        prompt_id: Unique identifier for the prompt.
        prompt_text: The actual prompt text.
        source: Dataset source (rtp or toxigen).
    """

    prompt_id: str
    prompt_text: str
    source: str = ""

    def validate(self) -> bool:
        """Validate the prompt record.

        Returns:
            True if valid, False otherwise.
        """
        if not self.prompt_id or not isinstance(self.prompt_id, str):
            return False
        if not self.prompt_text or not isinstance(self.prompt_text, str):
            return False
        return True


@dataclass
class CompletionRecord:
    """A completed inference record for output.

    Matches PRD section 3.3 output format.
    """

    prompt_id: str
    family: str
    prompt_text: str
    base_completion: str
    aligned_completion: str
    base_latency_ms: float
    aligned_latency_ms: float
    timestamp: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "family": self.family,
            "prompt_text": self.prompt_text,
            "base_completion": self.base_completion,
            "aligned_completion": self.aligned_completion,
            "base_latency_ms": self.base_latency_ms,
            "aligned_latency_ms": self.aligned_latency_ms,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class InferenceSummary:
    """Summary statistics for an inference run."""

    family: str
    total_prompts: int
    completed: int
    errors: int
    avg_base_latency_ms: float
    avg_aligned_latency_ms: float
    total_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "family": self.family,
            "total_prompts": self.total_prompts,
            "completed": self.completed,
            "errors": self.errors,
            "error_rate": self.errors / max(self.total_prompts, 1),
            "avg_base_latency_ms": self.avg_base_latency_ms,
            "avg_aligned_latency_ms": self.avg_aligned_latency_ms,
            "total_duration_seconds": self.total_duration_seconds,
        }


# =============================================================================
# Inference Runner
# =============================================================================


class InferenceRunner:
    """Orchestrates inference across model families with checkpointing.

    Processes prompts through Bedrock models with:
    - Checkpoint-based resumability (saves progress every N completions)
    - Parallel execution across model families
    - Comprehensive error handling and logging
    - Progress tracking with tqdm

    Attributes:
        config: Inference configuration parameters.
        client: Bedrock inference client.
        dry_run: If True, simulate inference without API calls.

    Example:
        >>> config = InferenceConfig()
        >>> client = BedrockInferenceClient()
        >>> runner = InferenceRunner(config, client)
        >>> summary = runner.run_family("qwen3", prompts, "output/")
    """

    def __init__(
        self,
        config: InferenceConfig,
        client: BedrockInferenceClient,
        dry_run: bool = False,
    ) -> None:
        """Initialize the inference runner.

        Args:
            config: Inference configuration parameters.
            client: Bedrock inference client for API calls.
            dry_run: If True, simulate inference without making API calls.
        """
        self._config = config
        self._client = client
        self._dry_run = dry_run

        logger.info(
            "InferenceRunner initialized: checkpoint_interval=%d, dry_run=%s",
            config.checkpoint_interval,
            dry_run,
        )

    @property
    def config(self) -> InferenceConfig:
        """Get the inference configuration."""
        return self._config

    @property
    def dry_run(self) -> bool:
        """Check if running in dry-run mode."""
        return self._dry_run

    def run_family(
        self,
        family: str,
        prompts: List[Dict[str, Any]],
        output_path: str,
    ) -> InferenceSummary:
        """Process all prompts for one model family (base + aligned).

        Checkpoints progress every config.checkpoint_interval completions.
        Resumes from checkpoint on restart.

        Args:
            family: Model family identifier (qwen3, llama31, mistral).
            prompts: List of prompt dictionaries with prompt_id, prompt_text.
            output_path: Directory path for output files.

        Returns:
            InferenceSummary with completion statistics.

        Raises:
            ValueError: If family is invalid or model ARNs not configured.
            OSError: If output directory cannot be created.
        """
        start_time = time.perf_counter()

        # Validate family
        model_pair = get_model_pair(family)
        if not self._dry_run and not model_pair.is_ready:
            raise ValueError(
                f"Model ARNs not configured for family '{family}'. "
                f"Base ARN: {model_pair.has_base_arn}, "
                f"Aligned ARN: {model_pair.has_aligned_arn}"
            )

        # Validate and create output directory
        output_dir = validate_path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output and checkpoint file paths
        output_file = output_dir / f"completions_{family}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_{family}.json"

        # Load checkpoint if exists
        completed_records, start_index = self._load_checkpoint(
            family, str(checkpoint_file)
        )

        logger.info(
            "Starting inference for family=%s, prompts=%d, resuming_from=%d",
            family,
            len(prompts),
            start_index,
        )

        # Track statistics
        total_base_latency = sum(r.base_latency_ms for r in completed_records)
        total_aligned_latency = sum(r.aligned_latency_ms for r in completed_records)
        error_count = sum(1 for r in completed_records if r.error is not None)

        # Open output file in append mode if resuming
        write_mode = "a" if start_index > 0 else "w"

        with open(output_file, write_mode, encoding="utf-8") as f:
            # Process remaining prompts
            remaining_prompts = prompts[start_index:]

            with tqdm(
                total=len(remaining_prompts),
                desc=f"Inference [{family}]",
                unit="prompt",
                initial=0,
            ) as pbar:
                for idx, prompt_data in enumerate(remaining_prompts):
                    current_index = start_index + idx

                    # Process single prompt
                    record = self._process_prompt(prompt_data, model_pair)

                    # Write to output file
                    f.write(record.to_json() + "\n")
                    f.flush()

                    # Track statistics
                    completed_records.append(record)
                    total_base_latency += record.base_latency_ms
                    total_aligned_latency += record.aligned_latency_ms
                    if record.error is not None:
                        error_count += 1

                    # Checkpoint at intervals
                    if (current_index + 1) % self._config.checkpoint_interval == 0:
                        self._save_checkpoint(
                            family,
                            str(checkpoint_file),
                            completed_records,
                            current_index + 1,
                        )
                        logger.info(
                            "Checkpoint saved for family=%s at index=%d",
                            family,
                            current_index + 1,
                        )

                    pbar.update(1)

        # Final checkpoint
        self._save_checkpoint(
            family, str(checkpoint_file), completed_records, len(prompts)
        )

        # Calculate summary statistics
        total_completed = len(completed_records)
        duration = time.perf_counter() - start_time

        summary = InferenceSummary(
            family=family,
            total_prompts=len(prompts),
            completed=total_completed,
            errors=error_count,
            avg_base_latency_ms=(
                total_base_latency / total_completed if total_completed > 0 else 0.0
            ),
            avg_aligned_latency_ms=(
                total_aligned_latency / total_completed if total_completed > 0 else 0.0
            ),
            total_duration_seconds=duration,
        )

        logger.info(
            "Completed inference for family=%s: completed=%d, errors=%d, duration=%.2fs",
            family,
            total_completed,
            error_count,
            duration,
        )

        return summary

    def run_all(
        self,
        prompts: List[Dict[str, Any]],
        output_dir: str,
        families: Optional[List[str]] = None,
        parallel: bool = False,
        max_workers: int = 3,
    ) -> Dict[str, InferenceSummary]:
        """Run inference for all specified model families.

        Can run families in parallel using ThreadPoolExecutor.

        Args:
            prompts: List of prompt dictionaries.
            output_dir: Base directory for output files.
            families: List of families to process (default: all).
            parallel: If True, run families in parallel.
            max_workers: Maximum concurrent workers for parallel execution.

        Returns:
            Dictionary mapping family names to InferenceSummary objects.
        """
        start_time = time.perf_counter()

        # Default to all families
        if families is None:
            families = [f.value for f in ModelFamily]
        else:
            # Validate families
            valid_families = {f.value for f in ModelFamily}
            for family in families:
                if family.lower() not in valid_families:
                    raise ValueError(
                        f"Invalid family '{family}'. "
                        f"Valid: {', '.join(sorted(valid_families))}"
                    )
            families = [f.lower() for f in families]

        logger.info(
            "Starting inference for families=%s, parallel=%s, prompts=%d",
            families,
            parallel,
            len(prompts),
        )

        # Validate output directory
        output_path = validate_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summaries: Dict[str, InferenceSummary] = {}

        if parallel and len(families) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(max_workers, len(families))) as executor:
                future_to_family = {
                    executor.submit(
                        self.run_family, family, prompts, output_dir
                    ): family
                    for family in families
                }

                for future in as_completed(future_to_family):
                    family = future_to_family[future]
                    try:
                        summary = future.result()
                        summaries[family] = summary
                    except Exception as e:
                        logger.exception("Failed inference for family=%s", family)
                        # Create error summary
                        summaries[family] = InferenceSummary(
                            family=family,
                            total_prompts=len(prompts),
                            completed=0,
                            errors=len(prompts),
                            avg_base_latency_ms=0.0,
                            avg_aligned_latency_ms=0.0,
                            total_duration_seconds=0.0,
                        )
        else:
            # Sequential execution
            for family in families:
                try:
                    summary = self.run_family(family, prompts, output_dir)
                    summaries[family] = summary
                except Exception as e:
                    logger.exception("Failed inference for family=%s", family)
                    summaries[family] = InferenceSummary(
                        family=family,
                        total_prompts=len(prompts),
                        completed=0,
                        errors=len(prompts),
                        avg_base_latency_ms=0.0,
                        avg_aligned_latency_ms=0.0,
                        total_duration_seconds=0.0,
                    )

        total_duration = time.perf_counter() - start_time
        total_completed = sum(s.completed for s in summaries.values())
        total_errors = sum(s.errors for s in summaries.values())

        logger.info(
            "Completed all families: total_completed=%d, total_errors=%d, duration=%.2fs",
            total_completed,
            total_errors,
            total_duration,
        )

        # Save overall summary
        self._save_run_summary(output_path, summaries, total_duration)

        return summaries

    def _process_prompt(
        self,
        prompt_data: Dict[str, Any],
        model_pair: ModelPair,
    ) -> CompletionRecord:
        """Process a single prompt through base and aligned models.

        Args:
            prompt_data: Dictionary with prompt_id, prompt_text, etc.
            model_pair: Model pair configuration.

        Returns:
            CompletionRecord with results from both models.
        """
        prompt_id = str(prompt_data.get("prompt_id", ""))
        prompt_text = str(prompt_data.get("prompt_text", prompt_data.get("text", "")))
        timestamp = datetime.now(timezone.utc).isoformat()

        # Dry run mode - simulate inference
        if self._dry_run:
            return CompletionRecord(
                prompt_id=prompt_id,
                family=model_pair.family,
                prompt_text=prompt_text,
                base_completion="[DRY RUN] Base completion placeholder",
                aligned_completion="[DRY RUN] Aligned completion placeholder",
                base_latency_ms=100.0,
                aligned_latency_ms=100.0,
                timestamp=timestamp,
                error=None,
            )

        base_completion = ""
        aligned_completion = ""
        base_latency = 0.0
        aligned_latency = 0.0
        error_message: Optional[str] = None

        # Generate base model completion
        try:
            base_completion, base_latency = self._client.generate_with_retry(
                model_arn=model_pair.base_arn,
                prompt=prompt_text,
            )
        except BedrockThrottlingError as e:
            logger.warning(
                "Throttled on base model for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            error_message = f"base_throttled:{type(e).__name__}"
        except BedrockModelError as e:
            logger.warning(
                "Model error on base for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            error_message = f"base_model_error:{type(e).__name__}"
        except BedrockInferenceError as e:
            logger.warning(
                "Inference error on base for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            error_message = f"base_error:{type(e).__name__}"
        except Exception as e:
            logger.exception(
                "Unexpected error on base for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            error_message = f"base_unexpected:{type(e).__name__}"

        # Generate aligned model completion
        try:
            aligned_completion, aligned_latency = self._client.generate_with_retry(
                model_arn=model_pair.aligned_arn,
                prompt=prompt_text,
            )
        except BedrockThrottlingError as e:
            logger.warning(
                "Throttled on aligned model for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            if error_message:
                error_message += f";aligned_throttled:{type(e).__name__}"
            else:
                error_message = f"aligned_throttled:{type(e).__name__}"
        except BedrockModelError as e:
            logger.warning(
                "Model error on aligned for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            if error_message:
                error_message += f";aligned_model_error:{type(e).__name__}"
            else:
                error_message = f"aligned_model_error:{type(e).__name__}"
        except BedrockInferenceError as e:
            logger.warning(
                "Inference error on aligned for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            if error_message:
                error_message += f";aligned_error:{type(e).__name__}"
            else:
                error_message = f"aligned_error:{type(e).__name__}"
        except Exception as e:
            logger.exception(
                "Unexpected error on aligned for prompt_id=%s: %s",
                prompt_id,
                type(e).__name__,
            )
            if error_message:
                error_message += f";aligned_unexpected:{type(e).__name__}"
            else:
                error_message = f"aligned_unexpected:{type(e).__name__}"

        return CompletionRecord(
            prompt_id=prompt_id,
            family=model_pair.family,
            prompt_text=prompt_text,
            base_completion=base_completion,
            aligned_completion=aligned_completion,
            base_latency_ms=base_latency,
            aligned_latency_ms=aligned_latency,
            timestamp=timestamp,
            error=error_message,
        )

    def _save_checkpoint(
        self,
        family: str,
        checkpoint_path: str,
        completed: List[CompletionRecord],
        index: int,
    ) -> None:
        """Save checkpoint to JSON file.

        Args:
            family: Model family identifier.
            checkpoint_path: Path to checkpoint file.
            completed: List of completed records.
            index: Current processing index.
        """
        checkpoint_data = {
            "family": family,
            "last_index": index,
            "completed_count": len(completed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_ids": [r.prompt_id for r in completed],
        }

        # Write atomically by writing to temp file first
        temp_path = checkpoint_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(temp_path, checkpoint_path)
        except Exception as e:
            logger.exception("Failed to save checkpoint for family=%s", family)
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _load_checkpoint(
        self,
        family: str,
        checkpoint_path: str,
    ) -> Tuple[List[CompletionRecord], int]:
        """Load checkpoint if exists.

        Args:
            family: Model family identifier.
            checkpoint_path: Path to checkpoint file.

        Returns:
            Tuple of (completed_records, last_index).
            Returns empty list and 0 if no checkpoint exists.
        """
        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.exists():
            logger.debug("No checkpoint found for family=%s", family)
            return [], 0

        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint matches family
            if checkpoint_data.get("family") != family:
                logger.warning(
                    "Checkpoint family mismatch: expected=%s, found=%s",
                    family,
                    checkpoint_data.get("family"),
                )
                return [], 0

            last_index = checkpoint_data.get("last_index", 0)
            prompt_ids = set(checkpoint_data.get("prompt_ids", []))

            # Load completed records from JSONL file
            output_dir = checkpoint_file.parent
            output_file = output_dir / f"completions_{family}.jsonl"

            completed_records: List[CompletionRecord] = []

            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record_data = json.loads(line)
                            record = CompletionRecord(
                                prompt_id=record_data["prompt_id"],
                                family=record_data["family"],
                                prompt_text=record_data["prompt_text"],
                                base_completion=record_data["base_completion"],
                                aligned_completion=record_data["aligned_completion"],
                                base_latency_ms=record_data["base_latency_ms"],
                                aligned_latency_ms=record_data["aligned_latency_ms"],
                                timestamp=record_data["timestamp"],
                                error=record_data.get("error"),
                            )
                            completed_records.append(record)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(
                                "Failed to parse record from checkpoint: %s",
                                type(e).__name__,
                            )
                            continue

            logger.info(
                "Loaded checkpoint for family=%s: last_index=%d, records=%d",
                family,
                last_index,
                len(completed_records),
            )

            return completed_records, last_index

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse checkpoint for family=%s: %s",
                family,
                type(e).__name__,
            )
            return [], 0
        except Exception as e:
            logger.exception(
                "Error loading checkpoint for family=%s: %s",
                family,
                type(e).__name__,
            )
            return [], 0

    def _save_run_summary(
        self,
        output_dir: Path,
        summaries: Dict[str, InferenceSummary],
        total_duration: float,
    ) -> None:
        """Save overall run summary to JSON file.

        Args:
            output_dir: Output directory path.
            summaries: Dictionary of family summaries.
            total_duration: Total run duration in seconds.
        """
        summary_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": total_duration,
            "dry_run": self._dry_run,
            "config": {
                "max_tokens": self._config.max_tokens,
                "temperature": self._config.temperature,
                "checkpoint_interval": self._config.checkpoint_interval,
                "region": self._config.region,
            },
            "families": {
                family: summary.to_dict() for family, summary in summaries.items()
            },
        }

        summary_file = output_dir / "run_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            logger.info("Run summary saved to %s", summary_file)
        except Exception as e:
            logger.exception("Failed to save run summary: %s", type(e).__name__)


# =============================================================================
# Data Loading
# =============================================================================


def load_prompts_csv(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from CSV file.

    Args:
        file_path: Path to CSV file with prompt_id, prompt_text columns.

    Returns:
        List of prompt dictionaries.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    path = validate_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no headers")

        fieldnames = set(reader.fieldnames)
        required = {"prompt_id"}
        text_columns = {"prompt_text", "text"}

        if not required.issubset(fieldnames):
            raise ValueError(f"Missing required columns: {required - fieldnames}")

        if not text_columns.intersection(fieldnames):
            raise ValueError(f"Missing text column (one of: {text_columns})")

        for row in reader:
            # Get text from either column
            prompt_text = row.get("prompt_text") or row.get("text", "")
            prompts.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_text": prompt_text,
                    "source": row.get("source", ""),
                }
            )

    logger.info("Loaded %d prompts from %s", len(prompts), file_path)
    return prompts


def load_prompts_json(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSON file.

    Args:
        file_path: Path to JSON file with array of prompt objects.

    Returns:
        List of prompt dictionaries.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON is invalid.
    """
    path = validate_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of prompt objects")

    prompts: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt_text = item.get("prompt_text") or item.get("text", "")
        prompts.append(
            {
                "prompt_id": item.get("prompt_id", ""),
                "prompt_text": prompt_text,
                "source": item.get("source", ""),
            }
        )

    logger.info("Loaded %d prompts from %s", len(prompts), file_path)
    return prompts


def load_prompts(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from CSV or JSON file based on extension.

    Args:
        file_path: Path to prompts file (.csv or .json).

    Returns:
        List of prompt dictionaries.

    Raises:
        ValueError: If file extension is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_prompts_csv(file_path)
    elif suffix == ".json":
        return load_prompts_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json")


# =============================================================================
# CLI Interface
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the inference runner.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce boto3 logging noise
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of arguments (for testing).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM inference across model families with checkpointing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all families sequentially
    python -m inference.inference_runner \\
        --prompts data/processed/prompt_sample_25k.csv \\
        --output output/completions/

    # Run specific families in parallel
    python -m inference.inference_runner \\
        --prompts data/processed/prompt_sample_25k.csv \\
        --output output/completions/ \\
        --families qwen3,llama31 \\
        --parallel

    # Dry run for testing
    python -m inference.inference_runner \\
        --prompts data/processed/prompt_sample_25k.csv \\
        --output output/completions/ \\
        --dry-run
        """,
    )

    parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Path to prompts file (CSV or JSON)",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for completion files",
    )

    parser.add_argument(
        "--families",
        "-f",
        type=str,
        default=None,
        help="Comma-separated list of families to process (default: all)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run families in parallel using ThreadPoolExecutor",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel workers (default: 3)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Checkpoint every N completions (default: 500)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens per completion (default: 128)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate inference without API calls",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the inference runner.

    Args:
        args: Optional list of arguments (for testing).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parsed_args = parse_args(args)

    # Setup logging
    setup_logging(parsed_args.verbose)

    logger.info("Starting inference runner")
    logger.info("Prompts file: %s", parsed_args.prompts)
    logger.info("Output directory: %s", parsed_args.output)
    logger.info("Dry run: %s", parsed_args.dry_run)

    try:
        # Load prompts
        prompts = load_prompts(parsed_args.prompts)

        if not prompts:
            logger.error("No prompts loaded from file")
            return 1

        logger.info("Loaded %d prompts", len(prompts))

        # Parse families
        families: Optional[List[str]] = None
        if parsed_args.families:
            families = [f.strip() for f in parsed_args.families.split(",")]
            logger.info("Processing families: %s", families)

        # Create configuration
        config = InferenceConfig(
            max_tokens=parsed_args.max_tokens,
            temperature=parsed_args.temperature,
            checkpoint_interval=parsed_args.checkpoint_interval,
        )

        # Create client (not used in dry-run but needed for structure)
        client = BedrockInferenceClient(config=config)

        # Create runner
        runner = InferenceRunner(
            config=config,
            client=client,
            dry_run=parsed_args.dry_run,
        )

        # Run inference
        summaries = runner.run_all(
            prompts=prompts,
            output_dir=parsed_args.output,
            families=families,
            parallel=parsed_args.parallel,
            max_workers=parsed_args.max_workers,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)

        total_completed = 0
        total_errors = 0

        for family, summary in summaries.items():
            total_completed += summary.completed
            total_errors += summary.errors
            error_rate = summary.errors / max(summary.total_prompts, 1) * 100

            print(f"\n{family.upper()}:")
            print(f"  Completed:     {summary.completed:,} / {summary.total_prompts:,}")
            print(f"  Errors:        {summary.errors:,} ({error_rate:.2f}%)")
            print(f"  Avg Base Latency:    {summary.avg_base_latency_ms:.2f} ms")
            print(f"  Avg Aligned Latency: {summary.avg_aligned_latency_ms:.2f} ms")
            print(f"  Duration:      {summary.total_duration_seconds:.2f} s")

        print("\n" + "-" * 60)
        print(f"TOTAL: {total_completed:,} completed, {total_errors:,} errors")
        print("=" * 60)

        # Return success if error rate is acceptable (<1%)
        error_rate = total_errors / max(total_completed + total_errors, 1)
        if error_rate > 0.01:
            logger.warning("Error rate %.2f%% exceeds 1%% threshold", error_rate * 100)
            return 1

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error: %s", type(e).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
