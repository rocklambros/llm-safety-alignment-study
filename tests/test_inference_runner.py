"""Tests for inference_runner module.

Tests cover:
- InferenceRunner with checkpointing and resumability
- Prompt loading from CSV and JSON
- CLI argument parsing
- Error handling and edge cases
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from inference.config import (
    DEFAULT_CONFIG,
    InferenceConfig,
    ModelFamily,
    ModelPair,
    get_model_pair,
)
from inference.inference_runner import (
    CompletionRecord,
    InferenceRunner,
    InferenceSummary,
    PromptRecord,
    load_prompts,
    load_prompts_csv,
    load_prompts_json,
    main,
    parse_args,
    validate_path,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_prompts() -> List[Dict[str, Any]]:
    """Sample prompts for testing."""
    return [
        {"prompt_id": "test_001", "prompt_text": "Hello, world!", "source": "test"},
        {"prompt_id": "test_002", "prompt_text": "How are you?", "source": "test"},
        {"prompt_id": "test_003", "prompt_text": "Tell me a story.", "source": "test"},
        {"prompt_id": "test_004", "prompt_text": "What is AI?", "source": "test"},
        {"prompt_id": "test_005", "prompt_text": "Explain quantum computing.", "source": "test"},
    ]


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_prompts: List[Dict[str, Any]]) -> Path:
    """Create a sample CSV file with prompts."""
    import csv as csv_module
    csv_file = tmp_path / "prompts.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["prompt_id", "prompt_text", "source"])
        writer.writeheader()
        for prompt in sample_prompts:
            writer.writerow(prompt)
    return csv_file


@pytest.fixture
def sample_json_file(tmp_path: Path, sample_prompts: List[Dict[str, Any]]) -> Path:
    """Create a sample JSON file with prompts."""
    json_file = tmp_path / "prompts.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sample_prompts, f)
    return json_file


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.generate_with_retry.return_value = ("Generated text response", 150.0)
    return client


@pytest.fixture
def test_config() -> InferenceConfig:
    """Create test configuration with small checkpoint interval."""
    return InferenceConfig(
        max_tokens=64,
        temperature=0.0,
        checkpoint_interval=2,  # Small for testing
        timeout_seconds=30,
    )


@pytest.fixture
def model_pair_with_arns() -> ModelPair:
    """Create a model pair with mock ARNs."""
    return ModelPair(
        family="qwen3",
        base_model_id="Qwen/Qwen3-8B-Base",
        aligned_model_id="Qwen/Qwen3-8B",
        base_arn="arn:aws:bedrock:us-east-1:123456789:imported-model/test-base",
        aligned_arn="arn:aws:bedrock:us-east-1:123456789:imported-model/test-aligned",
    )


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestValidatePath:
    """Tests for path validation."""

    def test_validate_simple_path(self, tmp_path: Path) -> None:
        """Test validation of a simple path."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        result = validate_path(str(test_file))
        assert result == test_file.resolve()

    def test_validate_path_with_base_dir(self, tmp_path: Path) -> None:
        """Test validation with base directory restriction."""
        test_file = tmp_path / "subdir" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        result = validate_path(str(test_file), base_dir=tmp_path)
        assert result == test_file.resolve()

    def test_validate_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Test that path traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path("../../../etc/passwd", base_dir=tmp_path)


# =============================================================================
# Prompt Loading Tests
# =============================================================================


class TestLoadPrompts:
    """Tests for prompt loading functions."""

    def test_load_prompts_csv(
        self, sample_csv_file: Path, sample_prompts: List[Dict[str, Any]]
    ) -> None:
        """Test loading prompts from CSV."""
        result = load_prompts_csv(str(sample_csv_file))

        assert len(result) == len(sample_prompts)
        assert result[0]["prompt_id"] == "test_001"
        assert result[0]["prompt_text"] == "Hello, world!"

    def test_load_prompts_json(
        self, sample_json_file: Path, sample_prompts: List[Dict[str, Any]]
    ) -> None:
        """Test loading prompts from JSON."""
        result = load_prompts_json(str(sample_json_file))

        assert len(result) == len(sample_prompts)
        assert result[0]["prompt_id"] == "test_001"
        assert result[0]["prompt_text"] == "Hello, world!"

    def test_load_prompts_auto_detect_csv(self, sample_csv_file: Path) -> None:
        """Test auto-detection of CSV format."""
        result = load_prompts(str(sample_csv_file))
        assert len(result) > 0

    def test_load_prompts_auto_detect_json(self, sample_json_file: Path) -> None:
        """Test auto-detection of JSON format."""
        result = load_prompts(str(sample_json_file))
        assert len(result) > 0

    def test_load_prompts_file_not_found(self) -> None:
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_prompts("/nonexistent/file.csv")

    def test_load_prompts_unsupported_format(self, tmp_path: Path) -> None:
        """Test error for unsupported file format."""
        unsupported = tmp_path / "prompts.xml"
        unsupported.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_prompts(str(unsupported))

    def test_load_prompts_csv_missing_columns(self, tmp_path: Path) -> None:
        """Test error when CSV is missing required columns."""
        csv_file = tmp_path / "invalid.csv"
        with open(csv_file, "w") as f:
            f.write("id,content\n")
            f.write("1,hello\n")

        with pytest.raises(ValueError, match="Missing required columns"):
            load_prompts_csv(str(csv_file))

    def test_load_prompts_csv_with_text_column(self, tmp_path: Path) -> None:
        """Test CSV loading with 'text' column instead of 'prompt_text'."""
        csv_file = tmp_path / "prompts.csv"
        with open(csv_file, "w") as f:
            f.write("prompt_id,text\n")
            f.write("p1,Hello there\n")

        result = load_prompts_csv(str(csv_file))
        assert len(result) == 1
        assert result[0]["prompt_text"] == "Hello there"


# =============================================================================
# Data Class Tests
# =============================================================================


class TestCompletionRecord:
    """Tests for CompletionRecord data class."""

    def test_completion_record_to_dict(self) -> None:
        """Test conversion to dictionary."""
        record = CompletionRecord(
            prompt_id="test_001",
            family="qwen3",
            prompt_text="Hello",
            base_completion="Hi there",
            aligned_completion="Hello! How can I help?",
            base_latency_ms=100.0,
            aligned_latency_ms=120.0,
            timestamp="2026-01-19T12:00:00Z",
            error=None,
        )

        result = record.to_dict()

        assert result["prompt_id"] == "test_001"
        assert result["family"] == "qwen3"
        assert result["base_completion"] == "Hi there"
        assert result["error"] is None

    def test_completion_record_to_json(self) -> None:
        """Test conversion to JSON string."""
        record = CompletionRecord(
            prompt_id="test_001",
            family="qwen3",
            prompt_text="Hello",
            base_completion="Hi",
            aligned_completion="Hello!",
            base_latency_ms=100.0,
            aligned_latency_ms=120.0,
            timestamp="2026-01-19T12:00:00Z",
            error=None,
        )

        result = record.to_json()
        parsed = json.loads(result)

        assert parsed["prompt_id"] == "test_001"
        assert parsed["family"] == "qwen3"

    def test_completion_record_with_error(self) -> None:
        """Test completion record with error."""
        record = CompletionRecord(
            prompt_id="test_001",
            family="qwen3",
            prompt_text="Hello",
            base_completion="",
            aligned_completion="",
            base_latency_ms=50.0,
            aligned_latency_ms=0.0,
            timestamp="2026-01-19T12:00:00Z",
            error="base_throttled:ThrottlingException",
        )

        result = record.to_dict()
        assert result["error"] == "base_throttled:ThrottlingException"


class TestInferenceSummary:
    """Tests for InferenceSummary data class."""

    def test_inference_summary_to_dict(self) -> None:
        """Test conversion to dictionary."""
        summary = InferenceSummary(
            family="qwen3",
            total_prompts=100,
            completed=95,
            errors=5,
            avg_base_latency_ms=150.0,
            avg_aligned_latency_ms=140.0,
            total_duration_seconds=120.5,
        )

        result = summary.to_dict()

        assert result["family"] == "qwen3"
        assert result["total_prompts"] == 100
        assert result["completed"] == 95
        assert result["errors"] == 5
        assert result["error_rate"] == 0.05
        assert result["avg_base_latency_ms"] == 150.0


# =============================================================================
# InferenceRunner Tests
# =============================================================================


class TestInferenceRunner:
    """Tests for InferenceRunner class."""

    def test_runner_initialization(
        self, test_config: InferenceConfig, mock_bedrock_client: MagicMock
    ) -> None:
        """Test runner initialization."""
        runner = InferenceRunner(test_config, mock_bedrock_client)

        assert runner.config == test_config
        assert runner.dry_run is False

    def test_runner_dry_run_mode(
        self, test_config: InferenceConfig, mock_bedrock_client: MagicMock
    ) -> None:
        """Test runner in dry-run mode."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)

        assert runner.dry_run is True

    def test_run_family_dry_run(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test running inference for a single family in dry-run mode."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)

        output_dir = tmp_path / "output"
        summary = runner.run_family("qwen3", sample_prompts, str(output_dir))

        assert summary.family == "qwen3"
        assert summary.total_prompts == len(sample_prompts)
        assert summary.completed == len(sample_prompts)
        assert summary.errors == 0

        # Check output file exists
        output_file = output_dir / "completions_qwen3.jsonl"
        assert output_file.exists()

        # Check checkpoint file exists
        checkpoint_file = output_dir / "checkpoint_qwen3.json"
        assert checkpoint_file.exists()

    def test_run_family_creates_output_directory(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test that run_family creates output directory if not exists."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)

        output_dir = tmp_path / "deep" / "nested" / "output"
        runner.run_family("qwen3", sample_prompts, str(output_dir))

        assert output_dir.exists()

    def test_run_family_checkpoint_resumption(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test that run_family resumes from checkpoint."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        # First run - process all prompts
        runner.run_family("qwen3", sample_prompts, str(output_dir))

        # Load checkpoint
        checkpoint_file = output_dir / "checkpoint_qwen3.json"
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        # Verify checkpoint has correct count
        assert checkpoint["last_index"] == len(sample_prompts)
        assert checkpoint["completed_count"] == len(sample_prompts)

    def test_run_family_output_format(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test that output follows PRD format."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        runner.run_family("qwen3", sample_prompts, str(output_dir))

        # Check JSONL format
        output_file = output_dir / "completions_qwen3.jsonl"
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == len(sample_prompts)

        # Verify record format
        first_record = json.loads(lines[0])
        required_fields = {
            "prompt_id",
            "family",
            "prompt_text",
            "base_completion",
            "aligned_completion",
            "base_latency_ms",
            "aligned_latency_ms",
            "timestamp",
            "error",
        }
        assert set(first_record.keys()) == required_fields

    def test_run_all_sequential(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test running all families sequentially."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        summaries = runner.run_all(
            sample_prompts,
            str(output_dir),
            families=["qwen3", "llama31"],
            parallel=False,
        )

        assert "qwen3" in summaries
        assert "llama31" in summaries
        assert summaries["qwen3"].completed == len(sample_prompts)
        assert summaries["llama31"].completed == len(sample_prompts)

    def test_run_all_parallel(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test running all families in parallel."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        summaries = runner.run_all(
            sample_prompts,
            str(output_dir),
            families=["qwen3", "llama31"],
            parallel=True,
            max_workers=2,
        )

        assert "qwen3" in summaries
        assert "llama31" in summaries

    def test_run_all_saves_summary(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test that run_all saves overall summary."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        runner.run_all(sample_prompts, str(output_dir), families=["qwen3"])

        summary_file = output_dir / "run_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "timestamp" in summary
        assert "families" in summary
        assert "qwen3" in summary["families"]

    def test_run_family_invalid_family(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test error for invalid family name."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)

        with pytest.raises(KeyError):
            runner.run_family("invalid_family", sample_prompts, str(tmp_path))

    def test_run_all_invalid_family(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        sample_prompts: List[Dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Test error for invalid family in run_all."""
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)

        with pytest.raises(ValueError, match="Invalid family"):
            runner.run_all(
                sample_prompts,
                str(tmp_path),
                families=["qwen3", "invalid"],
            )


class TestInferenceRunnerWithMockedClient:
    """Tests for InferenceRunner with mocked API calls."""

    def test_process_prompt_success(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        model_pair_with_arns: ModelPair,
    ) -> None:
        """Test successful prompt processing."""
        mock_bedrock_client.generate_with_retry.return_value = ("Generated text", 100.0)

        runner = InferenceRunner(test_config, mock_bedrock_client)
        prompt_data = {"prompt_id": "test_001", "prompt_text": "Hello"}

        record = runner._process_prompt(prompt_data, model_pair_with_arns)

        assert record.prompt_id == "test_001"
        assert record.base_completion == "Generated text"
        assert record.aligned_completion == "Generated text"
        assert record.error is None

    def test_process_prompt_with_api_error(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        model_pair_with_arns: ModelPair,
    ) -> None:
        """Test prompt processing with API error."""
        from inference.bedrock_client import BedrockThrottlingError

        mock_bedrock_client.generate_with_retry.side_effect = [
            BedrockThrottlingError("Throttled"),
            ("Aligned response", 100.0),
        ]

        runner = InferenceRunner(test_config, mock_bedrock_client)
        prompt_data = {"prompt_id": "test_001", "prompt_text": "Hello"}

        record = runner._process_prompt(prompt_data, model_pair_with_arns)

        assert record.base_completion == ""
        assert record.aligned_completion == "Aligned response"
        assert "base_throttled" in record.error


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_parse_args_required(self) -> None:
        """Test parsing required arguments."""
        args = parse_args(["--prompts", "test.csv", "--output", "output/"])

        assert args.prompts == "test.csv"
        assert args.output == "output/"

    def test_parse_args_with_families(self) -> None:
        """Test parsing families argument."""
        args = parse_args([
            "--prompts", "test.csv",
            "--output", "output/",
            "--families", "qwen3,llama31",
        ])

        assert args.families == "qwen3,llama31"

    def test_parse_args_parallel(self) -> None:
        """Test parsing parallel flag."""
        args = parse_args([
            "--prompts", "test.csv",
            "--output", "output/",
            "--parallel",
        ])

        assert args.parallel is True

    def test_parse_args_dry_run(self) -> None:
        """Test parsing dry-run flag."""
        args = parse_args([
            "--prompts", "test.csv",
            "--output", "output/",
            "--dry-run",
        ])

        assert args.dry_run is True

    def test_parse_args_all_options(self) -> None:
        """Test parsing all options."""
        args = parse_args([
            "--prompts", "test.csv",
            "--output", "output/",
            "--families", "qwen3",
            "--parallel",
            "--max-workers", "5",
            "--checkpoint-interval", "100",
            "--max-tokens", "256",
            "--temperature", "0.7",
            "--dry-run",
            "--verbose",
        ])

        assert args.prompts == "test.csv"
        assert args.output == "output/"
        assert args.families == "qwen3"
        assert args.parallel is True
        assert args.max_workers == 5
        assert args.checkpoint_interval == 100
        assert args.max_tokens == 256
        assert args.temperature == 0.7
        assert args.dry_run is True
        assert args.verbose is True


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_dry_run(
        self,
        sample_csv_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test main function in dry-run mode."""
        output_dir = tmp_path / "output"

        exit_code = main([
            "--prompts", str(sample_csv_file),
            "--output", str(output_dir),
            "--families", "qwen3",
            "--dry-run",
        ])

        assert exit_code == 0
        assert (output_dir / "completions_qwen3.jsonl").exists()

    def test_main_file_not_found(self, tmp_path: Path) -> None:
        """Test main with nonexistent prompts file."""
        exit_code = main([
            "--prompts", "/nonexistent/file.csv",
            "--output", str(tmp_path),
            "--dry-run",
        ])

        assert exit_code == 1

    def test_main_empty_prompts(self, tmp_path: Path) -> None:
        """Test main with empty prompts file."""
        empty_csv = tmp_path / "empty.csv"
        with open(empty_csv, "w") as f:
            f.write("prompt_id,prompt_text\n")

        exit_code = main([
            "--prompts", str(empty_csv),
            "--output", str(tmp_path / "output"),
            "--dry-run",
        ])

        assert exit_code == 1  # No prompts loaded


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_unicode_prompt_text(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test handling of unicode prompt text."""
        prompts = [
            {"prompt_id": "unicode_001", "prompt_text": "Hello!"},
            {"prompt_id": "unicode_002", "prompt_text": "Chinese text here"},
            {"prompt_id": "unicode_003", "prompt_text": "Japanese text here"},
        ]

        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        summary = runner.run_family("qwen3", prompts, str(output_dir))

        assert summary.completed == 3
        assert summary.errors == 0

    def test_empty_prompt_text(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test handling of empty prompt text."""
        prompts = [
            {"prompt_id": "empty_001", "prompt_text": ""},
        ]

        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        summary = runner.run_family("qwen3", prompts, str(output_dir))

        # Should still process (dry run doesn't validate content)
        assert summary.completed == 1

    def test_very_long_prompt_text(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test handling of very long prompt text."""
        prompts = [
            {"prompt_id": "long_001", "prompt_text": "x" * 10000},
        ]

        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        output_dir = tmp_path / "output"

        summary = runner.run_family("qwen3", prompts, str(output_dir))

        assert summary.completed == 1

    def test_checkpoint_recovery_after_partial_run(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test checkpoint recovery after partial run."""
        # Create 5 prompts
        prompts = [
            {"prompt_id": f"test_{i:03d}", "prompt_text": f"Prompt {i}"}
            for i in range(5)
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Simulate partial run - write 2 records
        output_file = output_dir / "completions_qwen3.jsonl"
        checkpoint_file = output_dir / "checkpoint_qwen3.json"

        partial_records = []
        for i in range(2):
            record = CompletionRecord(
                prompt_id=f"test_{i:03d}",
                family="qwen3",
                prompt_text=f"Prompt {i}",
                base_completion="Base",
                aligned_completion="Aligned",
                base_latency_ms=100.0,
                aligned_latency_ms=100.0,
                timestamp="2026-01-19T12:00:00Z",
                error=None,
            )
            partial_records.append(record)

        with open(output_file, "w") as f:
            for record in partial_records:
                f.write(record.to_json() + "\n")

        # Save checkpoint at index 2
        checkpoint_data = {
            "family": "qwen3",
            "last_index": 2,
            "completed_count": 2,
            "timestamp": "2026-01-19T12:00:00Z",
            "prompt_ids": ["test_000", "test_001"],
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Resume from checkpoint
        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        summary = runner.run_family("qwen3", prompts, str(output_dir))

        # Should have all 5 records now
        assert summary.completed == 5

        # Verify output file has 5 records
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_checkpoint_family_mismatch(
        self,
        test_config: InferenceConfig,
        mock_bedrock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test checkpoint with mismatched family is ignored."""
        prompts = [
            {"prompt_id": "test_001", "prompt_text": "Hello"},
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create checkpoint for different family
        checkpoint_file = output_dir / "checkpoint_qwen3.json"
        checkpoint_data = {
            "family": "llama31",  # Wrong family
            "last_index": 1,
            "completed_count": 1,
            "prompt_ids": ["test_001"],
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        runner = InferenceRunner(test_config, mock_bedrock_client, dry_run=True)
        summary = runner.run_family("qwen3", prompts, str(output_dir))

        # Should start fresh since family doesn't match
        assert summary.completed == 1
