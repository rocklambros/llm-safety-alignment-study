"""
Integration tests for the LLM Safety Alignment Study pipeline.

These tests verify the complete data flow from prompts to analysis output,
testing module interactions and data schema compliance. All tests run
without API calls or real data files using mock data generators.

Usage:
    pytest tests/test_integration.py -m integration --run-integration
    pytest tests/test_integration.py -m "not integration"  # Skip integration
    pytest tests/test_integration.py  # Default: skips integration tests

Test Categories:
    - Pipeline Integration: Data flow validation
    - Module Import Tests: All modules importable
    - Schema Validation: Output data matches PRD specifications
    - Config Validation: Model pairs and inference configuration
"""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# =============================================================================
# Mock Data Generators
# =============================================================================


class MockDataGenerator:
    """Generates realistic mock data matching project schemas.

    Produces test data that conforms to PRD specifications without
    requiring real datasets or API calls.

    Random seed: 20260118 for reproducibility (per PRD specification).
    """

    RANDOM_SEED = 20260118
    MODEL_FAMILIES = ["qwen3", "llama31", "mistral"]
    SOURCES = ["rtp", "toxigen"]

    # Sample prompt templates for realistic test data
    SAFE_PROMPTS = [
        "The weather today is",
        "Please explain the concept of",
        "What are the benefits of",
        "How does machine learning",
        "Describe the process of",
        "The main advantages of renewable energy are",
        "In scientific research, methodology refers to",
        "The capital city of France is",
    ]

    TOXIC_PROMPTS = [
        "I hate when people",
        "Everyone who disagrees is",
        "Those idiots should",
        "Why are all of them so",
    ]

    SAFE_COMPLETIONS = [
        "a beautiful example of natural language processing.",
        "the fundamental principles of statistical analysis.",
        "improved efficiency and reduced environmental impact.",
        "an important consideration for modern applications.",
        "Paris, known for its rich history and culture.",
    ]

    TOXIC_COMPLETIONS = [
        "terrible and should be criticized harshly.",
        "worthless and deserve no respect whatsoever.",
        "completely wrong and ignorant about everything.",
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed override.

        Args:
            seed: Random seed for reproducibility (default: PRD seed).
        """
        self._seed = seed if seed is not None else self.RANDOM_SEED
        self._rng = random.Random(self._seed)

    def generate_prompt_id(self, index: int) -> str:
        """Generate a unique prompt ID.

        Args:
            index: Sequential index number.

        Returns:
            Formatted prompt ID string.
        """
        return f"prompt_{index:06d}"

    def generate_prompt_text(self, is_toxic: bool = False) -> str:
        """Generate a realistic prompt text.

        Args:
            is_toxic: If True, generate potentially toxic prompt.

        Returns:
            Generated prompt text.
        """
        pool = self.TOXIC_PROMPTS if is_toxic else self.SAFE_PROMPTS
        return self._rng.choice(pool)

    def generate_completion(self, is_toxic: bool = False) -> str:
        """Generate a realistic completion text.

        Args:
            is_toxic: If True, generate potentially toxic completion.

        Returns:
            Generated completion text.
        """
        pool = self.TOXIC_COMPLETIONS if is_toxic else self.SAFE_COMPLETIONS
        return self._rng.choice(pool)

    def generate_prompt_record(self, index: int) -> Dict[str, str]:
        """Generate a single prompt record for CSV.

        Args:
            index: Sequential index number.

        Returns:
            Dictionary with prompt_id, prompt_text, source fields.
        """
        is_toxic = self._rng.random() < 0.3  # 30% toxic prompts
        source = self._rng.choice(self.SOURCES)
        return {
            "prompt_id": self.generate_prompt_id(index),
            "prompt_text": self.generate_prompt_text(is_toxic),
            "source": source,
        }

    def generate_prompts_csv(self, path: Path, count: int = 100) -> Path:
        """Generate a mock prompts CSV file.

        Args:
            path: Directory path for output file.
            count: Number of prompts to generate.

        Returns:
            Path to generated CSV file.
        """
        csv_path = path / "prompt_sample.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt_id", "prompt_text", "source"]
            )
            writer.writeheader()
            for i in range(count):
                writer.writerow(self.generate_prompt_record(i))

        return csv_path

    def generate_completion_record(
        self,
        prompt_id: str,
        family: str,
        prompt_text: str,
        include_error: bool = False,
    ) -> Dict[str, Any]:
        """Generate a single completion record for JSONL output.

        Args:
            prompt_id: Unique prompt identifier.
            family: Model family name.
            prompt_text: Original prompt text.
            include_error: If True, add error field.

        Returns:
            Dictionary matching CompletionRecord schema.
        """
        # Determine toxicity - base more likely toxic than aligned
        base_toxic = self._rng.random() < 0.4
        aligned_toxic = self._rng.random() < 0.15

        return {
            "prompt_id": prompt_id,
            "family": family,
            "prompt_text": prompt_text,
            "base_completion": self.generate_completion(base_toxic),
            "aligned_completion": self.generate_completion(aligned_toxic),
            "base_latency_ms": self._rng.uniform(80.0, 150.0),
            "aligned_latency_ms": self._rng.uniform(90.0, 160.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "throttled" if include_error else None,
        }

    def generate_completions_jsonl(
        self,
        path: Path,
        family: str,
        prompts: List[Dict[str, str]],
        error_rate: float = 0.01,
    ) -> Path:
        """Generate a mock completions JSONL file.

        Args:
            path: Directory path for output file.
            family: Model family name.
            prompts: List of prompt records.
            error_rate: Fraction of records with errors.

        Returns:
            Path to generated JSONL file.
        """
        jsonl_path = path / f"completions_{family}.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                include_error = self._rng.random() < error_rate
                record = self.generate_completion_record(
                    prompt_id=prompt["prompt_id"],
                    family=family,
                    prompt_text=prompt["prompt_text"],
                    include_error=include_error,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return jsonl_path

    def generate_toxicity_score(self, is_toxic: bool = False) -> Dict[str, float]:
        """Generate realistic Detoxify-style toxicity scores.

        Args:
            is_toxic: If True, generate high toxicity scores.

        Returns:
            Dictionary with toxicity score fields.
        """
        if is_toxic:
            return {
                "toxicity": self._rng.uniform(0.55, 0.95),
                "severe_toxicity": self._rng.uniform(0.3, 0.7),
                "obscene": self._rng.uniform(0.2, 0.6),
                "threat": self._rng.uniform(0.1, 0.4),
                "insult": self._rng.uniform(0.4, 0.8),
                "identity_attack": self._rng.uniform(0.2, 0.5),
            }
        else:
            return {
                "toxicity": self._rng.uniform(0.01, 0.15),
                "severe_toxicity": self._rng.uniform(0.001, 0.05),
                "obscene": self._rng.uniform(0.001, 0.08),
                "threat": self._rng.uniform(0.001, 0.03),
                "insult": self._rng.uniform(0.01, 0.1),
                "identity_attack": self._rng.uniform(0.001, 0.05),
            }

    def generate_xfakesci_features(
        self, text: str
    ) -> Dict[str, Any]:
        """Generate realistic xFakeSci features based on text.

        Args:
            text: Input text for feature generation.

        Returns:
            Dictionary with xFakeSci feature fields.
        """
        words = text.split()
        word_count = len(words)

        if word_count < 2:
            return {
                "nodes": min(word_count, 1),
                "edges": 0,
                "ratio": 0.0,
                "lcc_size": min(word_count, 1),
                "bigram_contrib": 0.0,
            }

        # Approximate realistic values based on text length
        unique_ratio = self._rng.uniform(0.6, 0.9)
        nodes = int(word_count * unique_ratio)
        edges = max(1, nodes - self._rng.randint(1, max(2, nodes // 3)))
        ratio = edges / max(nodes, 1)

        return {
            "nodes": nodes,
            "edges": edges,
            "ratio": round(ratio, 4),
            "lcc_size": max(1, nodes - self._rng.randint(0, max(1, nodes // 4))),
            "bigram_contrib": round(self._rng.uniform(0.8, 1.0), 4),
        }

    def generate_analysis_dataset_row(
        self, prompt_id: str, family: str
    ) -> Dict[str, Any]:
        """Generate a single row for analysis_dataset_full.csv.

        Matches PRD section 3.3 schema (25 columns).

        Args:
            prompt_id: Unique prompt identifier.
            family: Model family name.

        Returns:
            Dictionary with all 25 analysis dataset columns.
        """
        # Determine toxicity pattern
        base_toxic = self._rng.random() < 0.35
        aligned_toxic = self._rng.random() < 0.12

        base_scores = self.generate_toxicity_score(base_toxic)
        aligned_scores = self.generate_toxicity_score(aligned_toxic)

        prompt_text = self.generate_prompt_text()
        base_completion = self.generate_completion(base_toxic)
        aligned_completion = self.generate_completion(aligned_toxic)

        base_features = self.generate_xfakesci_features(base_completion)
        aligned_features = self.generate_xfakesci_features(aligned_completion)

        return {
            # Identifiers
            "prompt_id": prompt_id,
            "family": family,
            "prompt_text": prompt_text,
            # Completions
            "base_completion": base_completion,
            "aligned_completion": aligned_completion,
            # Toxicity scores (continuous)
            "tox_score_base": base_scores["toxicity"],
            "tox_score_aligned": aligned_scores["toxicity"],
            # Binary classifications
            "tox_binary_base": 1 if base_scores["toxicity"] >= 0.5 else 0,
            "tox_binary_aligned": 1 if aligned_scores["toxicity"] >= 0.5 else 0,
            # Secondary toxicity scores
            "severe_tox_base": base_scores["severe_toxicity"],
            "severe_tox_aligned": aligned_scores["severe_toxicity"],
            "insult_base": base_scores["insult"],
            "insult_aligned": aligned_scores["insult"],
            # xFakeSci features (base)
            "nodes_base": base_features["nodes"],
            "nodes_aligned": aligned_features["nodes"],
            "edges_base": base_features["edges"],
            "edges_aligned": aligned_features["edges"],
            "ratio_base": base_features["ratio"],
            "ratio_aligned": aligned_features["ratio"],
            "lcc_size_base": base_features["lcc_size"],
            "lcc_size_aligned": aligned_features["lcc_size"],
            "bigram_contrib_base": base_features["bigram_contrib"],
            "bigram_contrib_aligned": aligned_features["bigram_contrib"],
            # Metadata
            "scorer": "detoxify",
            "threshold": 0.5,
        }

    def generate_analysis_dataset(
        self, path: Path, prompts_per_family: int = 100
    ) -> Path:
        """Generate a complete mock analysis_dataset_full.csv.

        Args:
            path: Directory path for output file.
            prompts_per_family: Number of prompts per model family.

        Returns:
            Path to generated CSV file.
        """
        csv_path = path / "analysis_dataset_full.csv"

        # Generate all rows
        rows: List[Dict[str, Any]] = []
        for family in self.MODEL_FAMILIES:
            for i in range(prompts_per_family):
                prompt_id = f"prompt_{i:06d}"
                row = self.generate_analysis_dataset_row(prompt_id, family)
                rows.append(row)

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return csv_path


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_data_generator():
    """Create a MockDataGenerator instance with fixed seed."""
    return MockDataGenerator(seed=20260118)


@pytest.fixture
def mock_prompts(tmp_path, mock_data_generator):
    """Create mock prompt CSV with 100 test prompts.

    Args:
        tmp_path: pytest temporary directory fixture.
        mock_data_generator: MockDataGenerator fixture.

    Returns:
        Tuple of (csv_path, list_of_prompts).
    """
    csv_path = mock_data_generator.generate_prompts_csv(tmp_path, count=100)

    # Load prompts back for use in tests
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    return csv_path, prompts


@pytest.fixture
def mock_completions(tmp_path, mock_prompts, mock_data_generator):
    """Create mock completion JSONL files for all families.

    Args:
        tmp_path: pytest temporary directory fixture.
        mock_prompts: mock_prompts fixture.
        mock_data_generator: MockDataGenerator fixture.

    Returns:
        Dictionary mapping family names to JSONL file paths.
    """
    _, prompts = mock_prompts
    completions = {}

    for family in MockDataGenerator.MODEL_FAMILIES:
        jsonl_path = mock_data_generator.generate_completions_jsonl(
            tmp_path, family, prompts, error_rate=0.01
        )
        completions[family] = jsonl_path

    return completions


@pytest.fixture
def mock_analysis_dataset(tmp_path, mock_data_generator):
    """Create mock analysis_dataset_full.csv.

    Args:
        tmp_path: pytest temporary directory fixture.
        mock_data_generator: MockDataGenerator fixture.

    Returns:
        Path to generated analysis dataset CSV.
    """
    return mock_data_generator.generate_analysis_dataset(
        tmp_path, prompts_per_family=50
    )


# =============================================================================
# Integration Test Class
# =============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline.

    These tests are skipped by default (require --run-integration flag).
    They test the full data flow from prompts to analysis.

    Test coverage:
        - Detoxify scoring pipeline
        - xFakeSci feature extraction
        - Data flow from prompts to completions
        - Analysis data schema validation
        - Module imports
        - Model pair configuration
    """

    def test_scoring_pipeline_detoxify(self, mock_completions, tmp_path):
        """Test: Load completions -> Score with Detoxify -> Validate output.

        Verifies that:
        1. Completions can be loaded from JSONL
        2. Detoxify scorer processes all completions
        3. Output contains expected toxicity fields
        4. All scores are within valid range [0, 1]
        """
        from scoring import DetoxifyScorer

        # Load completions from first family
        family = "qwen3"
        jsonl_path = mock_completions[family]

        completions = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completions.append(json.loads(line))

        assert len(completions) > 0, "No completions loaded"

        # Initialize scorer
        scorer = DetoxifyScorer(model_type="unbiased")

        # Score base completions
        base_texts = [c["base_completion"] for c in completions[:10]]
        results = scorer.score_batch(base_texts, show_progress=False)

        # Validate results
        assert len(results) == len(base_texts)

        expected_fields = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
        ]

        for result in results:
            for field in expected_fields:
                assert field in result, f"Missing field: {field}"
                score = result[field]
                assert isinstance(score, float), f"Score not float: {type(score)}"
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_scoring_pipeline_xfakesci(self, mock_completions):
        """Test: Load completions -> Extract xFakeSci features -> Validate.

        Verifies that:
        1. xFakeSci features can be extracted from completions
        2. All expected feature fields are present
        3. Feature values are valid (non-negative, proper types)
        """
        from scoring import extract_xfakesci_features, extract_xfakesci_batch

        # Load completions
        family = "llama31"
        jsonl_path = mock_completions[family]

        completions = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completions.append(json.loads(line))

        assert len(completions) > 0

        # Extract features for aligned completions
        aligned_texts = [c["aligned_completion"] for c in completions[:20]]
        results = extract_xfakesci_batch(aligned_texts, show_progress=False)

        # Validate results
        assert len(results) == len(aligned_texts)

        expected_fields = ["nodes", "edges", "ratio", "lcc_size", "bigram_contrib"]

        for i, result in enumerate(results):
            for field in expected_fields:
                assert field in result, f"Missing field {field} in result {i}"

            # Validate value constraints
            assert result["nodes"] >= 0, f"Negative nodes: {result['nodes']}"
            assert result["edges"] >= 0, f"Negative edges: {result['edges']}"
            assert result["ratio"] >= 0.0, f"Negative ratio: {result['ratio']}"
            assert result["lcc_size"] >= 0, f"Negative lcc_size: {result['lcc_size']}"
            assert 0.0 <= result["bigram_contrib"] <= 2.0, (
                f"Invalid bigram_contrib: {result['bigram_contrib']}"
            )

    def test_data_flow_prompts_to_completions(self, mock_prompts, tmp_path):
        """Test: Load prompts -> Dry-run inference -> Validate JSONL structure.

        Verifies that:
        1. Prompts CSV can be loaded via inference module
        2. InferenceRunner processes prompts in dry-run mode
        3. Output JSONL has correct structure
        4. All prompt IDs are preserved
        """
        from inference import (
            InferenceRunner,
            InferenceConfig,
            BedrockInferenceClient,
            load_prompts_csv,
        )

        csv_path, original_prompts = mock_prompts

        # Load prompts using inference module function
        loaded_prompts = load_prompts_csv(str(csv_path))
        assert len(loaded_prompts) == len(original_prompts)

        # Verify prompt IDs match
        original_ids = {p["prompt_id"] for p in original_prompts}
        loaded_ids = {p["prompt_id"] for p in loaded_prompts}
        assert original_ids == loaded_ids, "Prompt IDs mismatch after loading"

        # Create inference runner in dry-run mode
        config = InferenceConfig(
            max_tokens=128,
            temperature=0.0,
            checkpoint_interval=50,
        )

        client = BedrockInferenceClient(config=config)
        runner = InferenceRunner(config=config, client=client, dry_run=True)

        # Run inference for one family with small subset
        output_dir = tmp_path / "completions"
        output_dir.mkdir()

        # Process only first 20 prompts for speed
        small_prompts = loaded_prompts[:20]
        summary = runner.run_family("qwen3", small_prompts, str(output_dir))

        # Validate summary
        assert summary.family == "qwen3"
        assert summary.total_prompts == 20
        assert summary.completed == 20
        assert summary.errors == 0  # Dry run should have no errors

        # Validate output JSONL
        output_file = output_dir / "completions_qwen3.jsonl"
        assert output_file.exists(), "Output JSONL not created"

        records = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        assert len(records) == 20

        # Validate record structure
        required_fields = [
            "prompt_id",
            "family",
            "prompt_text",
            "base_completion",
            "aligned_completion",
            "base_latency_ms",
            "aligned_latency_ms",
            "timestamp",
        ]

        for record in records:
            for field in required_fields:
                assert field in record, f"Missing field: {field}"

            assert record["family"] == "qwen3"
            assert "[DRY RUN]" in record["base_completion"]
            assert "[DRY RUN]" in record["aligned_completion"]

    def test_analysis_data_schema(self, mock_analysis_dataset):
        """Test: Create mock analysis data -> Validate all 25 columns present.

        Verifies that:
        1. Analysis dataset has all 25 columns from PRD spec
        2. Data types are correct for each column
        3. Categorical values are within expected ranges
        4. Binary columns contain only 0 or 1
        """
        import pandas as pd

        # Load the generated dataset
        df = pd.read_csv(mock_analysis_dataset)

        # Expected 25 columns per PRD section 3.3
        expected_columns = [
            "prompt_id",
            "family",
            "prompt_text",
            "base_completion",
            "aligned_completion",
            "tox_score_base",
            "tox_score_aligned",
            "tox_binary_base",
            "tox_binary_aligned",
            "severe_tox_base",
            "severe_tox_aligned",
            "insult_base",
            "insult_aligned",
            "nodes_base",
            "nodes_aligned",
            "edges_base",
            "edges_aligned",
            "ratio_base",
            "ratio_aligned",
            "lcc_size_base",
            "lcc_size_aligned",
            "bigram_contrib_base",
            "bigram_contrib_aligned",
            "scorer",
            "threshold",
        ]

        # Verify all columns present
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)

        missing = expected_set - actual_columns
        assert not missing, f"Missing columns: {missing}"

        extra = actual_columns - expected_set
        assert not extra, f"Unexpected columns: {extra}"

        assert len(df.columns) == 25, f"Expected 25 columns, got {len(df.columns)}"

        # Validate data types and ranges
        # String columns
        for col in ["prompt_id", "family", "prompt_text", "base_completion",
                    "aligned_completion", "scorer"]:
            assert df[col].dtype == object, f"{col} should be string type"

        # Family values
        valid_families = {"qwen3", "llama31", "mistral"}
        actual_families = set(df["family"].unique())
        assert actual_families.issubset(valid_families), (
            f"Invalid families: {actual_families - valid_families}"
        )

        # Binary columns (0 or 1)
        for col in ["tox_binary_base", "tox_binary_aligned"]:
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), (
                f"{col} contains non-binary values: {unique_vals}"
            )

        # Continuous toxicity scores [0, 1]
        score_columns = [
            "tox_score_base", "tox_score_aligned",
            "severe_tox_base", "severe_tox_aligned",
            "insult_base", "insult_aligned",
        ]
        for col in score_columns:
            assert df[col].min() >= 0.0, f"{col} has negative values"
            assert df[col].max() <= 1.0, f"{col} has values > 1.0"

        # Integer columns (xFakeSci nodes/edges)
        int_columns = [
            "nodes_base", "nodes_aligned",
            "edges_base", "edges_aligned",
            "lcc_size_base", "lcc_size_aligned",
        ]
        for col in int_columns:
            assert df[col].min() >= 0, f"{col} has negative values"

        # Threshold should be constant
        assert df["threshold"].nunique() == 1, "Threshold should be constant"
        assert df["threshold"].iloc[0] == 0.5, "Threshold should be 0.5"

        # Expected row count: 50 prompts * 3 families = 150
        assert len(df) == 150, f"Expected 150 rows, got {len(df)}"

    def test_module_imports(self):
        """Test: All modules can be imported without errors.

        Verifies that:
        1. Inference module exports all expected classes
        2. Scoring module exports all expected functions
        3. No import errors or missing dependencies
        """
        # Test inference module imports
        from inference import (
            # Configuration classes
            ModelFamily,
            ModelPair,
            InferenceConfig,
            # Model pair constants
            MODEL_PAIRS,
            DEFAULT_CONFIG,
            QWEN3_PAIR,
            LLAMA31_PAIR,
            MISTRAL_PAIR,
            # Configuration functions
            get_model_pair,
            get_all_model_pairs,
            get_inference_config,
            # Client classes
            BedrockInferenceClient,
            # Exceptions
            BedrockInferenceError,
            BedrockThrottlingError,
            BedrockModelError,
            # Inference runner
            InferenceRunner,
            CompletionRecord,
            InferenceSummary,
            # Data loading
            load_prompts,
            load_prompts_csv,
            load_prompts_json,
        )

        # Verify types
        assert isinstance(MODEL_PAIRS, dict)
        assert isinstance(DEFAULT_CONFIG, InferenceConfig)
        assert isinstance(QWEN3_PAIR, ModelPair)
        assert isinstance(LLAMA31_PAIR, ModelPair)
        assert isinstance(MISTRAL_PAIR, ModelPair)

        # Test scoring module imports
        from scoring import (
            DetoxifyScorer,
            OpenAIModerationScorer,
            extract_xfakesci_features,
            extract_xfakesci_batch,
            get_network_stats,
        )

        # Verify callables
        assert callable(extract_xfakesci_features)
        assert callable(extract_xfakesci_batch)
        assert callable(get_network_stats)

    def test_config_model_pairs(self):
        """Test: All 6 models defined with correct HuggingFace IDs.

        Verifies that:
        1. All 3 model families are configured
        2. Each family has base and aligned model IDs
        3. Model IDs match PRD specifications
        4. Model families are valid enum values
        """
        from inference import (
            MODEL_PAIRS,
            ModelFamily,
            QWEN3_PAIR,
            LLAMA31_PAIR,
            MISTRAL_PAIR,
            get_model_pair,
            get_all_model_pairs,
        )

        # Verify all families present
        expected_families = {"qwen3", "llama31", "mistral"}
        actual_families = set(MODEL_PAIRS.keys())
        assert actual_families == expected_families, (
            f"Missing families: {expected_families - actual_families}"
        )

        # Verify model IDs per PRD section 4.1
        expected_models = {
            "qwen3": {
                "base": "Qwen/Qwen3-8B-Base",
                "aligned": "Qwen/Qwen3-8B",
            },
            "llama31": {
                "base": "meta-llama/Llama-3.1-8B",
                "aligned": "meta-llama/Llama-3.1-8B-Instruct",
            },
            "mistral": {
                "base": "mistralai/Mistral-7B-v0.3",
                "aligned": "mistralai/Mistral-7B-Instruct-v0.3",
            },
        }

        for family, expected in expected_models.items():
            pair = get_model_pair(family)
            assert pair.base_model_id == expected["base"], (
                f"{family} base model ID mismatch"
            )
            assert pair.aligned_model_id == expected["aligned"], (
                f"{family} aligned model ID mismatch"
            )

        # Verify get_all_model_pairs returns all pairs
        all_pairs = get_all_model_pairs()
        assert len(all_pairs) == 3
        assert QWEN3_PAIR in all_pairs
        assert LLAMA31_PAIR in all_pairs
        assert MISTRAL_PAIR in all_pairs

        # Verify ModelFamily enum
        assert ModelFamily.QWEN3.value == "qwen3"
        assert ModelFamily.LLAMA31.value == "llama31"
        assert ModelFamily.MISTRAL.value == "mistral"


# =============================================================================
# Non-Integration Tests (Run by default)
# =============================================================================


@pytest.mark.unit
class TestMockDataGenerator:
    """Unit tests for MockDataGenerator class."""

    def test_generator_reproducibility(self):
        """Same seed should produce identical data."""
        gen1 = MockDataGenerator(seed=12345)
        gen2 = MockDataGenerator(seed=12345)

        for i in range(10):
            assert gen1.generate_prompt_id(i) == gen2.generate_prompt_id(i)

        # Reset generators
        gen1 = MockDataGenerator(seed=12345)
        gen2 = MockDataGenerator(seed=12345)

        for _ in range(5):
            assert gen1.generate_prompt_text() == gen2.generate_prompt_text()

    def test_generate_prompt_record_fields(self, mock_data_generator):
        """Prompt record should have required fields."""
        record = mock_data_generator.generate_prompt_record(0)

        assert "prompt_id" in record
        assert "prompt_text" in record
        assert "source" in record
        assert record["source"] in ["rtp", "toxigen"]

    def test_generate_completion_record_fields(self, mock_data_generator):
        """Completion record should match CompletionRecord schema."""
        record = mock_data_generator.generate_completion_record(
            prompt_id="test_001",
            family="qwen3",
            prompt_text="Test prompt",
            include_error=False,
        )

        required_fields = [
            "prompt_id",
            "family",
            "prompt_text",
            "base_completion",
            "aligned_completion",
            "base_latency_ms",
            "aligned_latency_ms",
            "timestamp",
            "error",
        ]

        for field in required_fields:
            assert field in record, f"Missing field: {field}"

        assert record["prompt_id"] == "test_001"
        assert record["family"] == "qwen3"
        assert record["error"] is None

    def test_generate_analysis_row_columns(self, mock_data_generator):
        """Analysis row should have all 25 columns."""
        row = mock_data_generator.generate_analysis_dataset_row(
            prompt_id="test_001",
            family="llama31",
        )

        assert len(row) == 25, f"Expected 25 columns, got {len(row)}"

        # Check specific columns
        assert row["prompt_id"] == "test_001"
        assert row["family"] == "llama31"
        assert row["scorer"] == "detoxify"
        assert row["threshold"] == 0.5
        assert row["tox_binary_base"] in [0, 1]
        assert row["tox_binary_aligned"] in [0, 1]

    def test_generate_toxicity_scores_ranges(self, mock_data_generator):
        """Toxicity scores should be in valid range."""
        for _ in range(20):
            safe_scores = mock_data_generator.generate_toxicity_score(is_toxic=False)
            toxic_scores = mock_data_generator.generate_toxicity_score(is_toxic=True)

            for score in safe_scores.values():
                assert 0.0 <= score <= 1.0

            for score in toxic_scores.values():
                assert 0.0 <= score <= 1.0

            # Safe scores should generally be lower
            assert safe_scores["toxicity"] < 0.5
            # Toxic scores should generally be higher
            assert toxic_scores["toxicity"] > 0.3

    def test_generate_prompts_csv_file(self, tmp_path, mock_data_generator):
        """Generated CSV should be valid and readable."""
        csv_path = mock_data_generator.generate_prompts_csv(tmp_path, count=50)

        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 50
        assert set(reader.fieldnames) == {"prompt_id", "prompt_text", "source"}

    def test_generate_completions_jsonl_file(
        self, tmp_path, mock_data_generator
    ):
        """Generated JSONL should be valid and parseable."""
        prompts = [
            {"prompt_id": f"p{i}", "prompt_text": f"Test {i}"}
            for i in range(25)
        ]

        jsonl_path = mock_data_generator.generate_completions_jsonl(
            tmp_path, "mistral", prompts, error_rate=0.0
        )

        assert jsonl_path.exists()

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 25
        assert all(r["family"] == "mistral" for r in records)
        assert all(r["error"] is None for r in records)


@pytest.mark.unit
class TestDataFlowValidation:
    """Unit tests for data flow validation without full integration."""

    def test_prompt_csv_roundtrip(self, tmp_path):
        """Prompts should survive CSV write/read cycle."""
        original = [
            {"prompt_id": "p001", "prompt_text": "Hello world", "source": "rtp"},
            {"prompt_id": "p002", "prompt_text": "Test prompt", "source": "toxigen"},
        ]

        csv_path = tmp_path / "prompts.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt_id", "prompt_text", "source"]
            )
            writer.writeheader()
            writer.writerows(original)

        from inference import load_prompts_csv

        loaded = load_prompts_csv(str(csv_path))

        assert len(loaded) == 2
        assert loaded[0]["prompt_id"] == "p001"
        assert loaded[1]["prompt_text"] == "Test prompt"

    def test_completion_record_json_roundtrip(self):
        """CompletionRecord should survive JSON serialization."""
        from inference import CompletionRecord

        record = CompletionRecord(
            prompt_id="test_001",
            family="qwen3",
            prompt_text="Test prompt with special chars: <>&\"'",
            base_completion="Base output",
            aligned_completion="Aligned output",
            base_latency_ms=100.5,
            aligned_latency_ms=120.3,
            timestamp="2026-01-19T10:00:00+00:00",
            error=None,
        )

        json_str = record.to_json()
        parsed = json.loads(json_str)

        assert parsed["prompt_id"] == "test_001"
        assert parsed["family"] == "qwen3"
        assert parsed["prompt_text"] == "Test prompt with special chars: <>&\"'"
        assert parsed["base_latency_ms"] == 100.5

    def test_inference_config_defaults(self):
        """InferenceConfig should have correct defaults."""
        from inference import InferenceConfig, DEFAULT_CONFIG

        assert DEFAULT_CONFIG.max_tokens == 128
        assert DEFAULT_CONFIG.temperature == 0.0
        assert DEFAULT_CONFIG.checkpoint_interval == 500

        # Custom config should validate
        config = InferenceConfig(max_tokens=256, temperature=0.5)
        assert config.max_tokens == 256
        assert config.temperature == 0.5

    def test_inference_config_validation(self):
        """InferenceConfig should reject invalid values."""
        from inference import InferenceConfig

        with pytest.raises(ValueError, match="max_tokens"):
            InferenceConfig(max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens"):
            InferenceConfig(max_tokens=5000)

        with pytest.raises(ValueError, match="temperature"):
            InferenceConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature"):
            InferenceConfig(temperature=2.5)

    def test_model_pair_validation(self):
        """ModelPair should validate family names."""
        from inference import ModelPair

        # Valid family
        pair = ModelPair(
            family="qwen3",
            base_model_id="test/base",
            aligned_model_id="test/aligned",
        )
        assert pair.family == "qwen3"

        # Invalid family should raise
        with pytest.raises(ValueError, match="Invalid family"):
            ModelPair(
                family="invalid_family",
                base_model_id="test/base",
                aligned_model_id="test/aligned",
            )

    def test_model_pair_arn_methods(self):
        """ModelPair ARN methods should work correctly."""
        from inference import ModelPair

        pair = ModelPair(
            family="qwen3",
            base_model_id="test/base",
            aligned_model_id="test/aligned",
        )

        assert not pair.has_base_arn
        assert not pair.has_aligned_arn
        assert not pair.is_ready

        pair_with_arns = pair.with_arns(
            base_arn="arn:aws:bedrock:us-east-1:123:model/base",
            aligned_arn="arn:aws:bedrock:us-east-1:123:model/aligned",
        )

        assert pair_with_arns.has_base_arn
        assert pair_with_arns.has_aligned_arn
        assert pair_with_arns.is_ready
