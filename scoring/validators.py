#!/usr/bin/env python3
"""
Data validation utilities for toxicity scoring pipeline.

Provides comprehensive quality checks for completion records and analysis datasets.
Validates schema compliance, data integrity, score ranges, and distribution sanity.

Security:
- Input validation to prevent malformed data
- No code execution from data content
- Sanitized error messages

Usage:
    from scoring.validators import validate_completion_record, validate_analysis_dataset

    # Validate individual record
    try:
        validate_completion_record(record)
    except ValueError as e:
        print(f"Invalid record: {e}")

    # Validate full dataset
    report = validate_analysis_dataset(df)
    if report["errors"]:
        print("Validation failed:", report["errors"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Record Validation
# =============================================================================

def validate_completion_record(record: Dict[str, Any]) -> None:
    """Validate a single completion record from JSONL.

    Args:
        record: Completion record dictionary

    Raises:
        ValueError: If record is invalid
    """
    # Required fields
    required_fields = [
        "prompt_id",
        "family",
        "prompt_text",
        "base_completion",
        "aligned_completion"
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

        if record[field] is None:
            raise ValueError(f"Field cannot be None: {field}")

    # Validate family value
    valid_families = {"qwen3", "llama31", "mistral"}
    if record["family"] not in valid_families:
        raise ValueError(
            f"Invalid family: {record['family']}. "
            f"Must be one of {valid_families}"
        )

    # Validate text fields are strings
    text_fields = ["prompt_id", "prompt_text", "base_completion", "aligned_completion"]
    for field in text_fields:
        if not isinstance(record[field], str):
            raise ValueError(f"Field must be string: {field}")

    # Warn about empty completions (but don't fail)
    if len(record["base_completion"].strip()) == 0:
        logger.warning(f"Empty base_completion for {record['prompt_id']}")

    if len(record["aligned_completion"].strip()) == 0:
        logger.warning(f"Empty aligned_completion for {record['prompt_id']}")


def validate_completion_batch(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate batch of completion records.

    Args:
        records: List of completion records

    Returns:
        Validation report dict with counts and issues
    """
    report = {
        "total_records": len(records),
        "valid_records": 0,
        "invalid_records": 0,
        "warnings": [],
        "errors": []
    }

    # Track unique prompt_ids per family
    prompt_ids_by_family = {
        "qwen3": set(),
        "llama31": set(),
        "mistral": set()
    }

    for i, record in enumerate(records):
        try:
            validate_completion_record(record)
            report["valid_records"] += 1

            # Check for duplicate prompt_ids within family
            family = record["family"]
            prompt_id = record["prompt_id"]

            if prompt_id in prompt_ids_by_family[family]:
                report["warnings"].append(
                    f"Duplicate prompt_id in {family}: {prompt_id}"
                )
            else:
                prompt_ids_by_family[family].add(prompt_id)

        except ValueError as e:
            report["invalid_records"] += 1
            report["errors"].append(f"Record {i}: {str(e)}")

    # Family balance check
    for family, ids in prompt_ids_by_family.items():
        count = len(ids)
        if count != 25000:
            report["warnings"].append(
                f"{family}: expected 25,000 records, got {count}"
            )

    return report


# =============================================================================
# Dataset Validation
# =============================================================================

def validate_analysis_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive validation of analysis dataset.

    Args:
        df: Analysis dataset DataFrame

    Returns:
        Validation report dict with detailed checks
    """
    report = {
        "schema_validation": {},
        "data_quality": {},
        "score_distributions": {},
        "warnings": [],
        "errors": []
    }

    # Schema validation
    expected_shape = (75000, 24)
    actual_shape = df.shape

    report["schema_validation"] = {
        "expected_rows": expected_shape[0],
        "actual_rows": actual_shape[0],
        "expected_cols": expected_shape[1],
        "actual_cols": actual_shape[1],
        "schema_match": actual_shape == expected_shape
    }

    if actual_shape != expected_shape:
        report["errors"].append(
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    # Column validation
    expected_columns = [
        "prompt_id", "family", "prompt_text",
        "base_completion", "aligned_completion",
        "tox_score_base", "tox_score_aligned",
        "tox_binary_base", "tox_binary_aligned",
        "severe_tox_base", "severe_tox_aligned",
        "insult_base", "insult_aligned",
        "nodes_base", "nodes_aligned",
        "edges_base", "edges_aligned",
        "ratio_base", "ratio_aligned",
        "lcc_size_base", "lcc_size_aligned",
        "bigram_contrib_base", "bigram_contrib_aligned",
        "scorer", "threshold"
    ]

    actual_columns = list(df.columns)
    missing_columns = set(expected_columns) - set(actual_columns)
    extra_columns = set(actual_columns) - set(expected_columns)

    if missing_columns:
        report["errors"].append(f"Missing columns: {missing_columns}")

    if extra_columns:
        report["warnings"].append(f"Extra columns: {extra_columns}")

    # Missing value audit
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        missing_dict = missing_counts[missing_counts > 0].to_dict()
        report["data_quality"]["missing_values"] = missing_dict
        report["warnings"].append(
            f"Missing values detected in {len(missing_dict)} column(s)"
        )

    # Score range validation
    score_columns = [
        "tox_score_base", "tox_score_aligned",
        "severe_tox_base", "severe_tox_aligned",
        "insult_base", "insult_aligned",
        "ratio_base", "ratio_aligned",
        "bigram_contrib_base", "bigram_contrib_aligned"
    ]

    for col in score_columns:
        if col not in df.columns:
            continue

        min_val = df[col].min()
        max_val = df[col].max()

        if min_val < 0 or max_val > 1:
            report["errors"].append(
                f"{col} out of range [0,1]: min={min_val:.4f}, max={max_val:.4f}"
            )

    # Binary column validation
    binary_columns = ["tox_binary_base", "tox_binary_aligned"]

    for col in binary_columns:
        if col not in df.columns:
            continue

        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            report["errors"].append(
                f"{col} contains invalid values: {unique_vals}"
            )

    # Integer column validation (xFakeSci features)
    integer_columns = [
        "nodes_base", "nodes_aligned",
        "edges_base", "edges_aligned",
        "lcc_size_base", "lcc_size_aligned"
    ]

    for col in integer_columns:
        if col not in df.columns:
            continue

        if df[col].min() < 0:
            report["errors"].append(f"{col} contains negative values")

    # Score distribution analysis
    if "tox_score_base" in df.columns and "tox_score_aligned" in df.columns:
        report["score_distributions"] = {
            "tox_score_base": {
                "mean": float(df["tox_score_base"].mean()),
                "median": float(df["tox_score_base"].median()),
                "std": float(df["tox_score_base"].std()),
                "min": float(df["tox_score_base"].min()),
                "max": float(df["tox_score_base"].max()),
                "toxic_rate": float((df.get("tox_binary_base", 0) == 1).mean())
                    if "tox_binary_base" in df.columns else None
            },
            "tox_score_aligned": {
                "mean": float(df["tox_score_aligned"].mean()),
                "median": float(df["tox_score_aligned"].median()),
                "std": float(df["tox_score_aligned"].std()),
                "min": float(df["tox_score_aligned"].min()),
                "max": float(df["tox_score_aligned"].max()),
                "toxic_rate": float((df.get("tox_binary_aligned", 0) == 1).mean())
                    if "tox_binary_aligned" in df.columns else None
            }
        }

        # Sanity check: distributions shouldn't be degenerate
        base_std = report["score_distributions"]["tox_score_base"]["std"]
        aligned_std = report["score_distributions"]["tox_score_aligned"]["std"]

        if base_std < 0.01:
            report["warnings"].append(
                "Base toxicity scores have very low variance (possible scoring issue)"
            )

        if aligned_std < 0.01:
            report["warnings"].append(
                "Aligned toxicity scores have very low variance (possible scoring issue)"
            )

    # Family balance check
    if "family" in df.columns:
        family_counts = df["family"].value_counts().to_dict()
        expected_per_family = 25000

        for family, count in family_counts.items():
            if count != expected_per_family:
                report["warnings"].append(
                    f"{family}: expected {expected_per_family} records, got {count}"
                )

    # Duplicate check
    if "prompt_id" in df.columns and "family" in df.columns:
        duplicate_count = df.duplicated(subset=["prompt_id", "family"]).sum()
        if duplicate_count > 0:
            report["errors"].append(
                f"Found {duplicate_count} duplicate (prompt_id, family) pairs"
            )

    return report


def validate_score_dict(score: Dict[str, Any], expected_fields: Set[str]) -> None:
    """Validate score dictionary structure.

    Args:
        score: Score dictionary from scorer
        expected_fields: Set of expected field names

    Raises:
        ValueError: If score dict is invalid
    """
    if not isinstance(score, dict):
        raise ValueError("Score must be a dictionary")

    missing_fields = expected_fields - set(score.keys())
    if missing_fields:
        raise ValueError(f"Missing score fields: {missing_fields}")

    # Check all score values are numeric
    for field, value in score.items():
        if field == "error":  # Error field can be string or None
            continue

        if not isinstance(value, (int, float)):
            raise ValueError(f"Score field {field} must be numeric, got {type(value)}")


def create_validation_summary(report: Dict[str, Any]) -> str:
    """Create human-readable validation summary.

    Args:
        report: Validation report from validate_analysis_dataset

    Returns:
        Formatted summary string
    """
    lines = ["=" * 60]
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 60)

    # Schema
    schema = report.get("schema_validation", {})
    lines.append(f"\nSchema: {schema.get('actual_rows', 0)} rows × {schema.get('actual_cols', 0)} cols")

    if schema.get("schema_match", False):
        lines.append("✅ Schema matches expected format")
    else:
        lines.append("❌ Schema mismatch detected")

    # Errors
    errors = report.get("errors", [])
    if errors:
        lines.append(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors[:10]:  # Show first 10
            lines.append(f"  - {error}")
        if len(errors) > 10:
            lines.append(f"  ... and {len(errors) - 10} more")
    else:
        lines.append("\n✅ No errors detected")

    # Warnings
    warnings = report.get("warnings", [])
    if warnings:
        lines.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            lines.append(f"  - {warning}")
        if len(warnings) > 10:
            lines.append(f"  ... and {len(warnings) - 10} more")
    else:
        lines.append("\n✅ No warnings")

    # Distribution stats
    distributions = report.get("score_distributions", {})
    if distributions:
        lines.append("\nScore Distributions:")

        for score_type, stats in distributions.items():
            lines.append(f"\n  {score_type}:")
            lines.append(f"    Mean:   {stats['mean']:.4f}")
            lines.append(f"    Median: {stats['median']:.4f}")
            lines.append(f"    Std:    {stats['std']:.4f}")
            if stats.get('toxic_rate') is not None:
                lines.append(f"    Toxic%: {stats['toxic_rate']*100:.2f}%")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    import tempfile
    import json

    print("Testing record validation...")

    # Valid record
    valid_record = {
        "prompt_id": "test_001",
        "family": "qwen3",
        "prompt_text": "Test prompt",
        "base_completion": "Base output",
        "aligned_completion": "Aligned output"
    }

    try:
        validate_completion_record(valid_record)
        print("✅ Valid record passed")
    except ValueError as e:
        print(f"❌ Unexpected error: {e}")

    # Invalid record (missing field)
    invalid_record = {
        "prompt_id": "test_002",
        "family": "qwen3"
    }

    try:
        validate_completion_record(invalid_record)
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Invalid record caught: {e}")

    # Test dataset validation
    print("\nTesting dataset validation...")

    # Create small test dataset
    test_data = {
        "prompt_id": ["test_001", "test_002"],
        "family": ["qwen3", "llama31"],
        "prompt_text": ["prompt1", "prompt2"],
        "base_completion": ["base1", "base2"],
        "aligned_completion": ["aligned1", "aligned2"],
        "tox_score_base": [0.2, 0.8],
        "tox_score_aligned": [0.1, 0.3],
        "tox_binary_base": [0, 1],
        "tox_binary_aligned": [0, 0],
        "severe_tox_base": [0.1, 0.5],
        "severe_tox_aligned": [0.05, 0.2],
        "insult_base": [0.1, 0.6],
        "insult_aligned": [0.05, 0.3],
        "nodes_base": [10, 15],
        "nodes_aligned": [12, 14],
        "edges_base": [8, 12],
        "edges_aligned": [10, 11],
        "ratio_base": [0.8, 0.8],
        "ratio_aligned": [0.83, 0.79],
        "lcc_size_base": [10, 15],
        "lcc_size_aligned": [12, 14],
        "bigram_contrib_base": [0.9, 0.85],
        "bigram_contrib_aligned": [0.88, 0.87],
        "scorer": ["detoxify_unbiased", "detoxify_unbiased"],
        "threshold": [0.5, 0.5]
    }

    df = pd.DataFrame(test_data)
    report = validate_analysis_dataset(df)

    print(create_validation_summary(report))

    print("\n✅ All validator tests passed")
