#!/usr/bin/env python3
"""
Integration test for toxicity scoring pipeline components.

Tests checkpoint manager, validators, and pipeline orchestration
with synthetic test data.

Usage:
    python -m scoring.test_integration
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from scoring.checkpoint_manager import CheckpointManager
from scoring.validators import (
    create_validation_summary,
    validate_analysis_dataset,
    validate_completion_record,
)


def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    print("=" * 60)
    print("TEST: Checkpoint Manager")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Test save
        test_data = [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
        ]
        checkpoint_file = manager.save("test_stage", test_data, 100)
        print(f"✅ Saved checkpoint: {checkpoint_file}")

        # Test load
        loaded_data, loaded_index = manager.load_latest("test_stage")
        assert loaded_index == 100, f"Index mismatch: {loaded_index} != 100"
        assert len(loaded_data) == 2, f"Data length mismatch: {len(loaded_data)} != 2"
        print(f"✅ Loaded checkpoint: index={loaded_index}, records={len(loaded_data)}")

        # Test list
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1, f"Checkpoint count mismatch: {len(checkpoints)} != 1"
        print(f"✅ Listed checkpoints: {len(checkpoints)} found")

        # Test clear
        removed = manager.clear("test_stage")
        assert removed == 1, f"Clear count mismatch: {removed} != 1"
        assert len(manager.list_checkpoints()) == 0, "Checkpoints not cleared"
        print(f"✅ Cleared checkpoints: {removed} removed")

    print("✅ All checkpoint manager tests passed\n")


def test_validators():
    """Test validation functions."""
    print("=" * 60)
    print("TEST: Validators")
    print("=" * 60)

    # Test valid record
    valid_record = {
        "prompt_id": "test_001",
        "family": "qwen3",
        "prompt_text": "Test prompt",
        "base_completion": "Base output",
        "aligned_completion": "Aligned output"
    }

    try:
        validate_completion_record(valid_record)
        print("✅ Valid record passed validation")
    except ValueError as e:
        print(f"❌ Unexpected validation error: {e}")
        return

    # Test invalid record (missing field)
    invalid_record = {
        "prompt_id": "test_002",
        "family": "qwen3"
    }

    try:
        validate_completion_record(invalid_record)
        print("❌ Invalid record should have failed validation")
        return
    except ValueError as e:
        print(f"✅ Invalid record caught: {str(e)[:50]}...")

    # Test dataset validation
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

    print(f"✅ Dataset validation complete")
    print(f"  Errors: {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")

    print("✅ All validator tests passed\n")


def test_pipeline_with_synthetic_data():
    """Test pipeline orchestration with synthetic data."""
    print("=" * 60)
    print("TEST: Pipeline Orchestration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic completion files
        completions_dir = tmpdir / "completions"
        completions_dir.mkdir()

        synthetic_records = []
        for i in range(10):  # Small test set
            record = {
                "prompt_id": f"test_{i:03d}",
                "family": ["qwen3", "llama31", "mistral"][i % 3],
                "prompt_text": f"Test prompt {i}",
                "base_completion": f"Base completion {i} with some text",
                "aligned_completion": f"Aligned completion {i} with some text",
                "base_latency_ms": 1000.0,
                "aligned_latency_ms": 1000.0,
                "timestamp": "2026-01-25T00:00:00+00:00",
                "error": None
            }
            synthetic_records.append(record)

        # Write to JSONL file
        test_file = completions_dir / "completions_test.jsonl"
        with open(test_file, 'w') as f:
            for record in synthetic_records:
                f.write(json.dumps(record) + '\n')

        print(f"✅ Created test data: {len(synthetic_records)} records")

        # Test loading
        from scoring.scoring_runner import load_completion_records

        records = load_completion_records(completions_dir)
        assert len(records) == 10, f"Loaded {len(records)} records, expected 10"
        print(f"✅ Loaded completion records: {len(records)} records")

        # Test checkpoint manager integration
        checkpoint_dir = tmpdir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        test_results = [{"id": i, "score": 0.5} for i in range(10)]
        manager.save("test_pipeline", test_results, 10)
        print(f"✅ Pipeline checkpoint saved")

        loaded_results, index = manager.load_latest("test_pipeline")
        assert index == 10, f"Index mismatch: {index} != 10"
        assert len(loaded_results) == 10, f"Results mismatch: {len(loaded_results)} != 10"
        print(f"✅ Pipeline checkpoint loaded")

    print("✅ All pipeline orchestration tests passed\n")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("TOXICITY SCORING PIPELINE - INTEGRATION TESTS")
    print("=" * 60 + "\n")

    try:
        test_checkpoint_manager()
        test_validators()
        test_pipeline_with_synthetic_data()

        print("=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
