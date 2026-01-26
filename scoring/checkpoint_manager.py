#!/usr/bin/env python3
"""
Checkpoint management for resumable toxicity scoring pipeline.

Uses JSON-based serialization for security (no unsafe deserialization).
Supports incremental checkpoints for fault tolerance during long-running operations.

Security:
- JSON only (safe serialization format)
- Path validation to prevent traversal attacks
- No sensitive data in checkpoint files

Usage:
    manager = CheckpointManager("output/checkpoints/")

    # Save checkpoint
    manager.save("detoxify", data, index=5000)

    # Load latest checkpoint
    data, index = manager.load_latest("detoxify")

    # Resume from checkpoint
    if checkpoint_exists:
        data, start_index = manager.load_latest("detoxify")
    else:
        data, start_index = [], 0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages JSON-based checkpoints for resumable processing.

    Attributes:
        checkpoint_dir: Directory for storing checkpoint files
    """

    def __init__(self, checkpoint_dir: str | Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory path for checkpoint storage.
                Will be created if it doesn't exist.
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save(
        self,
        stage: str,
        data: List[Dict[str, Any]],
        index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save checkpoint for given stage.

        Args:
            stage: Stage identifier (e.g., "detoxify", "xfakesci")
            data: List of result dictionaries to checkpoint
            index: Current processing index (for resume)
            metadata: Optional metadata dict

        Returns:
            Path to saved checkpoint file

        Raises:
            ValueError: If stage name is invalid
            OSError: If checkpoint cannot be written
        """
        # Validate stage name (alphanumeric + underscore only)
        if not stage.replace("_", "").isalnum():
            raise ValueError(f"Invalid stage name: {stage}")

        checkpoint_file = self.checkpoint_dir / f"{stage}_{index}.json"

        checkpoint = {
            "stage": stage,
            "index": index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_count": len(data),
            "data": data,
            "metadata": metadata or {}
        }

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(
                f"Checkpoint saved: {checkpoint_file.name} "
                f"(index={index}, records={len(data)})"
            )

            return checkpoint_file

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_file}: {e}")
            raise

    def load_latest(self, stage: str) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        """Load most recent checkpoint for stage.

        Args:
            stage: Stage identifier (e.g., "detoxify", "xfakesci")

        Returns:
            Tuple of (data, index) if checkpoint exists, None otherwise

        Raises:
            ValueError: If checkpoint data is invalid
            OSError: If checkpoint cannot be read
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{stage}_*.json"))

        if not checkpoints:
            logger.info(f"No checkpoints found for stage: {stage}")
            return None

        # Get latest checkpoint by index
        try:
            latest = max(
                checkpoints,
                key=lambda p: int(p.stem.split('_')[-1])
            )
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid checkpoint filename format: {e}")
            return None

        try:
            with open(latest, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # Validate checkpoint structure
            required_fields = ["stage", "index", "data"]
            for field in required_fields:
                if field not in checkpoint:
                    raise ValueError(f"Missing required field: {field}")

            data = checkpoint["data"]
            index = checkpoint["index"]

            logger.info(
                f"Loaded checkpoint: {latest.name} "
                f"(index={index}, records={len(data)})"
            )

            return data, index

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted checkpoint file {latest}: {e}")
            raise ValueError(f"Corrupted checkpoint: {latest}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest}: {e}")
            raise

    def list_checkpoints(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all checkpoints, optionally filtered by stage.

        Args:
            stage: Optional stage filter

        Returns:
            List of checkpoint info dicts
        """
        pattern = f"{stage}_*.json" if stage else "*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        checkpoint_info = []
        for cp in sorted(checkpoints):
            try:
                # Extract stage and index from filename
                stem = cp.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2:
                    stage_name, index_str = parts
                    checkpoint_info.append({
                        "file": cp.name,
                        "stage": stage_name,
                        "index": int(index_str),
                        "size_bytes": cp.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            cp.stat().st_mtime,
                            tz=timezone.utc
                        ).isoformat()
                    })
            except Exception as e:
                logger.warning(f"Skipping invalid checkpoint {cp}: {e}")

        return checkpoint_info

    def clear(self, stage: Optional[str] = None) -> int:
        """Remove checkpoints for stage (or all if stage=None).

        Args:
            stage: Optional stage filter. If None, removes all checkpoints.

        Returns:
            Number of checkpoints removed
        """
        pattern = f"{stage}_*.json" if stage else "*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        removed_count = 0
        for cp in checkpoints:
            try:
                cp.unlink()
                logger.info(f"Removed checkpoint: {cp.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {cp}: {e}")

        logger.info(f"Cleared {removed_count} checkpoint(s)")
        return removed_count

    def get_resume_point(self, stage: str) -> int:
        """Get index to resume from for given stage.

        Args:
            stage: Stage identifier

        Returns:
            Index to resume from (0 if no checkpoint exists)
        """
        result = self.load_latest(stage)
        if result is None:
            return 0

        _, index = result
        return index

    def checkpoint_exists(self, stage: str) -> bool:
        """Check if checkpoint exists for stage.

        Args:
            stage: Stage identifier

        Returns:
            True if at least one checkpoint exists
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{stage}_*.json"))
        return len(checkpoints) > 0


# Utility functions for common checkpoint patterns

def save_incremental_checkpoint(
    manager: CheckpointManager,
    stage: str,
    all_data: List[Dict[str, Any]],
    current_index: int,
    checkpoint_interval: int
) -> None:
    """Save checkpoint if interval reached.

    Args:
        manager: CheckpointManager instance
        stage: Stage identifier
        all_data: Complete data list up to current index
        current_index: Current processing index (1-based)
        checkpoint_interval: Save every N records
    """
    if current_index % checkpoint_interval == 0:
        manager.save(stage, all_data, current_index)


def load_or_initialize(
    manager: CheckpointManager,
    stage: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Load checkpoint or initialize empty state.

    Args:
        manager: CheckpointManager instance
        stage: Stage identifier

    Returns:
        Tuple of (data, start_index)
    """
    result = manager.load_latest(stage)

    if result is None:
        logger.info(f"Starting {stage} from beginning (no checkpoint)")
        return [], 0

    data, index = result
    logger.info(f"Resuming {stage} from index {index}")
    return data, index


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Test save
        test_data = [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
        ]
        checkpoint_file = manager.save("test_stage", test_data, 100)
        print(f"Saved: {checkpoint_file}")

        # Test load
        loaded_data, loaded_index = manager.load_latest("test_stage")
        print(f"Loaded: index={loaded_index}, records={len(loaded_data)}")
        assert loaded_index == 100
        assert len(loaded_data) == 2

        # Test list
        checkpoints = manager.list_checkpoints()
        print(f"Checkpoints: {checkpoints}")

        # Test clear
        removed = manager.clear("test_stage")
        print(f"Removed: {removed} checkpoint(s)")

        assert manager.list_checkpoints() == []

        print("\n✅ All checkpoint manager tests passed")
