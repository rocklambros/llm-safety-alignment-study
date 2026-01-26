"""
Toxicity scoring module for LLM Safety Alignment Study.

Scorers:
    - DetoxifyScorer: Primary scorer (local, free, fast)
    - OpenAIModerationScorer: Validation scorer (API, free)
    - extract_xfakesci_features: Bigram network features

Usage:
    from scoring import DetoxifyScorer, OpenAIModerationScorer
    from scoring import extract_xfakesci_features

    # Primary scoring
    detox = DetoxifyScorer()
    result = detox.score("some text")

    # Validation scoring
    openai_scorer = OpenAIModerationScorer()
    result = openai_scorer.score("some text")

    # Feature extraction
    features = extract_xfakesci_features("some text")
"""

from scoring.checkpoint_manager import CheckpointManager
from scoring.detoxify_scorer import DetoxifyScorer
from scoring.openai_moderation import OpenAIModerationScorer
from scoring.validators import (
    validate_analysis_dataset,
    validate_completion_record,
    create_validation_summary
)
from scoring.xfakesci_features import (
    extract_xfakesci_features,
    extract_xfakesci_batch,
    get_network_stats
)

__all__ = [
    "CheckpointManager",
    "DetoxifyScorer",
    "OpenAIModerationScorer",
    "validate_analysis_dataset",
    "validate_completion_record",
    "create_validation_summary",
    "extract_xfakesci_features",
    "extract_xfakesci_batch",
    "get_network_stats",
]
