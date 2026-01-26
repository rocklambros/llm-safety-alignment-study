"""
Shared pytest fixtures for LLM Safety Alignment Study tests.

Provides:
    - sample_texts: Test texts (safe, toxic, empty, edge cases)
    - mock_openai_response: Mocked OpenAI moderation response
    - tmp_data_dir: Temporary directory fixture
    - --run-integration: Command line option to run integration tests
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


# =============================================================================
# Pytest Hooks for Integration Tests
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options.

    Options:
        --run-integration: Include integration tests in test run.
            Integration tests are skipped by default as they may take longer
            and require certain dependencies to be available.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (skipped by default)",
    )


def pytest_configure(config):
    """Register custom markers.

    Markers:
        integration: Mark test as integration test (requires --run-integration).
        unit: Mark test as unit test (always runs).
        slow: Mark test as slow (can deselect with -m "not slow").
    """
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires --run-integration)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests by default.

    Integration tests are skipped unless --run-integration is provided.
    This allows quick test runs during development while still supporting
    comprehensive testing when needed.
    """
    if config.getoption("--run-integration"):
        # --run-integration given: run all tests including integration
        return

    skip_integration = pytest.mark.skip(
        reason="Integration test: use --run-integration to run"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# =============================================================================
# Sample Text Fixtures
# =============================================================================


@pytest.fixture
def sample_texts():
    """
    Provide a list of test texts covering various scenarios.

    Returns:
        dict: Dictionary with categorized test texts
    """
    return {
        "safe": [
            "This is a perfectly normal message.",
            "Have a wonderful day!",
            "The quick brown fox jumps over the lazy dog.",
            "Thank you for your help with this project.",
        ],
        "toxic": [
            "I hate you and want to hurt you.",
            "You are a terrible person and should be ashamed.",
            "I will destroy everything you care about.",
        ],
        "empty": [
            "",
            "   ",
            "\n\t",
        ],
        "edge_cases": [
            "Hello",  # Single word
            "a b",    # Minimal bigram
            "123 456 789",  # Numbers only (no alphabetic words)
            "!@#$%^&*()",  # Special characters only
            "a" * 10000,   # Very long repeated character
        ],
    }


# =============================================================================
# OpenAI Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_moderation_result():
    """
    Create a mock OpenAI moderation API result.

    Returns:
        MagicMock: Mock response mimicking OpenAI Moderation API structure
    """
    mock_result = MagicMock()
    mock_result.flagged = False

    # Create mock category scores
    mock_scores = MagicMock()
    mock_scores.harassment = 0.001
    mock_scores.harassment_threatening = 0.0001
    mock_scores.hate = 0.002
    mock_scores.hate_threatening = 0.0001
    mock_scores.violence = 0.003
    mock_scores.violence_graphic = 0.0001
    mock_scores.self_harm = 0.001
    mock_scores.self_harm_intent = 0.0001
    mock_scores.self_harm_instructions = 0.0001
    mock_scores.sexual = 0.001
    mock_scores.sexual_minors = 0.0001

    mock_result.category_scores = mock_scores

    return mock_result


@pytest.fixture
def mock_openai_moderation_result_toxic():
    """
    Create a mock OpenAI moderation API result for toxic content.

    Returns:
        MagicMock: Mock response with high toxicity scores
    """
    mock_result = MagicMock()
    mock_result.flagged = True

    mock_scores = MagicMock()
    mock_scores.harassment = 0.85
    mock_scores.harassment_threatening = 0.6
    mock_scores.hate = 0.75
    mock_scores.hate_threatening = 0.4
    mock_scores.violence = 0.9
    mock_scores.violence_graphic = 0.3
    mock_scores.self_harm = 0.2
    mock_scores.self_harm_intent = 0.1
    mock_scores.self_harm_instructions = 0.05
    mock_scores.sexual = 0.05
    mock_scores.sexual_minors = 0.01

    mock_result.category_scores = mock_scores

    return mock_result


@pytest.fixture
def mock_openai_response(mock_openai_moderation_result):
    """
    Create a full mock OpenAI moderation response object.

    Args:
        mock_openai_moderation_result: The fixture for mock result

    Returns:
        MagicMock: Mock response object with results list
    """
    mock_response = MagicMock()
    mock_response.results = [mock_openai_moderation_result]
    return mock_response


@pytest.fixture
def mock_openai_response_toxic(mock_openai_moderation_result_toxic):
    """
    Create a mock OpenAI moderation response for toxic content.

    Args:
        mock_openai_moderation_result_toxic: The fixture for toxic mock result

    Returns:
        MagicMock: Mock response object with toxic results
    """
    mock_response = MagicMock()
    mock_response.results = [mock_openai_moderation_result_toxic]
    return mock_response


# =============================================================================
# Directory and Environment Fixtures
# =============================================================================


@pytest.fixture
def tmp_data_dir():
    """
    Create a temporary directory for test data.

    Yields:
        Path: Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_openai_key(monkeypatch):
    """
    Set a mock OpenAI API key in the environment.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-for-testing")


# =============================================================================
# Scorer Test Case Fixtures
# =============================================================================


@pytest.fixture
def detoxify_test_cases():
    """
    Provide specific test cases for Detoxify scorer validation.

    Returns:
        list: List of tuples (text, expected_toxicity_range)
    """
    return [
        # (text, min_toxicity, max_toxicity)
        ("This is a wonderful day!", 0.0, 0.2),
        ("Thank you for your kindness.", 0.0, 0.2),
        ("I hate you and want to hurt you.", 0.5, 1.0),
        ("You are stupid and worthless.", 0.3, 1.0),
    ]


@pytest.fixture
def xfakesci_test_cases():
    """
    Provide specific test cases for xFakeSci feature extraction.

    Returns:
        list: List of tuples (text, expected_features)
    """
    return [
        # (text, expected_nodes, expected_edges_min)
        ("The quick brown fox jumps over the lazy dog.", 8, 7),
        ("Hello hello hello hello.", 1, 0),  # Single unique word, no bigrams
        ("a b c d e f g", 7, 6),  # Linear chain
    ]
