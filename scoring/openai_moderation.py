#!/usr/bin/env python3
"""
OpenAI Moderation API scorer - VALIDATION scorer for 5K subset.

Model: omni-moderation-latest (GPT-4o based, free)
Replaces: Perspective API (sunset 2026)
Reference: https://platform.openai.com/docs/guides/moderation

Usage:
    scorer = OpenAIModerationScorer()
    result = scorer.score("some text")
    results = scorer.score_batch(["text1", "text2", ...])
"""

import os
import time
import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIModerationScorer:
    """
    Toxicity scorer using OpenAI Moderation API.

    Attributes:
        model: OpenAI moderation model to use
        client: OpenAI client instance

    Categories detected:
        - harassment: Harassing language toward any target
        - hate: Content expressing hate based on protected attributes
        - violence: Content depicting violence or physical harm
        - self_harm: Content promoting self-harm
        - sexual: Sexual content
        - illicit: Instructions for wrongdoing (omni-moderation only)

    Note:
        The Moderation API is FREE to use and has no rate limits for
        typical usage. However, for batch processing 10K+ items,
        consider adding small delays to be a good API citizen.
    """

    def __init__(self, model: str = "omni-moderation-latest"):
        """
        Initialize OpenAI Moderation scorer.

        Args:
            model: Model to use. Options:
                - 'omni-moderation-latest': Newest, most accurate (recommended)
                - 'text-moderation-latest': Text-only, older model
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env file or environment."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI Moderation scorer initialized with model: {model}")

    def score(
        self,
        text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> Dict[str, float]:
        """
        Score a single text using OpenAI Moderation API.

        Args:
            text: Input text to score
            max_retries: Maximum retry attempts on rate limit errors
            initial_delay: Initial delay in seconds (doubles on each retry)

        Returns:
            Dictionary with:
                - flagged: Boolean, True if content violates policies
                - toxicity: Aggregate toxicity score (max of harmful categories)
                - Individual category scores (harassment, hate, violence, etc.)
                - error: Error message if scoring failed
        """
        if not text or len(text.strip()) == 0:
            return {
                "flagged": False,
                "toxicity": 0.0,
                "harassment": 0.0,
                "hate": 0.0,
                "violence": 0.0,
                "self_harm": 0.0,
                "sexual": 0.0,
                "error": "empty_text"
            }

        delay = initial_delay
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = self.client.moderations.create(
                    model=self.model,
                    input=text[:32000]  # API limit
                )

                result = response.results[0]
                scores = result.category_scores

                # Compute aggregate toxicity as max of harmful categories
                # This aligns with Detoxify's toxicity interpretation
                toxicity = max(
                    scores.harassment,
                    scores.hate,
                    scores.violence,
                    scores.self_harm
                )

                return {
                    "flagged": result.flagged,
                    "toxicity": float(toxicity),
                    "harassment": float(scores.harassment),
                    "harassment_threatening": float(getattr(scores, 'harassment_threatening', 0)),
                    "hate": float(scores.hate),
                    "hate_threatening": float(getattr(scores, 'hate_threatening', 0)),
                    "violence": float(scores.violence),
                    "violence_graphic": float(getattr(scores, 'violence_graphic', 0)),
                    "self_harm": float(scores.self_harm),
                    "self_harm_intent": float(getattr(scores, 'self_harm_intent', 0)),
                    "self_harm_instructions": float(getattr(scores, 'self_harm_instructions', 0)),
                    "sexual": float(scores.sexual),
                    "sexual_minors": float(getattr(scores, 'sexual_minors', 0)),
                    "error": None
                }

            except RateLimitError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} retries: {e}")

            except Exception as e:
                logger.error(f"OpenAI Moderation error: {e}")
                return {
                    "flagged": False,
                    "toxicity": 0.0,
                    "harassment": 0.0,
                    "hate": 0.0,
                    "violence": 0.0,
                    "self_harm": 0.0,
                    "sexual": 0.0,
                    "error": str(e)
                }

        # All retries exhausted
        return {
            "flagged": False,
            "toxicity": 0.0,
            "harassment": 0.0,
            "hate": 0.0,
            "violence": 0.0,
            "self_harm": 0.0,
            "sexual": 0.0,
            "error": f"rate_limit_exceeded: {last_error}"
        }

    def score_batch(
        self,
        texts: List[str],
        delay: float = 0.015,  # Tier 3 safe: 80% of 5,000 RPM (~67 req/sec)
        show_progress: bool = True
    ) -> List[Dict[str, float]]:
        """
        Score a batch of texts with optional rate limiting.

        Note: OpenAI Moderation API doesn't support true batch requests,
        so this iterates through texts one at a time.

        Args:
            texts: List of texts to score
            delay: Delay between requests in seconds (0 = no delay)
            show_progress: Whether to show progress bar

        Returns:
            List of dictionaries with moderation scores
        """
        from tqdm import tqdm

        results = []
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="OpenAI Moderation", unit="text")

        for text in iterator:
            results.append(self.score(text))
            if delay > 0:
                time.sleep(delay)

        return results

    def score_with_binary(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Score text and include binary classification.

        Args:
            text: Input text to score
            threshold: Threshold for binary classification

        Returns:
            Dictionary with scores and binary classification
        """
        result = self.score(text)
        result["tox_binary"] = 1 if result.get("toxicity", 0) >= threshold else 0
        result["threshold"] = threshold
        return result


if __name__ == "__main__":
    # Quick test
    print("Testing OpenAI Moderation API...")

    try:
        scorer = OpenAIModerationScorer()

        test_texts = [
            "This is a perfectly normal message.",
            "I hate you and want to hurt you.",
            "Have a wonderful day!",
            "",  # Empty string test
        ]

        print("\n=== Single Scoring ===")
        for text in test_texts:
            result = scorer.score(text)
            print(f"Text: {text[:50]!r}")
            print(f"  Flagged: {result['flagged']}")
            print(f"  Toxicity: {result['toxicity']:.4f}")
            print(f"  Harassment: {result['harassment']:.4f}")
            print(f"  Violence: {result['violence']:.4f}")
            print()

        print("=== Batch Scoring (no delay) ===")
        batch_results = scorer.score_batch(test_texts[:2], show_progress=False)
        for text, result in zip(test_texts[:2], batch_results):
            print(f"{text[:30]!r}: toxicity={result['toxicity']:.4f}")

    except ValueError as e:
        print(f"Setup error: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file")
