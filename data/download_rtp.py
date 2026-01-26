#!/usr/bin/env python3
"""
Download RealToxicityPrompts dataset from AI2.

Downloads the tarball, extracts the JSONL file, and validates the expected row count.
Implements retry logic with exponential backoff for network resilience.

URL: https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.tar.gz
Expected: 99,442 rows with prompt.text and prompt.toxicity fields

Usage:
    python download_rtp.py
"""

import hashlib
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Configuration
RTP_URL = "https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.tar.gz"
EXPECTED_ROWS = 99442
OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = OUTPUT_DIR / "realtoxicityprompts.jsonl"
JSONL_FILENAME_IN_TAR = "realtoxicityprompts-data/prompts.jsonl"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5
BACKOFF_MULTIPLIER = 2
CHUNK_SIZE = 8192  # 8KB chunks for streaming download

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Custom exception for download failures."""

    pass


class ValidationError(Exception):
    """Custom exception for validation failures."""

    pass


def download_with_retry(
    url: str,
    dest_path: Path,
    max_retries: int = MAX_RETRIES,
    timeout: int = 30,
) -> None:
    """
    Download a file with retry logic and progress display.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Raises:
        DownloadError: If download fails after all retries
    """
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries}: {url}")

            # Use streaming to handle large files efficiently
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Get total size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress display
            with open(dest_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading RTP",
                    disable=total_size == 0,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Download complete: {dest_path}")
            return

        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout on attempt {attempt}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt}: {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error on attempt {attempt}: {e}")
            if response.status_code in (400, 401, 403, 404):
                # Don't retry client errors
                raise DownloadError(f"HTTP {response.status_code}: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt}: {e}")

        if attempt < max_retries:
            logger.info(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER
        else:
            raise DownloadError(f"Download failed after {max_retries} attempts")


def extract_jsonl_from_tarball(
    tarball_path: Path,
    output_path: Path,
    member_name: str,
) -> None:
    """
    Extract a specific file from a tarball.

    Args:
        tarball_path: Path to the .tar.gz file
        output_path: Path to write the extracted file
        member_name: Name of the file within the tarball to extract

    Raises:
        ValidationError: If the file is not found in the tarball
    """
    logger.info(f"Extracting {member_name} from tarball...")

    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Security: Validate member names to prevent path traversal
            for member in tar.getmembers():
                if ".." in member.name or member.name.startswith("/"):
                    raise ValidationError(
                        f"Suspicious path in tarball: {member.name}"
                    )

            # Find and extract the target file
            try:
                member = tar.getmember(member_name)
            except KeyError:
                # List available files for debugging
                available = [m.name for m in tar.getmembers()]
                logger.error(f"Available files in tarball: {available[:10]}...")
                raise ValidationError(
                    f"File not found in tarball: {member_name}"
                )

            # Extract to a temporary location first, then move
            with tar.extractfile(member) as src:
                if src is None:
                    raise ValidationError(f"Could not read {member_name} from tarball")

                with open(output_path, "wb") as dst:
                    # Stream copy with progress
                    total_size = member.size
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Extracting",
                    ) as pbar:
                        while True:
                            chunk = src.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)
                            pbar.update(len(chunk))

        logger.info(f"Extraction complete: {output_path}")

    except tarfile.TarError as e:
        raise ValidationError(f"Failed to read tarball: {e}") from e


def validate_jsonl(file_path: Path, expected_rows: int) -> int:
    """
    Validate the JSONL file structure and row count.

    Args:
        file_path: Path to the JSONL file
        expected_rows: Expected number of rows

    Returns:
        Actual row count

    Raises:
        ValidationError: If validation fails
    """
    logger.info(f"Validating {file_path}...")

    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")

    row_count = 0
    missing_fields_count = 0
    required_fields = {"prompt"}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Validating", unit="rows"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Invalid JSON on line {line_num}: {e}"
                ) from e

            # Check for required fields
            if "prompt" not in data:
                missing_fields_count += 1
                if missing_fields_count <= 5:
                    logger.warning(f"Line {line_num}: missing 'prompt' field")
            else:
                prompt = data["prompt"]
                if not isinstance(prompt, dict):
                    missing_fields_count += 1
                elif "text" not in prompt:
                    missing_fields_count += 1

            row_count += 1

    # Log validation results
    logger.info(f"Total rows: {row_count}")
    logger.info(f"Expected rows: {expected_rows}")

    if missing_fields_count > 0:
        logger.warning(
            f"Rows with missing prompt.text: {missing_fields_count} "
            f"({100 * missing_fields_count / row_count:.2f}%)"
        )

    # Validate row count
    if row_count < expected_rows * 0.99:  # Allow 1% tolerance
        raise ValidationError(
            f"Row count {row_count} is significantly below expected {expected_rows}"
        )

    return row_count


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> int:
    """
    Main entry point for RealToxicityPrompts download.

    Returns:
        0 on success, 1 on failure
    """
    logger.info("=" * 60)
    logger.info("RealToxicityPrompts Dataset Downloader")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Check if file already exists
    if OUTPUT_FILE.exists():
        logger.info(f"File already exists: {OUTPUT_FILE}")
        try:
            row_count = validate_jsonl(OUTPUT_FILE, EXPECTED_ROWS)
            logger.info(f"Existing file is valid with {row_count} rows")
            logger.info("SUCCESS: RealToxicityPrompts dataset ready")
            return 0
        except ValidationError as e:
            logger.warning(f"Existing file invalid: {e}")
            logger.info("Re-downloading...")

    # Download to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_path = Path(tmpdir) / "realtoxicityprompts-data.tar.gz"

        try:
            # Download
            download_with_retry(RTP_URL, tarball_path)

            # Extract
            extract_jsonl_from_tarball(
                tarball_path,
                OUTPUT_FILE,
                JSONL_FILENAME_IN_TAR,
            )

            # Validate
            row_count = validate_jsonl(OUTPUT_FILE, EXPECTED_ROWS)

            # Compute and log file hash for integrity verification
            file_hash = compute_file_hash(OUTPUT_FILE)
            logger.info(f"File SHA256: {file_hash}")

            logger.info("=" * 60)
            logger.info("SUCCESS: RealToxicityPrompts dataset downloaded")
            logger.info(f"Location: {OUTPUT_FILE.resolve()}")
            logger.info(f"Rows: {row_count}")
            logger.info("=" * 60)
            return 0

        except DownloadError as e:
            logger.error(f"Download failed: {e}")
            return 1
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            # Clean up invalid file
            if OUTPUT_FILE.exists():
                OUTPUT_FILE.unlink()
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
