#!/usr/bin/env python3
"""
Download ToxiGen dataset from HuggingFace.

Downloads the train split from toxigen/toxigen-data and saves as CSV.
Handles authentication and provides progress feedback.

Dataset: toxigen/toxigen-data (train split)
Expected: 274,186 rows with text, target_group columns

Usage:
    python download_toxigen.py

    # If authentication is needed:
    export HF_TOKEN=your_token_here
    python download_toxigen.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Lazy import to handle missing datasets gracefully
try:
    from datasets import load_dataset, DownloadConfig
    from datasets.exceptions import DatasetNotFoundError
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Configuration
DATASET_NAME = "toxigen/toxigen-data"
DATASET_SPLIT = "train"
EXPECTED_ROWS = 274186
OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = OUTPUT_DIR / "toxigen_train.csv"

# Required columns in the output
REQUIRED_COLUMNS = ["text", "target_group"]

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


def get_hf_token() -> Optional[str]:
    """
    Retrieve HuggingFace token from environment.

    Returns:
        Token string or None if not set
    """
    # Check multiple possible environment variable names
    for var_name in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        token = os.environ.get(var_name)
        if token:
            logger.info(f"Using HuggingFace token from ${var_name}")
            return token

    # Also check if logged in via huggingface-cli
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            logger.info("Using cached HuggingFace token")
            return token
    except Exception:
        pass

    logger.info("No HuggingFace token found (may not be needed for public datasets)")
    return None


def download_toxigen_dataset(
    dataset_name: str,
    split: str,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download ToxiGen dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to download
        token: Optional HuggingFace token for authentication

    Returns:
        DataFrame containing the dataset

    Raises:
        DownloadError: If download fails
    """
    if not DATASETS_AVAILABLE:
        raise DownloadError(
            "The 'datasets' library is not installed. "
            "Please install it with: pip install datasets"
        )

    logger.info(f"Downloading {dataset_name} ({split} split)...")
    logger.info("This may take several minutes depending on your connection...")

    try:
        # Configure download settings
        download_config = DownloadConfig(
            resume_download=True,  # Resume interrupted downloads
            max_retries=3,
        )

        # Load the dataset
        # Note: ToxiGen uses "train" split
        dataset = load_dataset(
            dataset_name,
            split=split,
            token=token,
            download_config=download_config,
            trust_remote_code=False,  # Security: don't execute remote code
        )

        logger.info(f"Downloaded {len(dataset)} rows")

        # Convert to pandas DataFrame
        logger.info("Converting to DataFrame...")
        df = dataset.to_pandas()

        return df

    except DatasetNotFoundError as e:
        raise DownloadError(
            f"Dataset not found: {dataset_name}. "
            "This may require authentication. "
            "Set HF_TOKEN environment variable and try again."
        ) from e
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            raise DownloadError(
                "Authentication required. Please set HF_TOKEN environment variable "
                "with your HuggingFace access token."
            ) from e
        elif "403" in error_msg or "forbidden" in error_msg:
            raise DownloadError(
                "Access denied. You may need to accept the dataset's terms of use "
                "on HuggingFace before downloading."
            ) from e
        else:
            raise DownloadError(f"Failed to download dataset: {e}") from e


def validate_dataframe(
    df: pd.DataFrame,
    expected_rows: int,
    required_columns: list,
) -> None:
    """
    Validate the downloaded DataFrame.

    Args:
        df: DataFrame to validate
        expected_rows: Expected number of rows
        required_columns: List of required column names

    Raises:
        ValidationError: If validation fails
    """
    logger.info("Validating dataset...")

    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        available = list(df.columns)
        raise ValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available}"
        )

    # Check row count
    actual_rows = len(df)
    logger.info(f"Actual rows: {actual_rows}")
    logger.info(f"Expected rows: {expected_rows}")

    # Allow some tolerance for dataset updates
    if actual_rows < expected_rows * 0.95:
        raise ValidationError(
            f"Row count {actual_rows} is significantly below expected {expected_rows}"
        )

    if actual_rows > expected_rows * 1.05:
        logger.warning(
            f"Row count {actual_rows} is higher than expected {expected_rows}. "
            "Dataset may have been updated."
        )

    # Check for null values in required columns
    for col in required_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_pct = 100 * null_count / actual_rows
            logger.warning(
                f"Column '{col}' has {null_count} null values ({null_pct:.2f}%)"
            )

    # Validate text column has content
    if "text" in df.columns:
        empty_text = (df["text"].str.strip() == "").sum()
        if empty_text > 0:
            logger.warning(f"Found {empty_text} rows with empty text")

    logger.info("Validation passed")


def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame to CSV with progress indication.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    logger.info(f"Saving to {output_path}...")

    # Save with explicit encoding and quoting for safety
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=1,  # QUOTE_ALL for safety with potentially malicious text
    )

    # Verify the file was written
    if not output_path.exists():
        raise ValidationError(f"Failed to write file: {output_path}")

    file_size = output_path.stat().st_size
    logger.info(f"Saved {file_size / (1024 * 1024):.1f} MB to {output_path}")


def validate_existing_csv(file_path: Path, expected_rows: int) -> int:
    """
    Validate an existing CSV file.

    Args:
        file_path: Path to the CSV file
        expected_rows: Expected number of rows

    Returns:
        Actual row count

    Raises:
        ValidationError: If validation fails
    """
    logger.info(f"Validating existing file: {file_path}")

    try:
        # Read only headers to check columns
        df_sample = pd.read_csv(file_path, nrows=0)
        missing_columns = set(REQUIRED_COLUMNS) - set(df_sample.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")

        # Count rows efficiently
        row_count = sum(1 for _ in open(file_path, encoding="utf-8")) - 1  # Subtract header

        if row_count < expected_rows * 0.95:
            raise ValidationError(
                f"Row count {row_count} is below expected {expected_rows}"
            )

        return row_count

    except pd.errors.ParserError as e:
        raise ValidationError(f"Invalid CSV format: {e}") from e


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset."""
    logger.info("-" * 40)
    logger.info("Dataset Summary:")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")

    if "target_group" in df.columns:
        target_counts = df["target_group"].value_counts()
        logger.info(f"  Target groups: {len(target_counts)}")
        logger.info("  Top 5 target groups:")
        for group, count in target_counts.head(5).items():
            logger.info(f"    - {group}: {count:,}")

    if "text" in df.columns:
        avg_length = df["text"].str.len().mean()
        logger.info(f"  Average text length: {avg_length:.0f} characters")

    logger.info("-" * 40)


def main() -> int:
    """
    Main entry point for ToxiGen download.

    Returns:
        0 on success, 1 on failure
    """
    logger.info("=" * 60)
    logger.info("ToxiGen Dataset Downloader")
    logger.info("=" * 60)

    # Check dependencies
    if not DATASETS_AVAILABLE:
        logger.error(
            "The 'datasets' library is required but not installed. "
            "Install it with: pip install datasets"
        )
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Check if file already exists
    if OUTPUT_FILE.exists():
        logger.info(f"File already exists: {OUTPUT_FILE}")
        try:
            row_count = validate_existing_csv(OUTPUT_FILE, EXPECTED_ROWS)
            logger.info(f"Existing file is valid with {row_count:,} rows")
            logger.info("SUCCESS: ToxiGen dataset ready")
            return 0
        except ValidationError as e:
            logger.warning(f"Existing file invalid: {e}")
            logger.info("Re-downloading...")

    try:
        # Get authentication token
        token = get_hf_token()

        # Download dataset
        df = download_toxigen_dataset(
            dataset_name=DATASET_NAME,
            split=DATASET_SPLIT,
            token=token,
        )

        # Validate
        validate_dataframe(df, EXPECTED_ROWS, REQUIRED_COLUMNS)

        # Print summary
        print_dataset_summary(df)

        # Save to CSV
        save_to_csv(df, OUTPUT_FILE)

        logger.info("=" * 60)
        logger.info("SUCCESS: ToxiGen dataset downloaded")
        logger.info(f"Location: {OUTPUT_FILE.resolve()}")
        logger.info(f"Rows: {len(df):,}")
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
    except KeyboardInterrupt:
        logger.info("Download cancelled by user")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
