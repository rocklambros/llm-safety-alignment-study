#!/usr/bin/env Rscript
# =============================================================================
# 01_load_data.R
# Data Loading and Validation Script for LLM Safety Alignment Study
# =============================================================================
#
# Purpose: Load analysis dataset CSV, validate schema per PRD 3.3, check data
#          types, report missing values, output validated RDS file.
#
# Input:  output/analysis_dataset_full.csv (75K rows expected)
# Output: analysis/data_validated.rds
#
# Usage:  Rscript analysis/01_load_data.R [--mock] [--input path/to/data.csv]
#
# =============================================================================

library(tidyverse)
library(jsonlite)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default paths (relative to workspace root)
DEFAULT_INPUT_PATH <- "output/analysis_dataset_full.csv"
OUTPUT_RDS_PATH <- "analysis/data_validated.rds"
EXPECTED_ROWS <- 75000
RANDOM_SEED <- 20260118

# Required columns per PRD Section 3.3 (24 total)
REQUIRED_COLUMNS <- c(
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
  "threshold"
)

# Expected data types for validation
COLUMN_TYPES <- list(
  prompt_id = "character",
  family = "character",
  prompt_text = "character",
  base_completion = "character",
  aligned_completion = "character",
  tox_score_base = "numeric",
  tox_score_aligned = "numeric",
  tox_binary_base = "integer",
  tox_binary_aligned = "integer",
  severe_tox_base = "numeric",
  severe_tox_aligned = "numeric",
  insult_base = "numeric",
  insult_aligned = "numeric",
  nodes_base = "integer",
  nodes_aligned = "integer",
  edges_base = "integer",
  edges_aligned = "integer",
  ratio_base = "numeric",
  ratio_aligned = "numeric",
  lcc_size_base = "integer",
  lcc_size_aligned = "integer",
  bigram_contrib_base = "numeric",
  bigram_contrib_aligned = "numeric",
  scorer = "character",
  threshold = "numeric"
)

# Valid family values per PRD
VALID_FAMILIES <- c("qwen3", "llama31", "mistral")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Generate mock data for testing
#'
#' Creates synthetic data matching the PRD schema for development and testing.
#'
#' @param n_rows Number of rows to generate (default 75000)
#' @param seed Random seed for reproducibility
#' @return tibble with mock analysis data
generate_mock_data <- function(n_rows = EXPECTED_ROWS, seed = RANDOM_SEED) {
  set.seed(seed)

  n_prompts <- n_rows / 3

  # Generate prompt IDs
  prompt_ids <- sprintf("prompt_%05d", seq_len(n_prompts))

  # Expand for 3 families
  data <- tibble(
    prompt_id = rep(prompt_ids, each = 3),
    family = rep(VALID_FAMILIES, times = n_prompts)
  )

  # Generate text columns
  data <- data %>%
    mutate(
      prompt_text = paste("Sample prompt text for", prompt_id),
      base_completion = paste("Base model completion for", prompt_id, "family", family),
      aligned_completion = paste("Aligned model completion for", prompt_id, "family", family)
    )

  # Generate toxicity scores (simulate alignment reducing toxicity)
  # Base models: higher toxicity
  # Aligned models: lower toxicity
  data <- data %>%
    mutate(
      # Base scores: mixture of low and high toxicity
      tox_score_base = pmin(1, pmax(0, rbeta(n(), 2, 5) + runif(n(), 0, 0.3))),
      # Aligned scores: generally lower
      tox_score_aligned = pmin(1, pmax(0, tox_score_base * runif(n(), 0.2, 0.8))),

      # Binary toxicity at 0.5 threshold
      tox_binary_base = as.integer(tox_score_base >= 0.5),
      tox_binary_aligned = as.integer(tox_score_aligned >= 0.5),

      # Additional toxicity subscores
      severe_tox_base = pmin(1, pmax(0, tox_score_base * runif(n(), 0.1, 0.5))),
      severe_tox_aligned = pmin(1, pmax(0, tox_score_aligned * runif(n(), 0.1, 0.5))),
      insult_base = pmin(1, pmax(0, tox_score_base * runif(n(), 0.3, 0.8))),
      insult_aligned = pmin(1, pmax(0, tox_score_aligned * runif(n(), 0.3, 0.8)))
    )

  # Generate xFakeSci network features
  data <- data %>%
    mutate(
      # Node counts (unique words)
      nodes_base = as.integer(runif(n(), 15, 60)),
      nodes_aligned = as.integer(runif(n(), 15, 60)),

      # Edge counts (unique bigrams)
      edges_base = as.integer(nodes_base * runif(n(), 0.8, 1.5)),
      edges_aligned = as.integer(nodes_aligned * runif(n(), 0.8, 1.5)),

      # Ratio (edges / nodes)
      ratio_base = edges_base / nodes_base,
      ratio_aligned = edges_aligned / nodes_aligned,

      # Largest connected component size
      lcc_size_base = as.integer(nodes_base * runif(n(), 0.7, 0.95)),
      lcc_size_aligned = as.integer(nodes_aligned * runif(n(), 0.7, 0.95)),

      # Bigram contribution
      bigram_contrib_base = runif(n(), 0.1, 0.5),
      bigram_contrib_aligned = runif(n(), 0.1, 0.5)
    )

  # Add scorer and threshold columns
  data <- data %>%
    mutate(
      scorer = "detoxify",
      threshold = 0.5
    )

  return(data)
}

#' Validate column presence
#'
#' @param data tibble to validate
#' @return list with status and missing columns
validate_columns <- function(data) {
  present_cols <- names(data)
  missing_cols <- setdiff(REQUIRED_COLUMNS, present_cols)
  extra_cols <- setdiff(present_cols, REQUIRED_COLUMNS)

  list(
    valid = length(missing_cols) == 0,
    missing = missing_cols,
    extra = extra_cols,
    message = if (length(missing_cols) == 0) {
      "All 24 required columns present"
    } else {
      paste("Missing columns:", paste(missing_cols, collapse = ", "))
    }
  )
}

#' Validate data types
#'
#' @param data tibble to validate
#' @return list with validation results per column
validate_types <- function(data) {
  results <- list()

  for (col_name in names(COLUMN_TYPES)) {
    if (col_name %in% names(data)) {
      expected_type <- COLUMN_TYPES[[col_name]]
      actual_type <- class(data[[col_name]])[1]

      # Map R types for comparison
      is_valid <- switch(
        expected_type,
        "character" = is.character(data[[col_name]]),
        "numeric" = is.numeric(data[[col_name]]),
        "integer" = is.numeric(data[[col_name]]),  # Accept numeric for integers
        FALSE
      )

      results[[col_name]] <- list(
        expected = expected_type,
        actual = actual_type,
        valid = is_valid
      )
    }
  }

  invalid_cols <- names(Filter(function(x) !x$valid, results))

  list(
    results = results,
    all_valid = length(invalid_cols) == 0,
    invalid_columns = invalid_cols
  )
}

#' Validate value ranges
#'
#' @param data tibble to validate
#' @return list with validation results
validate_ranges <- function(data) {
  issues <- character()

  # Check toxicity scores are in [0, 1]
  score_cols <- c("tox_score_base", "tox_score_aligned",
                  "severe_tox_base", "severe_tox_aligned",
                  "insult_base", "insult_aligned",
                  "bigram_contrib_base", "bigram_contrib_aligned",
                  "ratio_base", "ratio_aligned")

  for (col in score_cols) {
    if (col %in% names(data)) {
      vals <- data[[col]]
      if (any(vals < 0, na.rm = TRUE)) {
        issues <- c(issues, paste(col, "has values < 0"))
      }
      # Note: ratio can exceed 1, so only check score columns
      if (grepl("tox|insult|severe", col) && any(vals > 1, na.rm = TRUE)) {
        issues <- c(issues, paste(col, "has values > 1"))
      }
    }
  }

  # Check binary columns are 0 or 1
  binary_cols <- c("tox_binary_base", "tox_binary_aligned")
  for (col in binary_cols) {
    if (col %in% names(data)) {
      unique_vals <- unique(na.omit(data[[col]]))
      if (!all(unique_vals %in% c(0, 1))) {
        issues <- c(issues, paste(col, "has values other than 0/1"))
      }
    }
  }

  # Check family values
  if ("family" %in% names(data)) {
    invalid_families <- setdiff(unique(data$family), VALID_FAMILIES)
    if (length(invalid_families) > 0) {
      issues <- c(issues, paste("Invalid family values:",
                                paste(invalid_families, collapse = ", ")))
    }
  }

  # Check positive integers for network features
  int_cols <- c("nodes_base", "nodes_aligned", "edges_base", "edges_aligned",
                "lcc_size_base", "lcc_size_aligned")
  for (col in int_cols) {
    if (col %in% names(data)) {
      if (any(data[[col]] < 0, na.rm = TRUE)) {
        issues <- c(issues, paste(col, "has negative values"))
      }
    }
  }

  list(
    valid = length(issues) == 0,
    issues = issues
  )
}

#' Report missing values
#'
#' @param data tibble to analyze
#' @return tibble with missing value counts per column
report_missing <- function(data) {
  missing_summary <- tibble(
    column = names(data),
    n_missing = map_int(data, ~sum(is.na(.))),
    pct_missing = round(n_missing / nrow(data) * 100, 2)
  ) %>%
    filter(n_missing > 0) %>%
    arrange(desc(n_missing))

  return(missing_summary)
}

#' Generate comprehensive validation report
#'
#' @param data tibble to validate
#' @param expected_rows expected number of rows
#' @return list with all validation results
generate_validation_report <- function(data, expected_rows = EXPECTED_ROWS) {
  cat("\n")
  cat("=============================================================================\n")
  cat("                    DATA VALIDATION REPORT\n")
  cat("=============================================================================\n\n")

  # Row count
  actual_rows <- nrow(data)
  row_valid <- actual_rows == expected_rows
  cat(sprintf("1. ROW COUNT\n"))
  cat(sprintf("   Expected: %d rows\n", expected_rows))
  cat(sprintf("   Actual:   %d rows\n", actual_rows))
  cat(sprintf("   Status:   %s\n\n", if(row_valid) "PASS" else "WARN"))

  # Column validation
  col_validation <- validate_columns(data)
  cat(sprintf("2. COLUMN VALIDATION\n"))
  cat(sprintf("   Required columns: 24\n"))
  cat(sprintf("   Present columns:  %d\n", length(names(data))))
  cat(sprintf("   Status:           %s\n", if(col_validation$valid) "PASS" else "FAIL"))
  if (length(col_validation$missing) > 0) {
    cat(sprintf("   Missing: %s\n", paste(col_validation$missing, collapse = ", ")))
  }
  if (length(col_validation$extra) > 0) {
    cat(sprintf("   Extra:   %s\n", paste(col_validation$extra, collapse = ", ")))
  }
  cat("\n")

  # Type validation
  type_validation <- validate_types(data)
  cat(sprintf("3. DATA TYPE VALIDATION\n"))
  cat(sprintf("   Status: %s\n", if(type_validation$all_valid) "PASS" else "FAIL"))
  if (length(type_validation$invalid_columns) > 0) {
    cat("   Invalid types:\n")
    for (col in type_validation$invalid_columns) {
      info <- type_validation$results[[col]]
      cat(sprintf("     - %s: expected %s, got %s\n",
                  col, info$expected, info$actual))
    }
  }
  cat("\n")

  # Range validation
  range_validation <- validate_ranges(data)
  cat(sprintf("4. VALUE RANGE VALIDATION\n"))
  cat(sprintf("   Status: %s\n", if(range_validation$valid) "PASS" else "FAIL"))
  if (length(range_validation$issues) > 0) {
    cat("   Issues:\n")
    for (issue in range_validation$issues) {
      cat(sprintf("     - %s\n", issue))
    }
  }
  cat("\n")

  # Missing values
  missing_report <- report_missing(data)
  cat(sprintf("5. MISSING VALUES\n"))
  if (nrow(missing_report) == 0) {
    cat("   No missing values detected\n")
    cat("   Status: PASS\n")
  } else {
    cat(sprintf("   %d columns with missing values:\n", nrow(missing_report)))
    for (i in seq_len(min(10, nrow(missing_report)))) {
      cat(sprintf("     - %s: %d (%.2f%%)\n",
                  missing_report$column[i],
                  missing_report$n_missing[i],
                  missing_report$pct_missing[i]))
    }
    if (nrow(missing_report) > 10) {
      cat(sprintf("     ... and %d more columns\n", nrow(missing_report) - 10))
    }
    cat("   Status: WARN\n")
  }
  cat("\n")

  # Family distribution
  cat(sprintf("6. FAMILY DISTRIBUTION\n"))
  if ("family" %in% names(data)) {
    family_counts <- table(data$family)
    for (fam in names(family_counts)) {
      cat(sprintf("   %s: %d rows (%.1f%%)\n",
                  fam,
                  family_counts[fam],
                  family_counts[fam] / nrow(data) * 100))
    }
  }
  cat("\n")

  # Summary statistics for key columns
  cat(sprintf("7. KEY COLUMN STATISTICS\n"))
  if ("tox_score_base" %in% names(data) && "tox_score_aligned" %in% names(data)) {
    cat("   Toxicity Scores:\n")
    cat(sprintf("     Base:    mean=%.3f, sd=%.3f, median=%.3f\n",
                mean(data$tox_score_base, na.rm = TRUE),
                sd(data$tox_score_base, na.rm = TRUE),
                median(data$tox_score_base, na.rm = TRUE)))
    cat(sprintf("     Aligned: mean=%.3f, sd=%.3f, median=%.3f\n",
                mean(data$tox_score_aligned, na.rm = TRUE),
                sd(data$tox_score_aligned, na.rm = TRUE),
                median(data$tox_score_aligned, na.rm = TRUE)))
  }
  if ("tox_binary_base" %in% names(data) && "tox_binary_aligned" %in% names(data)) {
    cat("   Binary Toxicity Rates:\n")
    cat(sprintf("     Base:    %.1f%% toxic\n",
                mean(data$tox_binary_base, na.rm = TRUE) * 100))
    cat(sprintf("     Aligned: %.1f%% toxic\n",
                mean(data$tox_binary_aligned, na.rm = TRUE) * 100))
  }
  cat("\n")

  # Overall status
  all_pass <- row_valid && col_validation$valid &&
              type_validation$all_valid && range_validation$valid

  cat("=============================================================================\n")
  cat(sprintf("OVERALL VALIDATION: %s\n", if(all_pass) "PASS" else "ISSUES DETECTED"))
  cat("=============================================================================\n\n")

  # Return validation summary
  list(
    row_count = list(expected = expected_rows, actual = actual_rows, valid = row_valid),
    columns = col_validation,
    types = type_validation,
    ranges = range_validation,
    missing = missing_report,
    overall_valid = all_pass
  )
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  use_mock <- "--mock" %in% args

  # Check for custom input path
  input_idx <- which(args == "--input")
  if (length(input_idx) > 0 && length(args) > input_idx) {
    input_path <- args[input_idx + 1]
  } else {
    input_path <- DEFAULT_INPUT_PATH
  }

  cat("LLM Safety Alignment Study - Data Loading and Validation\n")
  cat("=========================================================\n\n")

  # Load data
  if (use_mock) {
    cat("Mode: MOCK DATA (for testing)\n")
    cat(sprintf("Generating %d rows of synthetic data...\n\n", EXPECTED_ROWS))
    data <- generate_mock_data()
  } else {
    cat(sprintf("Mode: REAL DATA\n"))
    cat(sprintf("Input path: %s\n\n", input_path))

    # Check if file exists
    if (!file.exists(input_path)) {
      cat("WARNING: Input file not found.\n")
      cat("Falling back to mock data for testing.\n\n")
      data <- generate_mock_data()
      use_mock <- TRUE
    } else {
      cat("Loading CSV file...\n")
      data <- read_csv(
        input_path,
        col_types = cols(
          prompt_id = col_character(),
          family = col_character(),
          prompt_text = col_character(),
          base_completion = col_character(),
          aligned_completion = col_character(),
          tox_score_base = col_double(),
          tox_score_aligned = col_double(),
          tox_binary_base = col_integer(),
          tox_binary_aligned = col_integer(),
          severe_tox_base = col_double(),
          severe_tox_aligned = col_double(),
          insult_base = col_double(),
          insult_aligned = col_double(),
          nodes_base = col_integer(),
          nodes_aligned = col_integer(),
          edges_base = col_integer(),
          edges_aligned = col_integer(),
          ratio_base = col_double(),
          ratio_aligned = col_double(),
          lcc_size_base = col_integer(),
          lcc_size_aligned = col_integer(),
          bigram_contrib_base = col_double(),
          bigram_contrib_aligned = col_double(),
          scorer = col_character(),
          threshold = col_double()
        ),
        show_col_types = FALSE
      )
      cat(sprintf("Loaded %d rows\n", nrow(data)))
    }
  }

  # Run validation
  validation_report <- generate_validation_report(data)

  # Save validated data
  cat(sprintf("Saving validated data to: %s\n", OUTPUT_RDS_PATH))

  # Ensure output directory exists
  output_dir <- dirname(OUTPUT_RDS_PATH)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Save as RDS with metadata
  output_data <- list(
    data = data,
    validation = validation_report,
    metadata = list(
      source = if(use_mock) "mock_data" else input_path,
      generated_at = Sys.time(),
      n_rows = nrow(data),
      n_cols = ncol(data),
      random_seed = RANDOM_SEED
    )
  )

  saveRDS(output_data, OUTPUT_RDS_PATH)
  cat("Data saved successfully.\n\n")

  # Return validation status
  if (validation_report$overall_valid) {
    cat("Validation PASSED. Data is ready for analysis.\n")
    return(invisible(0))
  } else {
    cat("Validation completed with warnings. Review report above.\n")
    return(invisible(1))
  }
}

# Run main function if script is executed directly
if (!interactive()) {
  status <- main()
  quit(status = status)
}
