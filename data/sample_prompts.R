#!/usr/bin/env Rscript
# ==============================================================================
# Stratified Prompt Sampling Script
# Task A3: data/sample_prompts.R
#
# Creates a stratified sample of 25,000 prompts:
#   - 12,500 from RealToxicityPrompts (stratified by toxicity tertiles)
#   - 12,500 from ToxiGen (stratified by target_group categories)
#
# Outputs:
#   - data/processed/prompt_sample_25k.csv
#   - data/processed/prompt_sample_25k.json
#   - data/processed/sample_validation_report.txt
#
# Usage:
#   Rscript sample_prompts.R          # Normal mode (requires source files)
#   Rscript sample_prompts.R --mock   # Mock mode (generates synthetic data)
#
# Author: LLM Safety Alignment Study
# Random Seed: 20260118 (for reproducibility per PRD)
# ==============================================================================

# ==============================================================================
# Configuration
# ==============================================================================

RANDOM_SEED <- 20260118
RTP_SAMPLE_SIZE <- 12500
TOXIGEN_SAMPLE_SIZE <- 12500
TOTAL_SAMPLE_SIZE <- RTP_SAMPLE_SIZE + TOXIGEN_SAMPLE_SIZE

# File paths (absolute paths for security)
# Determine script directory robustly for both interactive and batch execution
get_script_dir <- function() {
  # Try multiple methods to find the script directory

  # Method 1: commandArgs (when run via Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(normalizePath(dirname(script_path), mustWork = FALSE))
  }

  # Method 2: sys.frame for sourced scripts
  tryCatch({
    frame <- sys.frame(1)
    if (!is.null(frame$ofile)) {
      return(normalizePath(dirname(frame$ofile), mustWork = FALSE))
    }
  }, error = function(e) NULL)

  # Method 3: Fall back to working directory
  return(normalizePath(getwd(), mustWork = TRUE))
}

SCRIPT_DIR <- get_script_dir()
if (is.na(SCRIPT_DIR) || SCRIPT_DIR == "") {
  SCRIPT_DIR <- normalizePath(getwd(), mustWork = TRUE)
}

DATA_DIR <- file.path(SCRIPT_DIR)
RAW_DIR <- file.path(DATA_DIR, "raw")
PROCESSED_DIR <- file.path(DATA_DIR, "processed")

RTP_INPUT_FILE <- file.path(RAW_DIR, "realtoxicityprompts.jsonl")
TOXIGEN_INPUT_FILE <- file.path(RAW_DIR, "toxigen_train.csv")

OUTPUT_CSV <- file.path(PROCESSED_DIR, "prompt_sample_25k.csv")
OUTPUT_JSON <- file.path(PROCESSED_DIR, "prompt_sample_25k.json")
VALIDATION_REPORT <- file.path(PROCESSED_DIR, "sample_validation_report.txt")

# Expected source sizes (for validation)
RTP_EXPECTED_ROWS <- 99442
TOXIGEN_EXPECTED_ROWS <- 274186

# ==============================================================================
# Load Required Libraries
# ==============================================================================

suppressPackageStartupMessages({
  required_packages <- c("tidyverse", "jsonlite")

  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("Required package '%s' is not installed. Install with: install.packages('%s')", pkg, pkg))
    }
    library(pkg, character.only = TRUE)
  }
})

# ==============================================================================
# Utility Functions
# ==============================================================================

#' Log a message with timestamp
#' @param msg Message to log
#' @param level Log level (INFO, WARN, ERROR)
log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, msg))
}

#' Validate file path is within allowed directory (prevent path traversal)
#' @param file_path Path to validate
#' @param allowed_base Allowed base directory
#' @return TRUE if valid, stops with error if not
validate_path <- function(file_path, allowed_base) {
  # Normalize paths to resolve any ../ components
  norm_path <- normalizePath(file_path, mustWork = FALSE)
  norm_base <- normalizePath(allowed_base, mustWork = TRUE)

  # Check if the normalized path starts with the allowed base

  if (!startsWith(norm_path, norm_base)) {
    stop(sprintf("Path traversal attempt detected: %s is outside %s", file_path, allowed_base))
  }

  return(TRUE)
}

#' Create toxicity tertiles from continuous scores
#' @param toxicity_scores Numeric vector of toxicity scores
#' @return Factor with levels: low, medium, high
create_toxicity_tertiles <- function(toxicity_scores) {
  # Handle NA values
  scores <- toxicity_scores[!is.na(toxicity_scores)]

  if (length(scores) == 0) {
    stop("No valid toxicity scores to create tertiles")
  }

  # Calculate tertile boundaries
  breaks <- quantile(scores, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)

  # Ensure unique breaks (in case of many ties)
  if (length(unique(breaks)) < 4) {
    log_message("Warning: Toxicity scores have limited variance, using adjusted breaks", "WARN")
    breaks <- c(min(scores) - 0.001,
                mean(c(min(scores), median(scores))),
                mean(c(median(scores), max(scores))),
                max(scores) + 0.001)
  }

  # Create tertile factor
  tertiles <- cut(toxicity_scores,
                  breaks = breaks,
                  labels = c("low", "medium", "high"),
                  include.lowest = TRUE)

  return(tertiles)
}

#' Generate mock RTP data for testing
#' @param n Number of rows to generate
#' @return Tibble with mock RTP data
generate_mock_rtp <- function(n = RTP_EXPECTED_ROWS) {
  log_message(sprintf("Generating %d mock RTP records...", n))

  set.seed(RANDOM_SEED)

  # Generate realistic toxicity distribution (beta distribution)
  # Most prompts have low toxicity, with a long tail
  toxicity_scores <- rbeta(n, shape1 = 0.5, shape2 = 2)

  # Generate mock prompts
  prompt_templates <- c(
    "The quick brown fox",
    "In today's news",
    "Scientists have discovered",
    "The weather forecast shows",
    "According to recent studies",
    "Local authorities reported",
    "The company announced",
    "Researchers at the university",
    "In a surprising development",
    "The report indicates that"
  )

  tibble(
    prompt_text = paste0(
      sample(prompt_templates, n, replace = TRUE),
      " [mock_", seq_len(n), "]"
    ),
    toxicity = toxicity_scores
  )
}

#' Generate mock ToxiGen data for testing
#' @param n Number of rows to generate
#' @return Tibble with mock ToxiGen data
generate_mock_toxigen <- function(n = TOXIGEN_EXPECTED_ROWS) {
  log_message(sprintf("Generating %d mock ToxiGen records...", n))

  set.seed(RANDOM_SEED + 1)  # Different seed for variety

  # ToxiGen target groups (based on actual dataset categories)
  target_groups <- c(
    "asian", "black", "chinese", "jewish", "latino",
    "lgbtq", "mental_disability", "mexican", "middle_eastern",
    "muslim", "native_american", "physical_disability", "women"
  )

  # Uneven distribution to match real data characteristics
  group_probs <- c(0.08, 0.12, 0.07, 0.09, 0.06,
                   0.10, 0.05, 0.06, 0.08,
                   0.11, 0.04, 0.05, 0.09)

  tibble(
    text = paste0(
      "Statement about [group_placeholder] [mock_", seq_len(n), "]"
    ),
    target_group = sample(target_groups, n, replace = TRUE, prob = group_probs)
  )
}

# ==============================================================================
# Data Loading Functions
# ==============================================================================

#' Load RealToxicityPrompts from JSONL file
#' @param file_path Path to the JSONL file
#' @return Tibble with prompt_text and toxicity columns
load_rtp_data <- function(file_path) {
  log_message(sprintf("Loading RealToxicityPrompts from: %s", file_path))

  # Validate path
  validate_path(file_path, DATA_DIR)

  if (!file.exists(file_path)) {
    stop(sprintf("RTP file not found: %s", file_path))
  }

  # Read JSONL line by line for memory efficiency
  lines <- readLines(file_path, warn = FALSE)
  log_message(sprintf("Read %d lines from JSONL", length(lines)))

  # Parse each line
  parsed_data <- lapply(seq_along(lines), function(i) {
    line <- lines[i]
    if (nchar(trimws(line)) == 0) return(NULL)

    tryCatch({
      json_obj <- fromJSON(line, simplifyVector = TRUE)

      # Extract prompt.text and prompt.toxicity
      prompt <- json_obj$prompt
      if (is.null(prompt)) return(NULL)

      text <- if (!is.null(prompt$text)) prompt$text else NA_character_
      toxicity <- if (!is.null(prompt$toxicity)) as.numeric(prompt$toxicity) else NA_real_

      list(prompt_text = text, toxicity = toxicity)
    }, error = function(e) {
      if (i <= 5) {
        log_message(sprintf("Parse error on line %d: %s", i, e$message), "WARN")
      }
      NULL
    })
  })

  # Filter NULL entries and convert to tibble
  valid_data <- Filter(Negate(is.null), parsed_data)
  log_message(sprintf("Successfully parsed %d records", length(valid_data)))

  result <- tibble(
    prompt_text = sapply(valid_data, function(x) x$prompt_text),
    toxicity = sapply(valid_data, function(x) x$toxicity)
  )

  # Remove rows with missing text
  result <- result %>%
    filter(!is.na(prompt_text), nchar(trimws(prompt_text)) > 0)

  log_message(sprintf("RTP data loaded: %d rows", nrow(result)))

  return(result)
}

#' Load ToxiGen data from CSV file
#' @param file_path Path to the CSV file
#' @return Tibble with text and target_group columns
load_toxigen_data <- function(file_path) {
  log_message(sprintf("Loading ToxiGen from: %s", file_path))

  # Validate path
  validate_path(file_path, DATA_DIR)

  if (!file.exists(file_path)) {
    stop(sprintf("ToxiGen file not found: %s", file_path))
  }

  # Read CSV with explicit column types for safety
  result <- read_csv(
    file_path,
    col_types = cols(
      text = col_character(),
      target_group = col_character(),
      .default = col_character()
    ),
    show_col_types = FALSE
  )

  # Ensure required columns exist
  if (!"text" %in% names(result)) {
    stop("ToxiGen CSV missing 'text' column")
  }
  if (!"target_group" %in% names(result)) {
    stop("ToxiGen CSV missing 'target_group' column")
  }

  # Select only needed columns and filter empty rows
  result <- result %>%
    select(text, target_group) %>%
    filter(!is.na(text), nchar(trimws(text)) > 0)

  log_message(sprintf("ToxiGen data loaded: %d rows", nrow(result)))

  return(result)
}

# ==============================================================================
# Stratified Sampling Functions
# ==============================================================================

#' Perform stratified sampling on RTP data by toxicity tertiles
#' @param data RTP data tibble
#' @param sample_size Target sample size
#' @return Sampled tibble with stratum column
sample_rtp_stratified <- function(data, sample_size = RTP_SAMPLE_SIZE) {
  log_message(sprintf("Stratified sampling %d prompts from RTP by toxicity tertiles...", sample_size))

  # Remove rows with NA toxicity (cannot stratify)
  data_with_tox <- data %>%
    filter(!is.na(toxicity))

  na_count <- nrow(data) - nrow(data_with_tox)
  if (na_count > 0) {
    log_message(sprintf("Excluded %d rows with NA toxicity", na_count), "WARN")
  }

  # Create toxicity tertiles
  data_with_tox <- data_with_tox %>%
    mutate(stratum = create_toxicity_tertiles(toxicity))

  # Calculate stratum sizes
  stratum_counts <- data_with_tox %>%
    count(stratum, name = "n_available")

  log_message("Stratum distribution in source data:")
  print(stratum_counts)

  # Calculate samples per stratum (equal allocation for tertiles)
  n_strata <- nrow(stratum_counts)
  samples_per_stratum <- floor(sample_size / n_strata)
  remainder <- sample_size - (samples_per_stratum * n_strata)

  # Allocate remainder to largest strata
  stratum_counts <- stratum_counts %>%
    arrange(desc(n_available)) %>%
    mutate(
      n_to_sample = as.integer(samples_per_stratum + if_else(row_number() <= remainder, 1L, 0L))
    )

  # Verify we can sample the requested amounts
  insufficient_strata <- stratum_counts %>%
    filter(n_to_sample > n_available)

  if (nrow(insufficient_strata) > 0) {
    log_message("Adjusting sample sizes for strata with insufficient data", "WARN")
    # Redistribute from strata with excess
    stratum_counts <- stratum_counts %>%
      mutate(
        actual_sample = pmin(n_to_sample, n_available),
        shortfall = n_to_sample - actual_sample
      )
    total_shortfall <- sum(stratum_counts$shortfall)

    # Add shortfall to strata with capacity
    stratum_counts <- stratum_counts %>%
      mutate(
        extra_capacity = n_available - actual_sample,
        extra_to_sample = if_else(extra_capacity > 0,
                                   as.integer(floor(total_shortfall * extra_capacity / sum(extra_capacity))),
                                   0L),
        n_to_sample = as.integer(actual_sample + extra_to_sample)
      ) %>%
      select(stratum, n_available, n_to_sample)
  }

  log_message("Final sampling allocation:")
  print(stratum_counts)

  # Perform stratified sampling by iterating over each stratum
  # This avoids the issue with slice_sample inside grouped data
  sampled_list <- lapply(seq_len(nrow(stratum_counts)), function(i) {
    stratum_name <- stratum_counts$stratum[i]
    n_sample <- stratum_counts$n_to_sample[i]

    data_with_tox %>%
      filter(stratum == stratum_name) %>%
      slice_sample(n = n_sample)
  })

  sampled <- bind_rows(sampled_list)

  # Verify sample size
  if (nrow(sampled) < sample_size) {
    shortfall <- sample_size - nrow(sampled)
    log_message(sprintf("Sample size shortfall: %d. Adding more from available data.", shortfall), "WARN")

    remaining <- data_with_tox %>%
      anti_join(sampled, by = "prompt_text") %>%
      slice_sample(n = min(shortfall, nrow(.)))

    sampled <- bind_rows(sampled, remaining)
  }

  log_message(sprintf("RTP sampling complete: %d prompts", nrow(sampled)))

  return(sampled)
}

#' Perform stratified sampling on ToxiGen data by target group
#' @param data ToxiGen data tibble
#' @param sample_size Target sample size
#' @return Sampled tibble with stratum column
sample_toxigen_stratified <- function(data, sample_size = TOXIGEN_SAMPLE_SIZE) {
  log_message(sprintf("Stratified sampling %d prompts from ToxiGen by target group...", sample_size))

  # Get stratum (target group) counts
  stratum_counts <- data %>%
    count(target_group, name = "n_available") %>%
    arrange(desc(n_available))

  log_message(sprintf("Found %d unique target groups", nrow(stratum_counts)))

  # Calculate proportional allocation
  total_available <- sum(stratum_counts$n_available)
  stratum_counts <- stratum_counts %>%
    mutate(
      proportion = n_available / total_available,
      n_to_sample = as.integer(round(proportion * sample_size))
    )

  # Adjust for rounding errors
  diff <- sample_size - sum(stratum_counts$n_to_sample)
  if (diff != 0) {
    # Add/subtract from largest groups
    stratum_counts <- stratum_counts %>%
      mutate(n_to_sample = as.integer(n_to_sample + if_else(row_number() <= abs(diff), as.integer(sign(diff)), 0L)))
  }

  # Ensure we don't sample more than available
  stratum_counts <- stratum_counts %>%
    mutate(n_to_sample = as.integer(pmin(n_to_sample, n_available)))

  log_message("Sampling allocation by target group:")
  print(head(stratum_counts, 10))

  # Perform stratified sampling by iterating over each stratum
  # This avoids the issue with slice_sample inside grouped data
  sampled_list <- lapply(seq_len(nrow(stratum_counts)), function(i) {
    group_name <- stratum_counts$target_group[i]
    n_sample <- stratum_counts$n_to_sample[i]

    data %>%
      filter(target_group == group_name) %>%
      slice_sample(n = n_sample) %>%
      mutate(stratum = target_group)
  })

  sampled <- bind_rows(sampled_list)

  # Handle any shortfall
  if (nrow(sampled) < sample_size) {
    shortfall <- sample_size - nrow(sampled)
    log_message(sprintf("Sample size shortfall: %d. Adding more from available data.", shortfall), "WARN")

    remaining <- data %>%
      anti_join(sampled, by = "text") %>%
      slice_sample(n = min(shortfall, nrow(.))) %>%
      mutate(stratum = target_group)

    sampled <- bind_rows(sampled, remaining)
  }

  log_message(sprintf("ToxiGen sampling complete: %d prompts", nrow(sampled)))

  return(sampled)
}

# ==============================================================================
# Output Generation Functions
# ==============================================================================

#' Create standardized output dataset from sampled data
#' @param rtp_sample Sampled RTP data
#' @param toxigen_sample Sampled ToxiGen data
#' @return Combined tibble in output schema format
create_output_dataset <- function(rtp_sample, toxigen_sample) {
  log_message("Creating unified output dataset...")

  # Process RTP samples
  rtp_output <- rtp_sample %>%
    mutate(
      prompt_id = sprintf("rtp_%05d", row_number()),
      source = "rtp",
      text = prompt_text,
      toxicity_source = toxicity,
      target_group = NA_character_
      # stratum already exists from sampling
    ) %>%
    select(prompt_id, source, text, toxicity_source, target_group, stratum)

  # Process ToxiGen samples
  toxigen_output <- toxigen_sample %>%
    mutate(
      prompt_id = sprintf("tg_%05d", row_number()),
      source = "toxigen",
      # text already exists
      toxicity_source = NA_real_
      # target_group and stratum already exist
    ) %>%
    select(prompt_id, source, text, toxicity_source, target_group, stratum)

  # Combine datasets
  combined <- bind_rows(rtp_output, toxigen_output)

  log_message(sprintf("Combined dataset: %d rows", nrow(combined)))

  return(combined)
}

#' Validate the output dataset
#' @param data Output dataset tibble
#' @return List with validation results
validate_output <- function(data) {
  log_message("Validating output dataset...")

  results <- list(
    total_rows = nrow(data),
    rtp_rows = sum(data$source == "rtp"),
    toxigen_rows = sum(data$source == "toxigen"),
    unique_prompt_ids = n_distinct(data$prompt_id),
    unique_texts = n_distinct(data$text),
    duplicate_prompt_ids = nrow(data) - n_distinct(data$prompt_id),
    duplicate_texts = nrow(data) - n_distinct(data$text),
    na_prompt_ids = sum(is.na(data$prompt_id)),
    na_texts = sum(is.na(data$text)),
    empty_texts = sum(nchar(trimws(data$text)) == 0, na.rm = TRUE),
    rtp_strata = if (any(data$source == "rtp")) {
      data %>% filter(source == "rtp") %>% count(stratum) %>% deframe()
    } else NULL,
    toxigen_strata = if (any(data$source == "toxigen")) {
      data %>% filter(source == "toxigen") %>% count(stratum) %>% deframe()
    } else NULL
  )

  # Check for validation failures
  results$is_valid <- (
    results$total_rows == TOTAL_SAMPLE_SIZE &&
    results$rtp_rows == RTP_SAMPLE_SIZE &&
    results$toxigen_rows == TOXIGEN_SAMPLE_SIZE &&
    results$duplicate_prompt_ids == 0 &&
    results$duplicate_texts == 0 &&
    results$na_prompt_ids == 0 &&
    results$na_texts == 0 &&
    results$empty_texts == 0
  )

  return(results)
}

#' Generate validation report
#' @param validation_results List from validate_output()
#' @param output_path Path to write report
generate_validation_report <- function(validation_results, output_path) {
  log_message(sprintf("Generating validation report: %s", output_path))

  # Validate output path
  validate_path(output_path, PROCESSED_DIR)

  # Build report content
  report_lines <- c(
    "================================================================================",
    "PROMPT SAMPLE VALIDATION REPORT",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    sprintf("Random Seed: %d", RANDOM_SEED),
    "================================================================================",
    "",
    "SUMMARY",
    "-------",
    sprintf("Total rows:              %d (expected: %d) %s",
            validation_results$total_rows,
            TOTAL_SAMPLE_SIZE,
            if (validation_results$total_rows == TOTAL_SAMPLE_SIZE) "[PASS]" else "[FAIL]"),
    sprintf("RTP rows:                %d (expected: %d) %s",
            validation_results$rtp_rows,
            RTP_SAMPLE_SIZE,
            if (validation_results$rtp_rows == RTP_SAMPLE_SIZE) "[PASS]" else "[FAIL]"),
    sprintf("ToxiGen rows:            %d (expected: %d) %s",
            validation_results$toxigen_rows,
            TOXIGEN_SAMPLE_SIZE,
            if (validation_results$toxigen_rows == TOXIGEN_SAMPLE_SIZE) "[PASS]" else "[FAIL]"),
    "",
    "DATA QUALITY",
    "------------",
    sprintf("Unique prompt_ids:       %d %s",
            validation_results$unique_prompt_ids,
            if (validation_results$duplicate_prompt_ids == 0) "[PASS]" else "[FAIL]"),
    sprintf("Duplicate prompt_ids:    %d %s",
            validation_results$duplicate_prompt_ids,
            if (validation_results$duplicate_prompt_ids == 0) "[PASS]" else "[FAIL]"),
    sprintf("Unique texts:            %d %s",
            validation_results$unique_texts,
            if (validation_results$duplicate_texts == 0) "[PASS]" else "[FAIL]"),
    sprintf("Duplicate texts:         %d %s",
            validation_results$duplicate_texts,
            if (validation_results$duplicate_texts == 0) "[PASS]" else "[FAIL]"),
    sprintf("NA prompt_ids:           %d %s",
            validation_results$na_prompt_ids,
            if (validation_results$na_prompt_ids == 0) "[PASS]" else "[FAIL]"),
    sprintf("NA texts:                %d %s",
            validation_results$na_texts,
            if (validation_results$na_texts == 0) "[PASS]" else "[FAIL]"),
    sprintf("Empty texts:             %d %s",
            validation_results$empty_texts,
            if (validation_results$empty_texts == 0) "[PASS]" else "[FAIL]"),
    "",
    "STRATIFICATION - RTP (Toxicity Tertiles)",
    "-----------------------------------------"
  )

  if (!is.null(validation_results$rtp_strata)) {
    for (stratum_name in names(validation_results$rtp_strata)) {
      report_lines <- c(report_lines,
        sprintf("  %-20s %d", stratum_name, validation_results$rtp_strata[stratum_name]))
    }
  }

  report_lines <- c(report_lines,
    "",
    "STRATIFICATION - ToxiGen (Target Groups)",
    "-----------------------------------------"
  )

  if (!is.null(validation_results$toxigen_strata)) {
    sorted_strata <- sort(validation_results$toxigen_strata, decreasing = TRUE)
    for (stratum_name in names(sorted_strata)) {
      report_lines <- c(report_lines,
        sprintf("  %-20s %d", stratum_name, sorted_strata[stratum_name]))
    }
  }

  report_lines <- c(report_lines,
    "",
    "================================================================================",
    sprintf("OVERALL STATUS: %s", if (validation_results$is_valid) "PASSED" else "FAILED"),
    "================================================================================"
  )

  # Write report
  writeLines(report_lines, output_path)

  log_message(sprintf("Validation report written to: %s", output_path))

  return(invisible(validation_results$is_valid))
}

#' Save output dataset to CSV and JSON
#' @param data Output dataset tibble
#' @param csv_path Path for CSV output
#' @param json_path Path for JSON output
save_outputs <- function(data, csv_path, json_path) {
  # Validate paths
  validate_path(csv_path, PROCESSED_DIR)
  validate_path(json_path, PROCESSED_DIR)

  # Save CSV
  log_message(sprintf("Saving CSV: %s", csv_path))
  write_csv(data, csv_path, na = "")

  # Save JSON (array of objects format)
  log_message(sprintf("Saving JSON: %s", json_path))
  json_output <- toJSON(data, dataframe = "rows", na = "null", pretty = TRUE)
  write(json_output, json_path)

  log_message("Output files saved successfully")
}

# ==============================================================================
# Main Execution
# ==============================================================================

main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  # Handle --help flag
  if ("--help" %in% args || "-h" %in% args) {
    cat("
Stratified Prompt Sampling Script
==================================

Creates a stratified sample of 25,000 prompts for LLM safety alignment study.

Usage:
  Rscript sample_prompts.R [OPTIONS]

Options:
  --mock    Generate mock data for testing (no source files required)
  --help    Show this help message and exit
  -h        Same as --help

Inputs (required unless --mock):
  data/raw/realtoxicityprompts.jsonl  (99,442 rows)
  data/raw/toxigen_train.csv          (274,186 rows)

Outputs:
  data/processed/prompt_sample_25k.csv
  data/processed/prompt_sample_25k.json
  data/processed/sample_validation_report.txt

Output Schema:
  prompt_id       : string (format: 'rtp_00001' or 'tg_00001')
  source          : string ('rtp' or 'toxigen')
  text            : string (prompt text)
  toxicity_source : float (original toxicity score, NA for ToxiGen)
  target_group    : string (NA for RTP, group name for ToxiGen)
  stratum         : string (toxicity tertile or target group name)

Random Seed: 20260118 (for reproducibility per PRD)

Examples:
  Rscript sample_prompts.R           # Sample from real data
  Rscript sample_prompts.R --mock    # Generate mock test data
")
    return(0)
  }

  log_message("=" %>% rep(60) %>% paste(collapse = ""))
  log_message("Stratified Prompt Sampling Script")
  log_message("=" %>% rep(60) %>% paste(collapse = ""))

  mock_mode <- "--mock" %in% args

  if (mock_mode) {
    log_message("Running in MOCK MODE - generating synthetic data for testing")
  }

  # Set random seed for reproducibility
  set.seed(RANDOM_SEED)
  log_message(sprintf("Random seed set: %d", RANDOM_SEED))

  # Create output directory if needed
  if (!dir.exists(PROCESSED_DIR)) {
    log_message(sprintf("Creating output directory: %s", PROCESSED_DIR))
    dir.create(PROCESSED_DIR, recursive = TRUE)
  }

  # Load or generate data
  if (mock_mode) {
    # Generate mock data
    rtp_data <- generate_mock_rtp()
    toxigen_data <- generate_mock_toxigen()
  } else {
    # Check if source files exist
    if (!file.exists(RTP_INPUT_FILE)) {
      stop(sprintf(
        "RTP source file not found: %s\nRun with --mock flag to generate test data, or download the dataset first.",
        RTP_INPUT_FILE
      ))
    }
    if (!file.exists(TOXIGEN_INPUT_FILE)) {
      stop(sprintf(
        "ToxiGen source file not found: %s\nRun with --mock flag to generate test data, or download the dataset first.",
        TOXIGEN_INPUT_FILE
      ))
    }

    # Load real data
    rtp_data <- load_rtp_data(RTP_INPUT_FILE)
    toxigen_data <- load_toxigen_data(TOXIGEN_INPUT_FILE)
  }

  # Validate source data sizes
  log_message(sprintf("RTP source size: %d rows", nrow(rtp_data)))
  log_message(sprintf("ToxiGen source size: %d rows", nrow(toxigen_data)))

  if (nrow(rtp_data) < RTP_SAMPLE_SIZE) {
    stop(sprintf("Insufficient RTP data: %d rows available, %d required",
                 nrow(rtp_data), RTP_SAMPLE_SIZE))
  }
  if (nrow(toxigen_data) < TOXIGEN_SAMPLE_SIZE) {
    stop(sprintf("Insufficient ToxiGen data: %d rows available, %d required",
                 nrow(toxigen_data), TOXIGEN_SAMPLE_SIZE))
  }

  # Perform stratified sampling
  log_message("-" %>% rep(60) %>% paste(collapse = ""))
  rtp_sample <- sample_rtp_stratified(rtp_data, RTP_SAMPLE_SIZE)

  log_message("-" %>% rep(60) %>% paste(collapse = ""))
  toxigen_sample <- sample_toxigen_stratified(toxigen_data, TOXIGEN_SAMPLE_SIZE)

  # Create unified output dataset
  log_message("-" %>% rep(60) %>% paste(collapse = ""))
  output_data <- create_output_dataset(rtp_sample, toxigen_sample)

  # Remove duplicate texts (across both sources)
  initial_count <- nrow(output_data)
  output_data <- output_data %>%
    distinct(text, .keep_all = TRUE)

  if (nrow(output_data) < initial_count) {
    removed <- initial_count - nrow(output_data)
    log_message(sprintf("Removed %d duplicate texts across sources", removed), "WARN")

    # Resample to fill the gap
    shortfall <- TOTAL_SAMPLE_SIZE - nrow(output_data)
    if (shortfall > 0) {
      log_message(sprintf("Resampling %d prompts to meet target size", shortfall))

      # Get texts already in output
      existing_texts <- output_data$text

      # Try to get more from RTP
      rtp_shortfall <- min(shortfall, ceiling(shortfall / 2))
      rtp_extra <- rtp_data %>%
        filter(!prompt_text %in% existing_texts) %>%
        slice_sample(n = rtp_shortfall) %>%
        mutate(
          prompt_id = sprintf("rtp_%05d", RTP_SAMPLE_SIZE + row_number()),
          source = "rtp",
          text = prompt_text,
          toxicity_source = toxicity,
          target_group = NA_character_,
          stratum = as.character(create_toxicity_tertiles(toxicity))
        ) %>%
        select(prompt_id, source, text, toxicity_source, target_group, stratum)

      # Get more from ToxiGen
      toxigen_shortfall <- shortfall - nrow(rtp_extra)
      if (toxigen_shortfall > 0) {
        toxigen_extra <- toxigen_data %>%
          filter(!text %in% c(existing_texts, rtp_extra$text)) %>%
          slice_sample(n = toxigen_shortfall) %>%
          mutate(
            prompt_id = sprintf("tg_%05d", TOXIGEN_SAMPLE_SIZE + row_number()),
            source = "toxigen",
            toxicity_source = NA_real_,
            stratum = target_group
          ) %>%
          select(prompt_id, source, text, toxicity_source, target_group, stratum)

        output_data <- bind_rows(output_data, rtp_extra, toxigen_extra)
      } else {
        output_data <- bind_rows(output_data, rtp_extra)
      }
    }
  }

  # Validate output
  log_message("-" %>% rep(60) %>% paste(collapse = ""))
  validation_results <- validate_output(output_data)

  # Generate validation report
  report_success <- generate_validation_report(validation_results, VALIDATION_REPORT)

  # Save outputs
  log_message("-" %>% rep(60) %>% paste(collapse = ""))
  save_outputs(output_data, OUTPUT_CSV, OUTPUT_JSON)

  # Final summary
  log_message("=" %>% rep(60) %>% paste(collapse = ""))
  if (validation_results$is_valid) {
    log_message("SUCCESS: Stratified sampling complete")
  } else {
    log_message("WARNING: Sampling complete with validation issues", "WARN")
  }
  log_message(sprintf("Output CSV: %s", OUTPUT_CSV))
  log_message(sprintf("Output JSON: %s", OUTPUT_JSON))
  log_message(sprintf("Validation Report: %s", VALIDATION_REPORT))
  log_message("=" %>% rep(60) %>% paste(collapse = ""))

  # Return exit code
  return(if (validation_results$is_valid) 0 else 1)
}

# Execute main function if script is run directly
if (!interactive()) {
  exit_code <- tryCatch(
    main(),
    error = function(e) {
      log_message(sprintf("FATAL ERROR: %s", e$message), "ERROR")
      return(1)
    }
  )
  quit(status = exit_code)
}
