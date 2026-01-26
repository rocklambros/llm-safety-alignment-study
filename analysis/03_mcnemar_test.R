#!/usr/bin/env Rscript
# =============================================================================
# 03_mcnemar_test.R
# McNemar's Test for Paired Categorical Data
# =============================================================================
#
# Purpose: Perform McNemar's exact test for paired binary toxicity outcomes
#          to evaluate whether safety alignment significantly reduces toxic
#          outputs in LLMs.
#
# Input:  analysis/data_validated.rds
# Output: output/tables/mcnemar_results.csv
#
# Statistical Method:
#   McNemar's test evaluates the null hypothesis that marginal probabilities
#   are equal in a paired 2x2 contingency table. For our study:
#   - H0: b = c (alignment has no effect on toxicity)
#   - H1: b > c (alignment reduces toxicity)
#
#   Contingency Table:
#   |                   | Aligned: Non-toxic | Aligned: Toxic |
#   |-------------------|-------------------|----------------|
#   | Base: Non-toxic   | a                 | c              |
#   | Base: Toxic       | b                 | d              |
#
#   Effect Size: ARR = (b - c) / n (Absolute Risk Reduction)
#
# Usage:  Rscript analysis/03_mcnemar_test.R [--mock]
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(exact2x2)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_RDS_PATH <- "analysis/data_validated.rds"
OUTPUT_DIR <- "output/tables"
OUTPUT_FILE <- "mcnemar_results.csv"
RANDOM_SEED <- 20260118
BOOTSTRAP_REPLICATES <- 2000
ALPHA <- 0.05

# Valid family values
VALID_FAMILIES <- c("qwen3", "llama31", "mistral")

# Family display labels
FAMILY_LABELS <- c(
  qwen3 = "Qwen 3 (8B)",
  llama31 = "Llama 3.1 (8B)",
  mistral = "Mistral (7B)"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Generate mock data for testing
#'
#' Creates synthetic data matching the expected schema for development testing.
#'
#' @param n_rows Number of rows to generate (default 75000)
#' @param seed Random seed for reproducibility
#' @return tibble with mock analysis data
generate_mock_data <- function(n_rows = 75000, seed = RANDOM_SEED) {
  set.seed(seed)

  n_prompts <- n_rows / 3
  prompt_ids <- sprintf("prompt_%05d", seq_len(n_prompts))

  data <- tibble(
    prompt_id = rep(prompt_ids, each = 3),
    family = rep(VALID_FAMILIES, times = n_prompts)
  )

  # Simulate realistic toxicity patterns
  # Base models: higher toxicity rate (~35%)
  # Aligned models: lower toxicity rate (~12%)
  data <- data %>%
    mutate(
      # Base scores: mixture distribution with higher toxicity
      tox_score_base = pmin(1, pmax(0, rbeta(n(), 2, 5) + runif(n(), 0, 0.25))),
      # Aligned scores: reduced toxicity
      tox_score_aligned = pmin(1, pmax(0, tox_score_base * runif(n(), 0.15, 0.7))),
      # Binary thresholding at 0.5
      tox_binary_base = as.integer(tox_score_base >= 0.5),
      tox_binary_aligned = as.integer(tox_score_aligned >= 0.5)
    )

  return(data)
}

#' Build 2x2 contingency table for McNemar's test
#'
#' @param data tibble with tox_binary_base and tox_binary_aligned columns
#' @return list with contingency table cells (a, b, c, d) and matrix
build_contingency_table <- function(data) {
  # Cell counts:
  # a: base non-toxic, aligned non-toxic (both good)
  # b: base toxic, aligned non-toxic (alignment success)
  # c: base non-toxic, aligned toxic (alignment failure)
  # d: base toxic, aligned toxic (both bad)
  a <- sum(data$tox_binary_base == 0 & data$tox_binary_aligned == 0, na.rm = TRUE)
  b <- sum(data$tox_binary_base == 1 & data$tox_binary_aligned == 0, na.rm = TRUE)
  c <- sum(data$tox_binary_base == 0 & data$tox_binary_aligned == 1, na.rm = TRUE)
  d <- sum(data$tox_binary_base == 1 & data$tox_binary_aligned == 1, na.rm = TRUE)

  # Create matrix in standard McNemar format
  # Rows = Base model output
  # Cols = Aligned model output
  mat <- matrix(
    c(a, b, c, d),
    nrow = 2,
    byrow = TRUE,
    dimnames = list(
      "Base" = c("Non-toxic", "Toxic"),
      "Aligned" = c("Non-toxic", "Toxic")
    )
  )

  list(
    a = a, b = b, c = c, d = d,
    n = a + b + c + d,
    matrix = mat
  )
}

#' Calculate Absolute Risk Reduction (ARR) with bootstrap CI
#'
#' @param data tibble with binary toxicity columns
#' @param n_boot number of bootstrap replicates
#' @param seed random seed
#' @return list with ARR, SE, and confidence interval
calculate_arr_bootstrap <- function(data, n_boot = BOOTSTRAP_REPLICATES, seed = RANDOM_SEED) {
  set.seed(seed)

  n <- nrow(data)

  # Point estimate
  ct <- build_contingency_table(data)
  arr <- (ct$b - ct$c) / ct$n

  # Bootstrap for CI
  boot_arrs <- numeric(n_boot)

  for (i in seq_len(n_boot)) {
    boot_idx <- sample(n, n, replace = TRUE)
    boot_data <- data[boot_idx, ]
    boot_ct <- build_contingency_table(boot_data)
    boot_arrs[i] <- (boot_ct$b - boot_ct$c) / boot_ct$n
  }

  # Calculate confidence interval (percentile method)
  ci <- quantile(boot_arrs, probs = c(ALPHA / 2, 1 - ALPHA / 2))

  list(
    arr = arr,
    se = sd(boot_arrs),
    ci_lower = ci[1],
    ci_upper = ci[2],
    boot_distribution = boot_arrs
  )
}

#' Perform McNemar's exact test for a single family
#'
#' @param data tibble filtered to single family
#' @param family family name
#' @return tibble with test results
perform_mcnemar_test <- function(data, family) {
  # Build contingency table
  ct <- build_contingency_table(data)

  # Perform exact McNemar's test using exact2x2 package
  # Uses conditional exact test based on binomial distribution
  mcnemar_result <- tryCatch({
    mcnemar.exact(ct$matrix)
  }, error = function(e) {
    warning(sprintf("McNemar test failed for %s: %s", family, e$message))
    NULL
  })

  # Calculate traditional chi-squared statistic
  chi_sq <- if (ct$b + ct$c > 0) {
    (ct$b - ct$c)^2 / (ct$b + ct$c)
  } else {
    NA_real_
  }

  # Calculate ARR with bootstrap CI
  arr_result <- calculate_arr_bootstrap(data)

  # Extract p-value and odds ratio
  if (!is.null(mcnemar_result)) {
    p_value <- mcnemar_result$p.value
    odds_ratio <- mcnemar_result$estimate
    odds_ratio_ci <- mcnemar_result$conf.int
  } else {
    # Fallback to base R mcnemar.test if exact2x2 fails
    fallback_test <- mcnemar.test(ct$matrix, correct = FALSE)
    p_value <- fallback_test$p.value
    odds_ratio <- ct$b / ct$c  # Simple odds ratio
    odds_ratio_ci <- c(NA_real_, NA_real_)
  }

  # Assemble results tibble
  tibble(
    family = family,
    family_label = FAMILY_LABELS[family],
    n = ct$n,
    cell_a = ct$a,
    cell_b = ct$b,
    cell_c = ct$c,
    cell_d = ct$d,
    discordant_pairs = ct$b + ct$c,
    chi_squared = chi_sq,
    p_value = p_value,
    odds_ratio = odds_ratio,
    odds_ratio_ci_lower = odds_ratio_ci[1],
    odds_ratio_ci_upper = odds_ratio_ci[2],
    arr = arr_result$arr,
    arr_se = arr_result$se,
    arr_ci_lower = arr_result$ci_lower,
    arr_ci_upper = arr_result$ci_upper,
    base_toxic_rate = (ct$b + ct$d) / ct$n,
    aligned_toxic_rate = (ct$c + ct$d) / ct$n,
    alignment_success_rate = ct$b / (ct$b + ct$d),
    alignment_failure_rate = ct$c / (ct$a + ct$c),
    significant = p_value < ALPHA
  )
}

#' Generate statistical interpretation text
#'
#' @param results tibble with McNemar test results
#' @return character vector with interpretation
generate_interpretation <- function(results) {
  interpretations <- character()

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]

    # Significance statement
    sig_text <- if (r$significant) {
      sprintf(
        "SIGNIFICANT: McNemar's test indicates a statistically significant difference (p = %.2e).",
        r$p_value
      )
    } else {
      sprintf(
        "NOT SIGNIFICANT: McNemar's test does not indicate a statistically significant difference (p = %.4f).",
        r$p_value
      )
    }

    # Effect direction
    direction_text <- if (r$arr > 0) {
      "Alignment REDUCED toxicity."
    } else if (r$arr < 0) {
      "Alignment INCREASED toxicity."
    } else {
      "No net change in toxicity."
    }

    # ARR interpretation
    arr_text <- sprintf(
      "Absolute Risk Reduction (ARR) = %.2f%% (95%% CI: %.2f%% to %.2f%%).",
      r$arr * 100,
      r$arr_ci_lower * 100,
      r$arr_ci_upper * 100
    )

    # Practical significance
    practical_text <- if (abs(r$arr) >= 0.10) {
      "This represents a LARGE practical effect (ARR >= 10%)."
    } else if (abs(r$arr) >= 0.05) {
      "This represents a MODERATE practical effect (5% <= ARR < 10%)."
    } else if (abs(r$arr) >= 0.01) {
      "This represents a SMALL practical effect (1% <= ARR < 5%)."
    } else {
      "This represents a NEGLIGIBLE practical effect (ARR < 1%)."
    }

    # Contingency table summary
    table_text <- sprintf(
      "Contingency table: a=%s, b=%s (successes), c=%s (failures), d=%s.",
      format(r$cell_a, big.mark = ","),
      format(r$cell_b, big.mark = ","),
      format(r$cell_c, big.mark = ","),
      format(r$cell_d, big.mark = ",")
    )

    # Rate summary
    rate_text <- sprintf(
      "Toxicity rates: Base=%.1f%%, Aligned=%.1f%% (reduction of %.1f percentage points).",
      r$base_toxic_rate * 100,
      r$aligned_toxic_rate * 100,
      (r$base_toxic_rate - r$aligned_toxic_rate) * 100
    )

    # Combine interpretation
    interpretation <- paste(
      sprintf("\n--- %s ---", r$family_label),
      sig_text,
      direction_text,
      arr_text,
      practical_text,
      table_text,
      rate_text,
      sep = "\n"
    )

    interpretations <- c(interpretations, interpretation)
  }

  # Overall summary
  all_sig <- all(results$significant)
  overall_direction <- if (all(results$arr > 0)) {
    "All three model families showed toxicity reduction with alignment."
  } else if (all(results$arr < 0)) {
    "All three model families showed toxicity increase with alignment."
  } else {
    "Mixed results across model families."
  }

  avg_arr <- mean(results$arr)
  overall_summary <- paste(
    "\n=== OVERALL SUMMARY ===",
    sprintf("All families significant: %s", if(all_sig) "YES" else "NO"),
    overall_direction,
    sprintf("Average ARR across families: %.2f%%", avg_arr * 100),
    sep = "\n"
  )

  c(interpretations, overall_summary)
}

#' Print formatted results summary
#'
#' @param results tibble with McNemar test results
#' @param interpretations character vector with interpretations
print_results_summary <- function(results, interpretations) {
  cat("\n")
  cat("=============================================================================\n")
  cat("                    McNEMAR'S TEST RESULTS\n")
  cat("=============================================================================\n\n")

  cat("HYPOTHESIS:\n")
  cat("  H0: b = c (alignment has no effect on toxicity)\n")
  cat("  H1: b != c (alignment affects toxicity)\n")
  cat(sprintf("  Alpha level: %.2f\n", ALPHA))
  cat(sprintf("  Bootstrap replicates for ARR CI: %d\n\n", BOOTSTRAP_REPLICATES))

  cat("CONTINGENCY TABLE STRUCTURE:\n")
  cat("  |                   | Aligned: Non-toxic | Aligned: Toxic |\n")
  cat("  |-------------------|-------------------|----------------|\n")
  cat("  | Base: Non-toxic   | a                 | c              |\n")
  cat("  | Base: Toxic       | b                 | d              |\n\n")
  cat("  b = alignment success (base toxic -> aligned non-toxic)\n")
  cat("  c = alignment failure (base non-toxic -> aligned toxic)\n\n")

  cat("-----------------------------------------------------------------------------\n")
  cat("RESULTS BY FAMILY:\n")
  cat("-----------------------------------------------------------------------------\n")

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]
    cat(sprintf("\n%s:\n", r$family_label))
    cat(sprintf("  Sample size: %s\n", format(r$n, big.mark = ",")))
    cat(sprintf("  Contingency table: a=%s, b=%s, c=%s, d=%s\n",
                format(r$cell_a, big.mark = ","),
                format(r$cell_b, big.mark = ","),
                format(r$cell_c, big.mark = ","),
                format(r$cell_d, big.mark = ",")))
    cat(sprintf("  Discordant pairs: %s\n", format(r$discordant_pairs, big.mark = ",")))
    cat(sprintf("  Chi-squared statistic: %.2f\n", r$chi_squared))
    cat(sprintf("  p-value: %.4e\n", r$p_value))
    cat(sprintf("  Odds ratio: %.3f (95%% CI: %.3f - %.3f)\n",
                r$odds_ratio, r$odds_ratio_ci_lower, r$odds_ratio_ci_upper))
    cat(sprintf("  ARR: %.4f (%.2f%%)\n", r$arr, r$arr * 100))
    cat(sprintf("  ARR 95%% CI: [%.4f, %.4f] ([%.2f%%, %.2f%%])\n",
                r$arr_ci_lower, r$arr_ci_upper,
                r$arr_ci_lower * 100, r$arr_ci_upper * 100))
    cat(sprintf("  Significant: %s\n", if(r$significant) "YES" else "NO"))
  }

  cat("\n-----------------------------------------------------------------------------\n")
  cat("INTERPRETATION:\n")
  cat("-----------------------------------------------------------------------------\n")
  cat(paste(interpretations, collapse = "\n"))
  cat("\n\n")

  cat("=============================================================================\n")
  cat("END OF RESULTS\n")
  cat("=============================================================================\n\n")
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  use_mock <- "--mock" %in% args

  cat("\nLLM Safety Alignment Study - McNemar's Test Analysis\n")
  cat("=====================================================\n\n")

  # Create output directory
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
    cat(sprintf("Created output directory: %s\n", OUTPUT_DIR))
  }

  # Load data
  if (file.exists(INPUT_RDS_PATH) && !use_mock) {
    cat(sprintf("Loading validated data from: %s\n", INPUT_RDS_PATH))
    loaded <- readRDS(INPUT_RDS_PATH)
    data <- loaded$data
    cat(sprintf("Loaded %s rows\n\n", format(nrow(data), big.mark = ",")))
  } else {
    if (!use_mock) {
      cat(sprintf("WARNING: %s not found.\n", INPUT_RDS_PATH))
    }
    cat("Using mock data for analysis.\n\n")
    data <- generate_mock_data()
  }

  # Validate required columns
  required_cols <- c("prompt_id", "family", "tox_binary_base", "tox_binary_aligned")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # Perform McNemar's test for each family
  cat("Performing McNemar's test for each model family...\n\n")

  results <- map_dfr(VALID_FAMILIES, function(fam) {
    cat(sprintf("  Processing %s...\n", FAMILY_LABELS[fam]))
    family_data <- data %>% filter(family == fam)

    if (nrow(family_data) == 0) {
      warning(sprintf("No data for family: %s", fam))
      return(NULL)
    }

    perform_mcnemar_test(family_data, fam)
  })

  # Generate interpretation
  interpretations <- generate_interpretation(results)

  # Print results summary
  print_results_summary(results, interpretations)

  # Save results to CSV
  output_path <- file.path(OUTPUT_DIR, OUTPUT_FILE)
  write_csv(results, output_path)
  cat(sprintf("Results saved to: %s\n\n", output_path))

  # Save interpretation to text file
  interpretation_path <- file.path(OUTPUT_DIR, "mcnemar_interpretation.txt")
  writeLines(
    c(
      "McNemar's Test Interpretation",
      "==============================",
      "",
      interpretations
    ),
    interpretation_path
  )
  cat(sprintf("Interpretation saved to: %s\n\n", interpretation_path))

  return(invisible(0))
}

# Run main function if script is executed directly
if (!interactive()) {
  status <- main()
  quit(status = status)
}
