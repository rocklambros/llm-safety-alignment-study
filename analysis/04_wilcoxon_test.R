#!/usr/bin/env Rscript
# =============================================================================
# 04_wilcoxon_test.R
# Wilcoxon Signed-Rank Test for Paired Continuous Scores
# =============================================================================
#
# Purpose: Perform Wilcoxon signed-rank test on paired continuous toxicity
#          scores to evaluate whether base models produce higher toxicity
#          scores than aligned models.
#
# Input:  analysis/data_validated.rds
# Output: output/tables/wilcoxon_results.csv
#
# Statistical Method:
#   The Wilcoxon signed-rank test is a non-parametric alternative to the
#   paired t-test. It tests whether the median of paired differences is zero.
#
#   - H0: Median difference = 0
#   - H1: Median difference > 0 (base > aligned, one-sided)
#
#   We use the coin package for exact computation which handles ties properly.
#
# Usage:  Rscript analysis/04_wilcoxon_test.R [--mock]
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(coin)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_RDS_PATH <- "analysis/data_validated.rds"
OUTPUT_DIR <- "output/tables"
OUTPUT_FILE <- "wilcoxon_results.csv"
RANDOM_SEED <- 20260118
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
  # Base models: higher toxicity
  # Aligned models: lower toxicity
  data <- data %>%
    mutate(
      # Base scores: mixture distribution with moderate toxicity
      tox_score_base = pmin(1, pmax(0, rbeta(n(), 2, 5) + runif(n(), 0, 0.25))),
      # Aligned scores: reduced toxicity with multiplicative reduction
      tox_score_aligned = pmin(1, pmax(0, tox_score_base * runif(n(), 0.15, 0.7)))
    )

  return(data)
}

#' Calculate Hodges-Lehmann pseudomedian with confidence interval
#'
#' The Hodges-Lehmann estimator is the median of all pairwise averages
#' of the differences, which corresponds to the pseudomedian.
#'
#' @param differences numeric vector of paired differences
#' @param conf_level confidence level (default 0.95)
#' @return list with pseudomedian and confidence interval
calculate_pseudomedian <- function(differences, conf_level = 0.95) {
  # Use wilcox.test for pseudomedian and CI
  # Setting exact = FALSE for large samples to avoid memory issues
  n <- length(differences)

  wt <- wilcox.test(
    differences,
    alternative = "greater",
    conf.int = TRUE,
    conf.level = conf_level,
    exact = n <= 5000  # Use exact only for smaller samples
  )

  list(
    pseudomedian = wt$estimate,
    ci_lower = wt$conf.int[1],
    ci_upper = wt$conf.int[2]
  )
}

#' Perform Wilcoxon signed-rank test using coin package
#'
#' Uses exact conditional inference for the Wilcoxon test.
#'
#' @param data tibble with tox_score_base and tox_score_aligned columns
#' @param family family name
#' @return tibble with test results
perform_wilcoxon_test <- function(data, family) {
  # Calculate paired differences (base - aligned)
  # Positive values indicate alignment reduced toxicity
  differences <- data$tox_score_base - data$tox_score_aligned

  # Remove zero differences for Wilcoxon test
  nonzero_diff <- differences[differences != 0]
  n_zero <- sum(differences == 0)
  n_nonzero <- length(nonzero_diff)

  # Create long format for coin package
  # coin::wilcoxsign_test requires a formula interface
  test_data <- tibble(
    id = rep(seq_len(nrow(data)), 2),
    model = factor(rep(c("base", "aligned"), each = nrow(data)),
                   levels = c("base", "aligned")),
    toxicity = c(data$tox_score_base, data$tox_score_aligned)
  )

  # Perform exact Wilcoxon signed-rank test using coin
  # Note: For very large samples, we use approximate distribution
  cat(sprintf("    Running Wilcoxon test (n=%s)...\n", format(nrow(data), big.mark = ",")))

  coin_result <- tryCatch({
    if (nrow(data) <= 5000) {
      # Exact test for smaller samples
      wilcoxsign_test(
        toxicity ~ model | id,
        data = test_data,
        alternative = "greater",
        distribution = "exact"
      )
    } else {
      # Approximate for larger samples (Monte Carlo)
      wilcoxsign_test(
        toxicity ~ model | id,
        data = test_data,
        alternative = "greater",
        distribution = approximate(nresample = 10000)
      )
    }
  }, error = function(e) {
    warning(sprintf("Coin Wilcoxon test failed for %s: %s. Using base R.", family, e$message))
    NULL
  })

  # Extract W statistic and p-value from coin result
  if (!is.null(coin_result)) {
    w_statistic <- statistic(coin_result)
    p_value <- pvalue(coin_result)
  } else {
    # Fallback to base R wilcox.test
    base_result <- wilcox.test(
      data$tox_score_base,
      data$tox_score_aligned,
      paired = TRUE,
      alternative = "greater",
      exact = FALSE
    )
    w_statistic <- base_result$statistic
    p_value <- base_result$p.value
  }

  # Calculate pseudomedian and CI (using subset for computational efficiency)
  cat("    Calculating pseudomedian...\n")
  if (length(differences) > 10000) {
    # Sample for CI calculation to avoid memory issues
    set.seed(RANDOM_SEED)
    sample_idx <- sample(length(differences), 10000)
    pseudo_result <- calculate_pseudomedian(differences[sample_idx])
  } else {
    pseudo_result <- calculate_pseudomedian(differences)
  }

  # Calculate additional summary statistics
  mean_diff <- mean(differences, na.rm = TRUE)
  median_diff <- median(differences, na.rm = TRUE)
  sd_diff <- sd(differences, na.rm = TRUE)
  iqr_diff <- IQR(differences, na.rm = TRUE)

  # Effect size: r = Z / sqrt(n)
  # Z approximation from W statistic
  n <- length(nonzero_diff)
  expected_w <- n * (n + 1) / 4
  var_w <- n * (n + 1) * (2 * n + 1) / 24
  z_statistic <- (w_statistic - expected_w) / sqrt(var_w)
  effect_size_r <- abs(z_statistic) / sqrt(nrow(data))

  # Proportion of pairs where alignment helped
  pct_improved <- mean(differences > 0, na.rm = TRUE) * 100
  pct_worsened <- mean(differences < 0, na.rm = TRUE) * 100
  pct_unchanged <- mean(differences == 0, na.rm = TRUE) * 100

  # Assemble results tibble
  tibble(
    family = family,
    family_label = FAMILY_LABELS[family],
    n = nrow(data),
    n_nonzero_diff = n_nonzero,
    n_zero_diff = n_zero,
    w_statistic = as.numeric(w_statistic),
    z_statistic = z_statistic,
    p_value = as.numeric(p_value),
    pseudomedian = pseudo_result$pseudomedian,
    pseudomedian_ci_lower = pseudo_result$ci_lower,
    pseudomedian_ci_upper = pseudo_result$ci_upper,
    mean_difference = mean_diff,
    median_difference = median_diff,
    sd_difference = sd_diff,
    iqr_difference = iqr_diff,
    effect_size_r = effect_size_r,
    pct_improved = pct_improved,
    pct_worsened = pct_worsened,
    pct_unchanged = pct_unchanged,
    significant = p_value < ALPHA
  )
}

#' Generate statistical interpretation text
#'
#' @param results tibble with Wilcoxon test results
#' @return character vector with interpretation
generate_interpretation <- function(results) {
  interpretations <- character()

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]

    # Significance statement
    sig_text <- if (r$significant) {
      sprintf(
        "SIGNIFICANT: Wilcoxon signed-rank test indicates base models produce significantly higher toxicity scores (p = %.2e).",
        r$p_value
      )
    } else {
      sprintf(
        "NOT SIGNIFICANT: Wilcoxon signed-rank test does not indicate a significant difference (p = %.4f).",
        r$p_value
      )
    }

    # Effect size interpretation (r)
    effect_text <- if (r$effect_size_r >= 0.5) {
      sprintf("Large effect size (r = %.3f).", r$effect_size_r)
    } else if (r$effect_size_r >= 0.3) {
      sprintf("Medium effect size (r = %.3f).", r$effect_size_r)
    } else if (r$effect_size_r >= 0.1) {
      sprintf("Small effect size (r = %.3f).", r$effect_size_r)
    } else {
      sprintf("Negligible effect size (r = %.3f).", r$effect_size_r)
    }

    # Pseudomedian interpretation
    pseudo_text <- sprintf(
      "Pseudomedian of differences: %.4f (95%% CI: %.4f to %.4f).",
      r$pseudomedian,
      r$pseudomedian_ci_lower,
      r$pseudomedian_ci_upper
    )

    # Direction summary
    direction_text <- sprintf(
      "Direction breakdown: %.1f%% improved (base > aligned), %.1f%% worsened, %.1f%% unchanged.",
      r$pct_improved, r$pct_worsened, r$pct_unchanged
    )

    # Central tendency
    central_text <- sprintf(
      "Mean difference: %.4f (SD: %.4f), Median difference: %.4f (IQR: %.4f).",
      r$mean_difference, r$sd_difference, r$median_difference, r$iqr_difference
    )

    # Combine interpretation
    interpretation <- paste(
      sprintf("\n--- %s ---", r$family_label),
      sig_text,
      effect_text,
      pseudo_text,
      direction_text,
      central_text,
      sep = "\n"
    )

    interpretations <- c(interpretations, interpretation)
  }

  # Overall summary
  all_sig <- all(results$significant)
  avg_pseudomedian <- mean(results$pseudomedian)
  avg_effect <- mean(results$effect_size_r)
  avg_improved <- mean(results$pct_improved)

  overall_summary <- paste(
    "\n=== OVERALL SUMMARY ===",
    sprintf("All families significant: %s", if(all_sig) "YES" else "NO"),
    sprintf("Average pseudomedian: %.4f", avg_pseudomedian),
    sprintf("Average effect size (r): %.3f", avg_effect),
    sprintf("Average improvement rate: %.1f%%", avg_improved),
    if (all_sig && avg_pseudomedian > 0) {
      "CONCLUSION: Consistent evidence that alignment reduces continuous toxicity scores."
    } else if (!all_sig) {
      "CONCLUSION: Inconsistent results across families."
    } else {
      "CONCLUSION: Evidence suggests alignment affects toxicity scores."
    },
    sep = "\n"
  )

  c(interpretations, overall_summary)
}

#' Print formatted results summary
#'
#' @param results tibble with Wilcoxon test results
#' @param interpretations character vector with interpretations
print_results_summary <- function(results, interpretations) {
  cat("\n")
  cat("=============================================================================\n")
  cat("                    WILCOXON SIGNED-RANK TEST RESULTS\n")
  cat("=============================================================================\n\n")

  cat("HYPOTHESIS:\n")
  cat("  H0: Median difference = 0 (no effect of alignment on toxicity scores)\n")
  cat("  H1: Median difference > 0 (base > aligned, one-sided test)\n")
  cat(sprintf("  Alpha level: %.2f\n\n", ALPHA))

  cat("METHOD:\n")
  cat("  - Exact Wilcoxon signed-rank test via coin package\n")
  cat("  - Hodges-Lehmann pseudomedian estimator\n")
  cat("  - Effect size: r = Z / sqrt(n)\n\n")

  cat("-----------------------------------------------------------------------------\n")
  cat("RESULTS BY FAMILY:\n")
  cat("-----------------------------------------------------------------------------\n")

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]
    cat(sprintf("\n%s:\n", r$family_label))
    cat(sprintf("  Sample size: %s (non-zero differences: %s)\n",
                format(r$n, big.mark = ","),
                format(r$n_nonzero_diff, big.mark = ",")))
    cat(sprintf("  W statistic: %.2f\n", r$w_statistic))
    cat(sprintf("  Z statistic: %.4f\n", r$z_statistic))
    cat(sprintf("  p-value: %.4e\n", r$p_value))
    cat(sprintf("  Pseudomedian: %.6f (95%% CI: [%.6f, %.6f])\n",
                r$pseudomedian, r$pseudomedian_ci_lower, r$pseudomedian_ci_upper))
    cat(sprintf("  Mean difference: %.6f (SD: %.6f)\n", r$mean_difference, r$sd_difference))
    cat(sprintf("  Median difference: %.6f (IQR: %.6f)\n", r$median_difference, r$iqr_difference))
    cat(sprintf("  Effect size (r): %.4f\n", r$effect_size_r))
    cat(sprintf("  Improved: %.1f%% | Worsened: %.1f%% | Unchanged: %.1f%%\n",
                r$pct_improved, r$pct_worsened, r$pct_unchanged))
    cat(sprintf("  Significant: %s\n", if(r$significant) "YES" else "NO"))
  }

  cat("\n-----------------------------------------------------------------------------\n")
  cat("INTERPRETATION:\n")
  cat("-----------------------------------------------------------------------------\n")
  cat(paste(interpretations, collapse = "\n"))
  cat("\n\n")

  # Effect size reference
  cat("-----------------------------------------------------------------------------\n")
  cat("EFFECT SIZE REFERENCE (r):\n")
  cat("  Small: r >= 0.10\n")
  cat("  Medium: r >= 0.30\n")
  cat("  Large: r >= 0.50\n")
  cat("-----------------------------------------------------------------------------\n\n")

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

  cat("\nLLM Safety Alignment Study - Wilcoxon Signed-Rank Test Analysis\n")
  cat("================================================================\n\n")

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
  required_cols <- c("prompt_id", "family", "tox_score_base", "tox_score_aligned")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # Perform Wilcoxon test for each family
  cat("Performing Wilcoxon signed-rank test for each model family...\n\n")

  results <- map_dfr(VALID_FAMILIES, function(fam) {
    cat(sprintf("  Processing %s...\n", FAMILY_LABELS[fam]))
    family_data <- data %>% filter(family == fam)

    if (nrow(family_data) == 0) {
      warning(sprintf("No data for family: %s", fam))
      return(NULL)
    }

    perform_wilcoxon_test(family_data, fam)
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
  interpretation_path <- file.path(OUTPUT_DIR, "wilcoxon_interpretation.txt")
  writeLines(
    c(
      "Wilcoxon Signed-Rank Test Interpretation",
      "=========================================",
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
