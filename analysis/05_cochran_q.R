#!/usr/bin/env Rscript
# =============================================================================
# 05_cochran_q.R
# Cochran's Q Test for Cross-Family Comparison
# =============================================================================
#
# Purpose: Perform Cochran's Q test to determine whether alignment success
#          rates differ significantly across model families.
#
# Input:  analysis/data_validated.rds
# Output: output/tables/cochran_q_results.csv
#
# Statistical Method:
#   Cochran's Q test is an extension of McNemar's test for k related samples.
#   It tests whether the proportion of "successes" differs across groups.
#
#   Definition of "success" for this study:
#   - Base model produced toxic output (tox_binary_base = 1)
#   - Aligned model produced non-toxic output (tox_binary_aligned = 0)
#   - This represents alignment successfully mitigating toxicity
#
#   Hypotheses:
#   - H0: Alignment success rate is equal across all model families
#   - H1: At least one family differs in alignment success rate
#
# Usage:  Rscript analysis/05_cochran_q.R [--mock]
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(DescTools)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_RDS_PATH <- "analysis/data_validated.rds"
OUTPUT_DIR <- "output/tables"
OUTPUT_FILE <- "cochran_q_results.csv"
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
#' Ensures same prompts appear across all three families for proper pairing.
#'
#' @param n_prompts Number of unique prompts (default 25000)
#' @param seed Random seed for reproducibility
#' @return tibble with mock analysis data
generate_mock_data <- function(n_prompts = 25000, seed = RANDOM_SEED) {
  set.seed(seed)

  prompt_ids <- sprintf("prompt_%05d", seq_len(n_prompts))

  # Create base data with all family combinations
  data <- expand_grid(
    prompt_id = prompt_ids,
    family = VALID_FAMILIES
  )

  # Simulate realistic toxicity patterns
  # Base models: ~35% toxic
  # Aligned models: ~12% toxic
  # But with some prompt-level correlation (some prompts are harder)

  # Generate prompt-level difficulty (affects all families)
  prompt_difficulty <- tibble(
    prompt_id = prompt_ids,
    difficulty = rbeta(n_prompts, 2, 3)  # Higher = more likely to be toxic
  )

  data <- data %>%
    left_join(prompt_difficulty, by = "prompt_id") %>%
    mutate(
      # Base toxicity probability varies by difficulty and family
      base_prob = case_when(
        family == "qwen3" ~ pmin(1, difficulty + runif(n(), 0, 0.2)),
        family == "llama31" ~ pmin(1, difficulty + runif(n(), 0, 0.25)),
        family == "mistral" ~ pmin(1, difficulty + runif(n(), 0, 0.3))
      ),
      # Generate base toxicity
      tox_binary_base = as.integer(runif(n()) < base_prob),
      # Aligned reduces toxicity with some family variation
      aligned_reduction = case_when(
        family == "qwen3" ~ runif(n(), 0.6, 0.85),
        family == "llama31" ~ runif(n(), 0.5, 0.75),
        family == "mistral" ~ runif(n(), 0.55, 0.8)
      ),
      aligned_prob = base_prob * (1 - aligned_reduction),
      tox_binary_aligned = as.integer(runif(n()) < aligned_prob)
    ) %>%
    select(-difficulty, -base_prob, -aligned_reduction, -aligned_prob)

  return(data)
}

#' Prepare data for Cochran's Q test
#'
#' Reshapes data into wide format with one row per prompt and columns
#' for each family's alignment success indicator.
#'
#' @param data tibble with tox_binary_base and tox_binary_aligned columns
#' @return tibble in wide format for Cochran's Q test
prepare_cochran_data <- function(data) {
  # Define alignment success: base was toxic AND aligned was non-toxic
  data_with_success <- data %>%
    mutate(
      alignment_success = as.integer(tox_binary_base == 1 & tox_binary_aligned == 0)
    )

  # Pivot to wide format: one row per prompt, columns for each family
  wide_data <- data_with_success %>%
    select(prompt_id, family, alignment_success) %>%
    pivot_wider(
      names_from = family,
      values_from = alignment_success,
      names_prefix = "success_"
    )

  # Verify all families present
  expected_cols <- paste0("success_", VALID_FAMILIES)
  if (!all(expected_cols %in% names(wide_data))) {
    missing <- setdiff(expected_cols, names(wide_data))
    stop(sprintf("Missing family columns: %s", paste(missing, collapse = ", ")))
  }

  # Remove rows with any NA (incomplete observations)
  complete_data <- wide_data %>%
    filter(complete.cases(.))

  return(complete_data)
}

#' Perform Cochran's Q test
#'
#' @param wide_data tibble in wide format from prepare_cochran_data
#' @return list with test results
perform_cochran_q_test <- function(wide_data) {
  # Extract success columns as matrix
  success_cols <- paste0("success_", VALID_FAMILIES)
  success_matrix <- as.matrix(wide_data[, success_cols])

  # Perform Cochran's Q test using DescTools
  cochran_result <- CochranQTest(success_matrix)

  # Extract results
  q_statistic <- cochran_result$statistic
  df <- cochran_result$parameter
  p_value <- cochran_result$p.value

  list(
    q_statistic = q_statistic,
    df = df,
    p_value = p_value,
    method = cochran_result$method,
    n_subjects = nrow(wide_data),
    n_treatments = length(VALID_FAMILIES)
  )
}

#' Calculate family-level success rates
#'
#' @param data tibble with alignment success indicator
#' @return tibble with success rates by family
calculate_family_rates <- function(data) {
  data %>%
    mutate(
      alignment_success = as.integer(tox_binary_base == 1 & tox_binary_aligned == 0),
      base_toxic = tox_binary_base,
      aligned_toxic = tox_binary_aligned
    ) %>%
    group_by(family) %>%
    summarize(
      n = n(),
      n_base_toxic = sum(base_toxic, na.rm = TRUE),
      n_aligned_toxic = sum(aligned_toxic, na.rm = TRUE),
      n_alignment_success = sum(alignment_success, na.rm = TRUE),
      base_toxic_rate = mean(base_toxic, na.rm = TRUE),
      aligned_toxic_rate = mean(aligned_toxic, na.rm = TRUE),
      success_rate = mean(alignment_success, na.rm = TRUE),
      success_given_base_toxic = n_alignment_success / n_base_toxic,
      .groups = "drop"
    ) %>%
    mutate(family_label = FAMILY_LABELS[family])
}

#' Perform pairwise McNemar tests for post-hoc analysis
#'
#' @param wide_data tibble in wide format
#' @return tibble with pairwise comparison results
perform_pairwise_comparisons <- function(wide_data) {
  # All pairs of families
  pairs <- combn(VALID_FAMILIES, 2, simplify = FALSE)

  pairwise_results <- map_dfr(pairs, function(pair) {
    fam1 <- pair[1]
    fam2 <- pair[2]

    col1 <- paste0("success_", fam1)
    col2 <- paste0("success_", fam2)

    # Build 2x2 table
    a <- sum(wide_data[[col1]] == 0 & wide_data[[col2]] == 0)
    b <- sum(wide_data[[col1]] == 1 & wide_data[[col2]] == 0)
    c <- sum(wide_data[[col1]] == 0 & wide_data[[col2]] == 1)
    d <- sum(wide_data[[col1]] == 1 & wide_data[[col2]] == 1)

    mat <- matrix(c(a, b, c, d), nrow = 2, byrow = TRUE)

    # McNemar test for this pair
    test_result <- tryCatch({
      mcnemar.test(mat)
    }, error = function(e) {
      list(statistic = NA, p.value = NA)
    })

    tibble(
      family_1 = fam1,
      family_2 = fam2,
      family_1_label = FAMILY_LABELS[fam1],
      family_2_label = FAMILY_LABELS[fam2],
      cell_a = a,
      cell_b = b,
      cell_c = c,
      cell_d = d,
      chi_squared = as.numeric(test_result$statistic),
      p_value = test_result$p.value
    )
  })

  # Apply Bonferroni correction
  pairwise_results <- pairwise_results %>%
    mutate(
      p_value_bonferroni = pmin(1, p_value * nrow(pairwise_results)),
      significant_bonferroni = p_value_bonferroni < ALPHA
    )

  return(pairwise_results)
}

#' Calculate effect size (Kendall's W) for Cochran's Q
#'
#' @param wide_data tibble in wide format
#' @return numeric Kendall's W coefficient
calculate_kendalls_w <- function(wide_data) {
  success_cols <- paste0("success_", VALID_FAMILIES)
  success_matrix <- as.matrix(wide_data[, success_cols])

  n <- nrow(success_matrix)  # Number of subjects (prompts)
  k <- ncol(success_matrix)  # Number of treatments (families)

  # Row sums (total successes per prompt)
  row_sums <- rowSums(success_matrix)

  # Column sums (total successes per family)
  col_sums <- colSums(success_matrix)

  # Cochran's Q statistic
  T_total <- sum(row_sums)
  SS_between <- k * sum((col_sums - T_total/k)^2)

  # Variance
  row_variance <- sum(row_sums * (k - row_sums))

  # Kendall's W
  if (row_variance > 0) {
    w <- SS_between / row_variance
  } else {
    w <- NA
  }

  return(w)
}

#' Generate statistical interpretation text
#'
#' @param cochran_result list with Cochran's Q test results
#' @param family_rates tibble with family-level rates
#' @param pairwise tibble with pairwise comparisons
#' @param kendalls_w Kendall's W effect size
#' @return character vector with interpretation
generate_interpretation <- function(cochran_result, family_rates, pairwise, kendalls_w) {
  significant <- cochran_result$p_value < ALPHA

  # Main result interpretation
  main_text <- if (significant) {
    sprintf(
      "SIGNIFICANT: Cochran's Q test indicates that alignment success rates differ significantly across model families (Q = %.2f, df = %d, p = %.4e).",
      cochran_result$q_statistic,
      cochran_result$df,
      cochran_result$p_value
    )
  } else {
    sprintf(
      "NOT SIGNIFICANT: Cochran's Q test does not indicate significant differences in alignment success rates across families (Q = %.2f, df = %d, p = %.4f).",
      cochran_result$q_statistic,
      cochran_result$df,
      cochran_result$p_value
    )
  }

  # Effect size interpretation
  effect_text <- if (!is.na(kendalls_w)) {
    effect_desc <- if (kendalls_w >= 0.5) {
      "large"
    } else if (kendalls_w >= 0.3) {
      "medium"
    } else if (kendalls_w >= 0.1) {
      "small"
    } else {
      "negligible"
    }
    sprintf("Effect size (Kendall's W): %.4f (%s effect).", kendalls_w, effect_desc)
  } else {
    "Effect size could not be calculated."
  }

  # Family rates summary
  rates_text <- paste(
    "Alignment success rates by family:",
    paste(sprintf("  %s: %.2f%%",
                  family_rates$family_label,
                  family_rates$success_rate * 100),
          collapse = "\n"),
    sep = "\n"
  )

  # Best/worst performers
  best_family <- family_rates$family_label[which.max(family_rates$success_rate)]
  worst_family <- family_rates$family_label[which.min(family_rates$success_rate)]
  range_text <- sprintf(
    "Range: %s (highest: %.2f%%) to %s (lowest: %.2f%%).",
    best_family,
    max(family_rates$success_rate) * 100,
    worst_family,
    min(family_rates$success_rate) * 100
  )

  # Pairwise comparisons
  if (significant && nrow(pairwise) > 0) {
    sig_pairs <- pairwise %>% filter(significant_bonferroni)
    if (nrow(sig_pairs) > 0) {
      pairwise_text <- paste(
        "\nSignificant pairwise differences (Bonferroni-corrected):",
        paste(sprintf("  %s vs %s: p = %.4e",
                      sig_pairs$family_1_label,
                      sig_pairs$family_2_label,
                      sig_pairs$p_value_bonferroni),
              collapse = "\n"),
        sep = "\n"
      )
    } else {
      pairwise_text <- "\nNo significant pairwise differences after Bonferroni correction."
    }
  } else {
    pairwise_text <- ""
  }

  # Conclusion
  conclusion_text <- if (significant) {
    sprintf(
      "\nCONCLUSION: The effectiveness of safety alignment varies significantly across model families. %s shows the highest success rate, while %s shows the lowest.",
      best_family, worst_family
    )
  } else {
    "\nCONCLUSION: Safety alignment appears to be equally effective across all three model families."
  }

  paste(
    main_text,
    effect_text,
    "",
    rates_text,
    range_text,
    pairwise_text,
    conclusion_text,
    sep = "\n"
  )
}

#' Print formatted results summary
#'
#' @param cochran_result list with Cochran's Q test results
#' @param family_rates tibble with family-level rates
#' @param pairwise tibble with pairwise comparisons
#' @param kendalls_w Kendall's W effect size
#' @param interpretation character with interpretation text
print_results_summary <- function(cochran_result, family_rates, pairwise, kendalls_w, interpretation) {
  cat("\n")
  cat("=============================================================================\n")
  cat("                    COCHRAN'S Q TEST RESULTS\n")
  cat("=============================================================================\n\n")

  cat("HYPOTHESIS:\n")
  cat("  H0: Alignment success rate is equal across all model families\n")
  cat("  H1: At least one family differs in alignment success rate\n")
  cat(sprintf("  Alpha level: %.2f\n\n", ALPHA))

  cat("DEFINITION OF SUCCESS:\n")
  cat("  'Alignment success' = Base model was toxic AND aligned model was non-toxic\n")
  cat("  This measures cases where alignment actively prevented toxic output.\n\n")

  cat("-----------------------------------------------------------------------------\n")
  cat("COCHRAN'S Q TEST:\n")
  cat("-----------------------------------------------------------------------------\n")
  cat(sprintf("  Number of subjects (prompts): %s\n", format(cochran_result$n_subjects, big.mark = ",")))
  cat(sprintf("  Number of treatments (families): %d\n", cochran_result$n_treatments))
  cat(sprintf("  Q statistic: %.4f\n", cochran_result$q_statistic))
  cat(sprintf("  Degrees of freedom: %d\n", cochran_result$df))
  cat(sprintf("  p-value: %.4e\n", cochran_result$p_value))
  cat(sprintf("  Kendall's W (effect size): %.4f\n", kendalls_w))
  cat(sprintf("  Significant: %s\n\n", if(cochran_result$p_value < ALPHA) "YES" else "NO"))

  cat("-----------------------------------------------------------------------------\n")
  cat("FAMILY-LEVEL SUCCESS RATES:\n")
  cat("-----------------------------------------------------------------------------\n")
  for (i in seq_len(nrow(family_rates))) {
    r <- family_rates[i, ]
    cat(sprintf("\n%s:\n", r$family_label))
    cat(sprintf("  Sample size: %s\n", format(r$n, big.mark = ",")))
    cat(sprintf("  Base toxic: %s (%.1f%%)\n",
                format(r$n_base_toxic, big.mark = ","),
                r$base_toxic_rate * 100))
    cat(sprintf("  Aligned toxic: %s (%.1f%%)\n",
                format(r$n_aligned_toxic, big.mark = ","),
                r$aligned_toxic_rate * 100))
    cat(sprintf("  Alignment successes: %s\n",
                format(r$n_alignment_success, big.mark = ",")))
    cat(sprintf("  Success rate (overall): %.2f%%\n", r$success_rate * 100))
    cat(sprintf("  Success rate (given base toxic): %.2f%%\n", r$success_given_base_toxic * 100))
  }

  cat("\n-----------------------------------------------------------------------------\n")
  cat("PAIRWISE COMPARISONS (Bonferroni-corrected):\n")
  cat("-----------------------------------------------------------------------------\n")
  for (i in seq_len(nrow(pairwise))) {
    p <- pairwise[i, ]
    cat(sprintf("\n%s vs %s:\n", p$family_1_label, p$family_2_label))
    cat(sprintf("  Chi-squared: %.4f\n", p$chi_squared))
    cat(sprintf("  Uncorrected p-value: %.4e\n", p$p_value))
    cat(sprintf("  Bonferroni p-value: %.4e\n", p$p_value_bonferroni))
    cat(sprintf("  Significant: %s\n", if(p$significant_bonferroni) "YES" else "NO"))
  }

  cat("\n-----------------------------------------------------------------------------\n")
  cat("INTERPRETATION:\n")
  cat("-----------------------------------------------------------------------------\n")
  cat(interpretation)
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

  cat("\nLLM Safety Alignment Study - Cochran's Q Test Analysis\n")
  cat("======================================================\n\n")

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

  # Validate all families present
  available_families <- unique(data$family)
  missing_families <- setdiff(VALID_FAMILIES, available_families)
  if (length(missing_families) > 0) {
    stop(sprintf("Missing families: %s", paste(missing_families, collapse = ", ")))
  }

  # Calculate family-level rates
  cat("Calculating family-level success rates...\n")
  family_rates <- calculate_family_rates(data)

  # Prepare data for Cochran's Q test
  cat("Preparing data for Cochran's Q test...\n")
  wide_data <- prepare_cochran_data(data)
  cat(sprintf("  Prepared %s complete observations\n\n", format(nrow(wide_data), big.mark = ",")))

  # Perform Cochran's Q test
  cat("Performing Cochran's Q test...\n")
  cochran_result <- perform_cochran_q_test(wide_data)

  # Calculate effect size
  cat("Calculating effect size (Kendall's W)...\n")
  kendalls_w <- calculate_kendalls_w(wide_data)

  # Perform pairwise comparisons
  cat("Performing pairwise comparisons...\n\n")
  pairwise <- perform_pairwise_comparisons(wide_data)

  # Generate interpretation
  interpretation <- generate_interpretation(cochran_result, family_rates, pairwise, kendalls_w)

  # Print results summary
  print_results_summary(cochran_result, family_rates, pairwise, kendalls_w, interpretation)

  # Prepare output tibble
  results <- tibble(
    test = "Cochran's Q",
    q_statistic = cochran_result$q_statistic,
    df = cochran_result$df,
    p_value = cochran_result$p_value,
    kendalls_w = kendalls_w,
    n_subjects = cochran_result$n_subjects,
    n_treatments = cochran_result$n_treatments,
    significant = cochran_result$p_value < ALPHA,
    alpha = ALPHA
  )

  # Save main results to CSV
  output_path <- file.path(OUTPUT_DIR, OUTPUT_FILE)
  write_csv(results, output_path)
  cat(sprintf("Main results saved to: %s\n", output_path))

  # Save family rates to CSV
  family_rates_path <- file.path(OUTPUT_DIR, "cochran_q_family_rates.csv")
  write_csv(family_rates, family_rates_path)
  cat(sprintf("Family rates saved to: %s\n", family_rates_path))

  # Save pairwise comparisons to CSV
  pairwise_path <- file.path(OUTPUT_DIR, "cochran_q_pairwise.csv")
  write_csv(pairwise, pairwise_path)
  cat(sprintf("Pairwise comparisons saved to: %s\n", pairwise_path))

  # Save interpretation to text file
  interpretation_path <- file.path(OUTPUT_DIR, "cochran_q_interpretation.txt")
  writeLines(
    c(
      "Cochran's Q Test Interpretation",
      "================================",
      "",
      interpretation
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
