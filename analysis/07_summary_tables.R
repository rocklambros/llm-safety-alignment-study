#!/usr/bin/env Rscript
# =============================================================================
# 07_summary_tables.R
# Publication-Ready Summary Tables
# =============================================================================
#
# Purpose: Generate publication-ready summary tables combining results from
#          all statistical tests (McNemar, Wilcoxon, Cochran's Q) with proper
#          formatting for the final exposition.
#
# Input:  output/tables/mcnemar_results.csv
#         output/tables/wilcoxon_results.csv
#         output/tables/cochran_q_results.csv
#         analysis/data_validated.rds (for descriptive statistics)
#
# Output: output/tables/summary_table.csv
#         output/tables/summary_table.tex (LaTeX format)
#         output/tables/descriptive_stats.csv
#
# Usage:  Rscript analysis/07_summary_tables.R [--mock]
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_DIR <- "output/tables"
INPUT_RDS_PATH <- "analysis/data_validated.rds"
OUTPUT_DIR <- "output/tables"
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

#' Generate mock McNemar results
#'
#' @param seed Random seed
#' @return tibble with mock McNemar results
generate_mock_mcnemar <- function(seed = RANDOM_SEED) {
  set.seed(seed)

  tibble(
    family = VALID_FAMILIES,
    family_label = FAMILY_LABELS[VALID_FAMILIES],
    n = c(25000, 25000, 25000),
    cell_a = c(16500, 16200, 16000),
    cell_b = c(5800, 5900, 6100),  # Alignment successes
    cell_c = c(200, 250, 300),     # Alignment failures
    cell_d = c(2500, 2650, 2600),
    discordant_pairs = c(6000, 6150, 6400),
    chi_squared = c(5230.27, 5192.84, 5253.52),
    p_value = c(1.2e-50, 2.3e-48, 8.7e-52),
    odds_ratio = c(29.0, 23.6, 20.3),
    odds_ratio_ci_lower = c(25.1, 20.7, 18.0),
    odds_ratio_ci_upper = c(33.5, 26.9, 22.9),
    arr = c(0.224, 0.226, 0.232),
    arr_se = c(0.003, 0.003, 0.003),
    arr_ci_lower = c(0.218, 0.220, 0.226),
    arr_ci_upper = c(0.230, 0.232, 0.238),
    base_toxic_rate = c(0.332, 0.342, 0.348),
    aligned_toxic_rate = c(0.108, 0.116, 0.116),
    alignment_success_rate = c(0.699, 0.690, 0.701),
    alignment_failure_rate = c(0.012, 0.015, 0.018),
    significant = c(TRUE, TRUE, TRUE)
  )
}

#' Generate mock Wilcoxon results
#'
#' @param seed Random seed
#' @return tibble with mock Wilcoxon results
generate_mock_wilcoxon <- function(seed = RANDOM_SEED) {
  set.seed(seed)

  tibble(
    family = VALID_FAMILIES,
    family_label = FAMILY_LABELS[VALID_FAMILIES],
    n = c(25000, 25000, 25000),
    n_nonzero_diff = c(24850, 24780, 24900),
    n_zero_diff = c(150, 220, 100),
    w_statistic = c(285000000, 282000000, 290000000),
    z_statistic = c(125.4, 122.8, 128.1),
    p_value = c(1e-100, 1e-98, 1e-102),
    pseudomedian = c(0.182, 0.175, 0.188),
    pseudomedian_ci_lower = c(0.178, 0.171, 0.184),
    pseudomedian_ci_upper = c(0.186, 0.179, 0.192),
    mean_difference = c(0.195, 0.188, 0.201),
    median_difference = c(0.178, 0.172, 0.185),
    sd_difference = c(0.142, 0.148, 0.139),
    iqr_difference = c(0.185, 0.192, 0.180),
    effect_size_r = c(0.793, 0.776, 0.810),
    pct_improved = c(85.2, 83.8, 86.1),
    pct_worsened = c(8.3, 9.5, 7.8),
    pct_unchanged = c(6.5, 6.7, 6.1),
    significant = c(TRUE, TRUE, TRUE)
  )
}

#' Generate mock Cochran's Q results
#'
#' @param seed Random seed
#' @return tibble with mock Cochran's Q results
generate_mock_cochran_q <- function(seed = RANDOM_SEED) {
  tibble(
    test = "Cochran's Q",
    q_statistic = 12.84,
    df = 2,
    p_value = 0.0016,
    kendalls_w = 0.0003,
    n_subjects = 25000,
    n_treatments = 3,
    significant = TRUE,
    alpha = 0.05
  )
}

#' Generate mock analysis data
#'
#' @param n_rows Number of rows
#' @param seed Random seed
#' @return tibble with mock data
generate_mock_data <- function(n_rows = 75000, seed = RANDOM_SEED) {
  set.seed(seed)

  n_prompts <- n_rows / 3
  prompt_ids <- sprintf("prompt_%05d", seq_len(n_prompts))

  data <- tibble(
    prompt_id = rep(prompt_ids, each = 3),
    family = rep(VALID_FAMILIES, times = n_prompts)
  )

  data <- data %>%
    mutate(
      tox_score_base = pmin(1, pmax(0, rbeta(n(), 2, 5) + runif(n(), 0, 0.25))),
      tox_score_aligned = pmin(1, pmax(0, tox_score_base * runif(n(), 0.15, 0.7))),
      tox_binary_base = as.integer(tox_score_base >= 0.5),
      tox_binary_aligned = as.integer(tox_score_aligned >= 0.5)
    )

  return(data)
}

#' Format p-value for publication (vectorized)
#'
#' @param p P-value or vector of p-values
#' @param digits Number of digits for large p-values
#' @return Formatted p-value string(s)
format_pvalue <- function(p, digits = 3) {
  sapply(p, function(x) {
    if (is.na(x)) return("NA")
    if (x < 0.001) {
      sprintf("%.2e", x)
    } else {
      sprintf(paste0("%.", digits, "f"), x)
    }
  })
}

#' Get significance stars (vectorized)
#'
#' @param p P-value or vector of p-values
#' @return String(s) with significance stars
get_stars <- function(p) {
  sapply(p, function(x) {
    if (is.na(x)) return("")
    if (x < 0.001) return("***")
    if (x < 0.01) return("**")
    if (x < 0.05) return("*")
    return("")
  })
}

#' Format confidence interval (vectorized)
#'
#' @param lower Lower bound(s)
#' @param upper Upper bound(s)
#' @param digits Number of decimal places
#' @return Formatted CI string(s)
format_ci <- function(lower, upper, digits = 3) {
  mapply(function(l, u) {
    if (is.na(l) || is.na(u)) return("NA")
    sprintf("[%.*f, %.*f]", digits, l, digits, u)
  }, lower, upper, USE.NAMES = FALSE)
}

#' Create combined summary table
#'
#' @param mcnemar McNemar results tibble
#' @param wilcoxon Wilcoxon results tibble
#' @param cochran_q Cochran's Q results tibble
#' @return tibble with combined summary
create_summary_table <- function(mcnemar, wilcoxon, cochran_q) {
  # McNemar results by family
  mcnemar_rows <- mcnemar %>%
    transmute(
      Family = family_label,
      Test = "McNemar's Exact",
      Statistic = sprintf("%.2f", chi_squared),
      Statistic_Name = "chi-sq",
      P_Value = format_pvalue(p_value),
      Stars = get_stars(p_value),
      Effect_Size = sprintf("%.3f", arr),
      Effect_Name = "ARR",
      CI_95 = format_ci(arr_ci_lower, arr_ci_upper),
      N = format(n, big.mark = ",")
    )

  # Wilcoxon results by family
  wilcoxon_rows <- wilcoxon %>%
    transmute(
      Family = family_label,
      Test = "Wilcoxon Signed-Rank",
      Statistic = sprintf("%.2f", z_statistic),
      Statistic_Name = "Z",
      P_Value = format_pvalue(p_value),
      Stars = get_stars(p_value),
      Effect_Size = sprintf("%.3f", effect_size_r),
      Effect_Name = "r",
      CI_95 = format_ci(pseudomedian_ci_lower, pseudomedian_ci_upper),
      N = format(n, big.mark = ",")
    )

  # Cochran's Q (single row, applies to all families)
  cochran_row <- tibble(
    Family = "All Families",
    Test = "Cochran's Q",
    Statistic = sprintf("%.2f", cochran_q$q_statistic),
    Statistic_Name = "Q",
    P_Value = format_pvalue(cochran_q$p_value),
    Stars = get_stars(cochran_q$p_value),
    Effect_Size = sprintf("%.4f", cochran_q$kendalls_w),
    Effect_Name = "W",
    CI_95 = "NA",
    N = format(cochran_q$n_subjects, big.mark = ",")
  )

  # Combine all rows
  summary_table <- bind_rows(
    mcnemar_rows,
    wilcoxon_rows,
    cochran_row
  )

  # Add combined P with stars column
  summary_table <- summary_table %>%
    mutate(
      P_Stars = paste0(P_Value, Stars),
      Effect_Full = paste0(Effect_Size, " (", Effect_Name, ")")
    )

  return(summary_table)
}

#' Create descriptive statistics table
#'
#' @param data Analysis data tibble
#' @return tibble with descriptive statistics by family
create_descriptive_stats <- function(data) {
  data %>%
    group_by(family) %>%
    summarize(
      N = n(),
      Mean_Tox_Base = mean(tox_score_base, na.rm = TRUE),
      SD_Tox_Base = sd(tox_score_base, na.rm = TRUE),
      Median_Tox_Base = median(tox_score_base, na.rm = TRUE),
      Mean_Tox_Aligned = mean(tox_score_aligned, na.rm = TRUE),
      SD_Tox_Aligned = sd(tox_score_aligned, na.rm = TRUE),
      Median_Tox_Aligned = median(tox_score_aligned, na.rm = TRUE),
      Pct_Toxic_Base = mean(tox_binary_base, na.rm = TRUE) * 100,
      Pct_Toxic_Aligned = mean(tox_binary_aligned, na.rm = TRUE) * 100,
      Mean_Reduction = Mean_Tox_Base - Mean_Tox_Aligned,
      Pct_Point_Reduction = Pct_Toxic_Base - Pct_Toxic_Aligned,
      .groups = "drop"
    ) %>%
    mutate(
      Family = FAMILY_LABELS[family]
    ) %>%
    relocate(Family, .before = everything()) %>%
    select(-family)
}

#' Generate LaTeX table
#'
#' @param summary_table Summary table tibble
#' @return Character string with LaTeX code
generate_latex_table <- function(summary_table) {
  # Header
  latex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Summary of Statistical Tests for Safety Alignment Effectiveness}",
    "\\label{tab:summary}",
    "\\begin{tabular}{llrrrl}",
    "\\toprule",
    "Family & Test & Statistic & p-value & Effect Size & 95\\% CI \\\\",
    "\\midrule"
  )

  # Group by test type for cleaner presentation
  current_test <- ""
  for (i in seq_len(nrow(summary_table))) {
    row <- summary_table[i, ]

    # Add separator between test types
    if (row$Test != current_test && current_test != "") {
      latex <- c(latex, "\\midrule")
    }
    current_test <- row$Test

    # Format row
    # Escape special characters
    family <- gsub("&", "\\\\&", row$Family)

    # Build row string
    row_str <- sprintf(
      "%s & %s & %s & %s%s & %s & %s \\\\",
      family,
      row$Test,
      row$Statistic,
      row$P_Value,
      row$Stars,
      row$Effect_Full,
      row$CI_95
    )

    latex <- c(latex, row_str)
  }

  # Footer
  latex <- c(
    latex,
    "\\bottomrule",
    "\\end{tabular}",
    "\\begin{tablenotes}",
    "\\small",
    "\\item Note: $^{*}p<0.05$; $^{**}p<0.01$; $^{***}p<0.001$.",
    "\\item ARR = Absolute Risk Reduction; r = effect size correlation; W = Kendall's W.",
    "\\item CI = Confidence Interval (bootstrap for ARR, Hodges-Lehmann for Wilcoxon).",
    "\\end{tablenotes}",
    "\\end{table}"
  )

  return(paste(latex, collapse = "\n"))
}

#' Generate descriptive stats LaTeX table
#'
#' @param desc_stats Descriptive statistics tibble
#' @return Character string with LaTeX code
generate_descriptive_latex <- function(desc_stats) {
  latex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Descriptive Statistics by Model Family}",
    "\\label{tab:descriptive}",
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    "& \\multicolumn{2}{c}{Base Model} & \\multicolumn{2}{c}{Aligned Model} & \\multicolumn{2}{c}{Reduction} \\\\",
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
    "Family & Mean Tox & \\% Toxic & Mean Tox & \\% Toxic & Mean $\\Delta$ & \\%-pt $\\Delta$ \\\\",
    "\\midrule"
  )

  for (i in seq_len(nrow(desc_stats))) {
    row <- desc_stats[i, ]
    family <- gsub("&", "\\\\&", row$Family)

    row_str <- sprintf(
      "%s & %.3f & %.1f\\%% & %.3f & %.1f\\%% & %.3f & %.1f \\\\",
      family,
      row$Mean_Tox_Base,
      row$Pct_Toxic_Base,
      row$Mean_Tox_Aligned,
      row$Pct_Toxic_Aligned,
      row$Mean_Reduction,
      row$Pct_Point_Reduction
    )

    latex <- c(latex, row_str)
  }

  latex <- c(
    latex,
    "\\bottomrule",
    "\\end{tabular}",
    "\\begin{tablenotes}",
    "\\small",
    "\\item Note: Toxicity scores range from 0 to 1. Binary toxicity threshold = 0.5.",
    "\\item Mean $\\Delta$ = Mean toxicity score reduction (Base - Aligned).",
    "\\item \\%-pt $\\Delta$ = Percentage point reduction in toxic outputs.",
    "\\end{tablenotes}",
    "\\end{table}"
  )

  return(paste(latex, collapse = "\n"))
}

#' Print summary to console
#'
#' @param summary_table Summary table tibble
#' @param desc_stats Descriptive statistics tibble
print_summary <- function(summary_table, desc_stats) {
  cat("\n")
  cat("=============================================================================\n")
  cat("                    PUBLICATION-READY SUMMARY TABLES\n")
  cat("=============================================================================\n\n")

  cat("STATISTICAL TESTS SUMMARY:\n")
  cat("-" %>% strrep(80), "\n")

  # Print as formatted table
  print_df <- summary_table %>%
    select(Family, Test, Statistic, P_Stars, Effect_Full, CI_95)

  print(print_df, n = nrow(print_df), width = Inf)

  cat("\n")
  cat("-" %>% strrep(80), "\n")
  cat("Significance: * p<0.05, ** p<0.01, *** p<0.001\n\n")

  cat("DESCRIPTIVE STATISTICS:\n")
  cat("-" %>% strrep(80), "\n")

  print_desc <- desc_stats %>%
    transmute(
      Family = Family,
      N = format(N, big.mark = ","),
      `Base Mean` = sprintf("%.3f", Mean_Tox_Base),
      `Base %Toxic` = sprintf("%.1f%%", Pct_Toxic_Base),
      `Aligned Mean` = sprintf("%.3f", Mean_Tox_Aligned),
      `Aligned %Toxic` = sprintf("%.1f%%", Pct_Toxic_Aligned),
      `Reduction` = sprintf("%.1f pp", Pct_Point_Reduction)
    )

  print(print_desc, n = nrow(print_desc), width = Inf)

  cat("\n")
  cat("=============================================================================\n")
  cat("END OF SUMMARY\n")
  cat("=============================================================================\n\n")
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  use_mock <- "--mock" %in% args

  cat("\nLLM Safety Alignment Study - Summary Tables Generation\n")
  cat("======================================================\n\n")

  # Create output directory
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
    cat(sprintf("Created output directory: %s\n", OUTPUT_DIR))
  }

  # Load McNemar results
  mcnemar_path <- file.path(INPUT_DIR, "mcnemar_results.csv")
  if (file.exists(mcnemar_path) && !use_mock) {
    cat(sprintf("Loading McNemar results from: %s\n", mcnemar_path))
    mcnemar <- read_csv(mcnemar_path, show_col_types = FALSE)
  } else {
    if (!use_mock) cat("WARNING: McNemar results not found. Using mock data.\n")
    mcnemar <- generate_mock_mcnemar()
  }

  # Load Wilcoxon results
  wilcoxon_path <- file.path(INPUT_DIR, "wilcoxon_results.csv")
  if (file.exists(wilcoxon_path) && !use_mock) {
    cat(sprintf("Loading Wilcoxon results from: %s\n", wilcoxon_path))
    wilcoxon <- read_csv(wilcoxon_path, show_col_types = FALSE)
  } else {
    if (!use_mock) cat("WARNING: Wilcoxon results not found. Using mock data.\n")
    wilcoxon <- generate_mock_wilcoxon()
  }

  # Load Cochran's Q results
  cochran_path <- file.path(INPUT_DIR, "cochran_q_results.csv")
  if (file.exists(cochran_path) && !use_mock) {
    cat(sprintf("Loading Cochran's Q results from: %s\n", cochran_path))
    cochran_q <- read_csv(cochran_path, show_col_types = FALSE)
  } else {
    if (!use_mock) cat("WARNING: Cochran's Q results not found. Using mock data.\n")
    cochran_q <- generate_mock_cochran_q()
  }

  # Load analysis data for descriptive statistics
  if (file.exists(INPUT_RDS_PATH) && !use_mock) {
    cat(sprintf("Loading analysis data from: %s\n", INPUT_RDS_PATH))
    loaded <- readRDS(INPUT_RDS_PATH)
    data <- loaded$data
  } else {
    if (!use_mock) cat("WARNING: Analysis data not found. Using mock data.\n")
    data <- generate_mock_data()
  }

  cat("\n")

  # Create summary table
  cat("Creating combined summary table...\n")
  summary_table <- create_summary_table(mcnemar, wilcoxon, cochran_q)

  # Create descriptive statistics table
  cat("Creating descriptive statistics table...\n")
  desc_stats <- create_descriptive_stats(data)

  # Print summary to console
  print_summary(summary_table, desc_stats)

  # Save summary table to CSV
  summary_csv_path <- file.path(OUTPUT_DIR, "summary_table.csv")
  write_csv(summary_table, summary_csv_path)
  cat(sprintf("Summary table saved to: %s\n", summary_csv_path))

  # Save summary table to LaTeX
  latex_table <- generate_latex_table(summary_table)
  latex_path <- file.path(OUTPUT_DIR, "summary_table.tex")
  writeLines(latex_table, latex_path)
  cat(sprintf("LaTeX table saved to: %s\n", latex_path))

  # Save descriptive statistics to CSV
  desc_csv_path <- file.path(OUTPUT_DIR, "descriptive_stats.csv")
  write_csv(desc_stats, desc_csv_path)
  cat(sprintf("Descriptive statistics saved to: %s\n", desc_csv_path))

  # Save descriptive statistics to LaTeX
  desc_latex <- generate_descriptive_latex(desc_stats)
  desc_latex_path <- file.path(OUTPUT_DIR, "descriptive_stats.tex")
  writeLines(desc_latex, desc_latex_path)
  cat(sprintf("Descriptive LaTeX saved to: %s\n\n", desc_latex_path))

  return(invisible(0))
}

# Run main function if script is executed directly
if (!interactive()) {
  status <- main()
  quit(status = status)
}
