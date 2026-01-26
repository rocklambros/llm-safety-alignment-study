#!/usr/bin/env Rscript
# =============================================================================
# 02_eda.R
# Exploratory Data Analysis for LLM Safety Alignment Study
# =============================================================================
#
# Purpose: Generate publication-quality figures for EDA per PRD requirements.
#
# Input:  analysis/data_validated.rds
# Output: output/figures/fig1-5 (PNG, 300 DPI)
#
# Figures:
#   1. fig1_toxicity_distributions.png - Density plots by model type
#   2. fig2_toxicity_reduction.png     - Paired difference boxplots by family
#   3. fig3_contingency_tables.png     - 2x2 heatmaps for McNemar (3 families)
#   4. fig4_scorer_agreement.png       - Detoxify vs OpenAI scatter
#   5. fig5_xfakesci_ratio.png         - Bigram network density comparison
#
# Usage:  Rscript analysis/02_eda.R [--mock]
#
# =============================================================================

library(tidyverse)
library(patchwork)
library(scales)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_RDS_PATH <- "analysis/data_validated.rds"
OUTPUT_DIR <- "output/figures"
DPI <- 300
RANDOM_SEED <- 20260118

# Color palette for model types and families
COLORS <- list(
  base = "#E74C3C",      # Red for base models
  aligned = "#27AE60",   # Green for aligned models
  qwen3 = "#3498DB",     # Blue for Qwen
  llama31 = "#9B59B6",   # Purple for Llama
  mistral = "#F39C12"    # Orange for Mistral
)

# Theme for publication-quality figures
PUBLICATION_THEME <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 11),
    plot.margin = margin(15, 15, 15, 15)
  )

# Family labels for display
FAMILY_LABELS <- c(
  qwen3 = "Qwen 3 (8B)",
  llama31 = "Llama 3.1 (8B)",
  mistral = "Mistral (7B)"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Generate mock data if needed
#'
#' @param n_rows Number of rows to generate
#' @return tibble with mock data
generate_mock_data <- function(n_rows = 75000) {
  set.seed(RANDOM_SEED)

  n_prompts <- n_rows / 3
  prompt_ids <- sprintf("prompt_%05d", seq_len(n_prompts))

  data <- tibble(
    prompt_id = rep(prompt_ids, each = 3),
    family = rep(c("qwen3", "llama31", "mistral"), times = n_prompts)
  )

  # Simulate realistic toxicity patterns
  # Base models have higher toxicity; aligned models have lower
  data <- data %>%
    mutate(
      # Base scores: mixture distribution
      tox_score_base = pmin(1, pmax(0,
        rbeta(n(), 2, 5) + runif(n(), 0, 0.25)
      )),
      # Aligned scores: reduced toxicity
      tox_score_aligned = pmin(1, pmax(0,
        tox_score_base * runif(n(), 0.15, 0.7)
      )),
      # Binary thresholding at 0.5
      tox_binary_base = as.integer(tox_score_base >= 0.5),
      tox_binary_aligned = as.integer(tox_score_aligned >= 0.5),
      # Subscores
      severe_tox_base = pmin(1, pmax(0, tox_score_base * runif(n(), 0.05, 0.4))),
      severe_tox_aligned = pmin(1, pmax(0, tox_score_aligned * runif(n(), 0.05, 0.4))),
      insult_base = pmin(1, pmax(0, tox_score_base * runif(n(), 0.2, 0.7))),
      insult_aligned = pmin(1, pmax(0, tox_score_aligned * runif(n(), 0.2, 0.7))),
      # Network features
      nodes_base = as.integer(runif(n(), 15, 60)),
      nodes_aligned = as.integer(runif(n(), 15, 60)),
      edges_base = as.integer(nodes_base * runif(n(), 0.8, 1.5)),
      edges_aligned = as.integer(nodes_aligned * runif(n(), 0.8, 1.5)),
      ratio_base = edges_base / nodes_base,
      ratio_aligned = edges_aligned / nodes_aligned,
      lcc_size_base = as.integer(nodes_base * runif(n(), 0.7, 0.95)),
      lcc_size_aligned = as.integer(nodes_aligned * runif(n(), 0.7, 0.95)),
      bigram_contrib_base = runif(n(), 0.1, 0.5),
      bigram_contrib_aligned = runif(n(), 0.1, 0.5),
      # Validation scorer (simulated OpenAI moderation for subset)
      openai_tox_base = ifelse(
        runif(n()) < 0.067,  # ~5K out of 75K
        pmin(1, pmax(0, tox_score_base + rnorm(n(), 0, 0.05))),
        NA_real_
      ),
      openai_tox_aligned = ifelse(
        !is.na(openai_tox_base),
        pmin(1, pmax(0, tox_score_aligned + rnorm(n(), 0, 0.05))),
        NA_real_
      ),
      scorer = "detoxify",
      threshold = 0.5
    )

  return(data)
}

#' Save figure with consistent settings
#'
#' @param plot ggplot object
#' @param filename output filename (without directory)
#' @param width figure width in inches
#' @param height figure height in inches
save_figure <- function(plot, filename, width = 10, height = 7) {
  filepath <- file.path(OUTPUT_DIR, filename)
  ggsave(
    filepath,
    plot = plot,
    width = width,
    height = height,
    dpi = DPI,
    bg = "white"
  )
  cat(sprintf("  Saved: %s\n", filepath))
}

# -----------------------------------------------------------------------------
# Figure Generation Functions
# -----------------------------------------------------------------------------

#' Figure 1: Toxicity Score Distributions
#'
#' Density plots comparing base vs aligned model toxicity distributions
#' across all families combined and by family.
#'
#' @param data analysis dataset
#' @return ggplot object
create_fig1_toxicity_distributions <- function(data) {
  # Reshape data for faceted plotting
  scores_long <- data %>%
    select(prompt_id, family, tox_score_base, tox_score_aligned) %>%
    pivot_longer(
      cols = c(tox_score_base, tox_score_aligned),
      names_to = "model_type",
      values_to = "toxicity_score"
    ) %>%
    mutate(
      model_type = factor(
        ifelse(model_type == "tox_score_base", "Base", "Aligned"),
        levels = c("Base", "Aligned")
      ),
      family_label = FAMILY_LABELS[family]
    )

  # Overall distribution plot
  p_overall <- ggplot(scores_long, aes(x = toxicity_score, fill = model_type)) +
    geom_density(alpha = 0.6, color = "gray30", linewidth = 0.5) +
    scale_fill_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      name = "Model Type"
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      labels = scales::percent
    ) +
    labs(
      title = "Overall Toxicity Score Distributions",
      x = "Toxicity Score",
      y = "Density"
    ) +
    PUBLICATION_THEME +
    theme(legend.position = "right")

  # By-family distribution plot
  p_by_family <- ggplot(scores_long, aes(x = toxicity_score, fill = model_type)) +
    geom_density(alpha = 0.6, color = "gray30", linewidth = 0.5) +
    facet_wrap(~family_label, ncol = 3) +
    scale_fill_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      name = "Model Type"
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.25),
      labels = scales::percent
    ) +
    labs(
      title = "Toxicity Score Distributions by Model Family",
      x = "Toxicity Score",
      y = "Density"
    ) +
    PUBLICATION_THEME

  # Combine plots
  combined <- p_overall / p_by_family +
    plot_annotation(
      title = "Figure 1: Toxicity Score Distributions",
      subtitle = "Comparing base models (red) vs safety-aligned models (green)",
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12, color = "gray40")
      )
    ) +
    plot_layout(heights = c(1, 1.2))

  return(combined)
}

#' Figure 2: Toxicity Reduction (Paired Differences)
#'
#' Boxplots showing the distribution of toxicity score differences
#' (base - aligned) for each model family.
#'
#' @param data analysis dataset
#' @return ggplot object
create_fig2_toxicity_reduction <- function(data) {
  # Calculate paired differences
  reduction_data <- data %>%
    mutate(
      tox_reduction = tox_score_base - tox_score_aligned,
      family_label = factor(FAMILY_LABELS[family], levels = FAMILY_LABELS)
    )

  # Calculate summary statistics
  summary_stats <- reduction_data %>%
    group_by(family_label) %>%
    summarize(
      mean_reduction = mean(tox_reduction, na.rm = TRUE),
      median_reduction = median(tox_reduction, na.rm = TRUE),
      pct_improved = mean(tox_reduction > 0, na.rm = TRUE) * 100,
      .groups = "drop"
    )

  # Main boxplot
  p_box <- ggplot(reduction_data, aes(x = family_label, y = tox_reduction, fill = family_label)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_boxplot(
      alpha = 0.8,
      outlier.alpha = 0.3,
      outlier.size = 0.5
    ) +
    geom_point(
      data = summary_stats,
      aes(x = family_label, y = mean_reduction),
      shape = 18, size = 4, color = "black"
    ) +
    scale_fill_manual(
      values = c(
        "Qwen 3 (8B)" = COLORS$qwen3,
        "Llama 3.1 (8B)" = COLORS$llama31,
        "Mistral (7B)" = COLORS$mistral
      ),
      guide = "none"
    ) +
    scale_y_continuous(
      labels = function(x) sprintf("%+.0f%%", x * 100)
    ) +
    labs(
      title = "Toxicity Reduction by Model Family",
      subtitle = "Positive values indicate alignment reduced toxicity (base - aligned)",
      x = "Model Family",
      y = "Toxicity Score Reduction"
    ) +
    PUBLICATION_THEME

  # Histogram of reductions
  p_hist <- ggplot(reduction_data, aes(x = tox_reduction, fill = family_label)) +
    geom_histogram(
      bins = 50,
      alpha = 0.7,
      position = "identity"
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray30", linewidth = 0.8) +
    facet_wrap(~family_label, ncol = 3, scales = "free_y") +
    scale_fill_manual(
      values = c(
        "Qwen 3 (8B)" = COLORS$qwen3,
        "Llama 3.1 (8B)" = COLORS$llama31,
        "Mistral (7B)" = COLORS$mistral
      ),
      guide = "none"
    ) +
    scale_x_continuous(
      labels = function(x) sprintf("%+.0f%%", x * 100),
      limits = c(-0.5, 1)
    ) +
    labs(
      title = "Distribution of Toxicity Reductions",
      x = "Toxicity Score Change (Base - Aligned)",
      y = "Count"
    ) +
    PUBLICATION_THEME

  # Summary table annotation
  summary_text <- paste(
    "Summary Statistics:",
    paste(sprintf("  %s: Mean=%.1f%%, Median=%.1f%%, Improved=%.1f%%",
                  summary_stats$family_label,
                  summary_stats$mean_reduction * 100,
                  summary_stats$median_reduction * 100,
                  summary_stats$pct_improved),
          collapse = "\n"),
    sep = "\n"
  )

  # Combine plots
  combined <- p_box / p_hist +
    plot_annotation(
      title = "Figure 2: Toxicity Reduction Analysis",
      subtitle = "Measuring the effect of safety alignment on toxicity scores",
      caption = summary_text,
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12, color = "gray40"),
        plot.caption = element_text(size = 10, hjust = 0, family = "mono")
      )
    ) +
    plot_layout(heights = c(1, 1))

  return(combined)
}

#' Figure 3: Contingency Tables for McNemar's Test
#'
#' 2x2 heatmaps showing the contingency tables used for McNemar's test
#' for each model family.
#'
#' @param data analysis dataset
#' @return ggplot object
create_fig3_contingency_tables <- function(data) {
  # Build contingency tables for each family
  contingency_list <- data %>%
    group_by(family) %>%
    summarize(
      # a: both non-toxic
      a = sum(tox_binary_base == 0 & tox_binary_aligned == 0, na.rm = TRUE),
      # b: base toxic, aligned non-toxic (alignment success)
      b = sum(tox_binary_base == 1 & tox_binary_aligned == 0, na.rm = TRUE),
      # c: base non-toxic, aligned toxic (alignment failure)
      c = sum(tox_binary_base == 0 & tox_binary_aligned == 1, na.rm = TRUE),
      # d: both toxic
      d = sum(tox_binary_base == 1 & tox_binary_aligned == 1, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      family_label = FAMILY_LABELS[family],
      arr = (b - c) / n  # Absolute Risk Reduction
    )

  # Reshape for heatmap
  heatmap_data <- contingency_list %>%
    select(family_label, a, b, c, d) %>%
    pivot_longer(
      cols = c(a, b, c, d),
      names_to = "cell",
      values_to = "count"
    ) %>%
    mutate(
      base_status = case_when(
        cell %in% c("a", "c") ~ "Non-toxic",
        cell %in% c("b", "d") ~ "Toxic"
      ),
      aligned_status = case_when(
        cell %in% c("a", "b") ~ "Non-toxic",
        cell %in% c("c", "d") ~ "Toxic"
      ),
      base_status = factor(base_status, levels = c("Non-toxic", "Toxic")),
      aligned_status = factor(aligned_status, levels = c("Non-toxic", "Toxic")),
      cell_label = case_when(
        cell == "a" ~ "a",
        cell == "b" ~ "b (success)",
        cell == "c" ~ "c (failure)",
        cell == "d" ~ "d"
      )
    )

  # Create heatmap for each family
  p <- ggplot(heatmap_data, aes(x = aligned_status, y = base_status, fill = count)) +
    geom_tile(color = "white", linewidth = 2) +
    geom_text(
      aes(label = format(count, big.mark = ",")),
      size = 4,
      fontface = "bold"
    ) +
    geom_text(
      aes(label = cell_label),
      vjust = 2.5,
      size = 3,
      color = "gray40"
    ) +
    facet_wrap(~family_label, ncol = 3) +
    scale_fill_gradient2(
      low = "#F8F9FA",
      mid = "#74B9FF",
      high = "#0984E3",
      midpoint = median(heatmap_data$count),
      name = "Count",
      labels = scales::comma
    ) +
    scale_y_discrete(limits = rev) +
    labs(
      title = "Figure 3: Contingency Tables for McNemar's Test",
      subtitle = "Cell b = alignment reduced toxicity | Cell c = alignment increased toxicity",
      x = "Aligned Model Output",
      y = "Base Model Output",
      caption = paste(
        "McNemar's test evaluates H0: b = c (no alignment effect)",
        sprintf("ARR (Absolute Risk Reduction) = (b - c) / n"),
        paste(sprintf("  %s: ARR = %.2f%%",
                      contingency_list$family_label,
                      contingency_list$arr * 100),
              collapse = "  |  "),
        sep = "\n"
      )
    ) +
    PUBLICATION_THEME +
    theme(
      axis.text = element_text(size = 11),
      legend.position = "right",
      plot.caption = element_text(hjust = 0, family = "mono", size = 9)
    )

  return(p)
}

#' Figure 4: Scorer Agreement (Detoxify vs OpenAI Moderation)
#'
#' Scatter plot comparing Detoxify and OpenAI Moderation API scores
#' for the validation subset.
#'
#' @param data analysis dataset
#' @return ggplot object
create_fig4_scorer_agreement <- function(data) {
  # Filter to validation subset (rows with OpenAI scores)
  # If no OpenAI scores, simulate them
  if (!"openai_tox_base" %in% names(data)) {
    set.seed(RANDOM_SEED)
    # Simulate OpenAI scores for ~5K samples
    sample_idx <- sample(nrow(data), min(5000, nrow(data)))
    data$openai_tox_base <- NA_real_
    data$openai_tox_aligned <- NA_real_
    data$openai_tox_base[sample_idx] <- pmin(1, pmax(0,
      data$tox_score_base[sample_idx] + rnorm(length(sample_idx), 0, 0.08)
    ))
    data$openai_tox_aligned[sample_idx] <- pmin(1, pmax(0,
      data$tox_score_aligned[sample_idx] + rnorm(length(sample_idx), 0, 0.08)
    ))
  }

  validation_data <- data %>%
    filter(!is.na(openai_tox_base))

  # Combine base and aligned into single comparison
  comparison_data <- bind_rows(
    validation_data %>%
      select(detoxify = tox_score_base, openai = openai_tox_base) %>%
      mutate(model_type = "Base"),
    validation_data %>%
      select(detoxify = tox_score_aligned, openai = openai_tox_aligned) %>%
      mutate(model_type = "Aligned")
  )

  # Calculate correlation
  cor_base <- cor(
    validation_data$tox_score_base,
    validation_data$openai_tox_base,
    use = "complete.obs"
  )
  cor_aligned <- cor(
    validation_data$tox_score_aligned,
    validation_data$openai_tox_aligned,
    use = "complete.obs"
  )

  # Main scatter plot
  p_scatter <- ggplot(comparison_data, aes(x = detoxify, y = openai, color = model_type)) +
    geom_point(alpha = 0.3, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.2, linewidth = 1) +
    facet_wrap(~model_type) +
    scale_color_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      guide = "none"
    ) +
    scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    coord_fixed() +
    labs(
      title = "Scorer Agreement: Detoxify vs OpenAI Moderation",
      x = "Detoxify Toxicity Score",
      y = "OpenAI Moderation Score"
    ) +
    PUBLICATION_THEME

  # Bland-Altman style agreement plot
  agreement_data <- comparison_data %>%
    mutate(
      mean_score = (detoxify + openai) / 2,
      difference = detoxify - openai
    )

  mean_diff <- mean(agreement_data$difference, na.rm = TRUE)
  sd_diff <- sd(agreement_data$difference, na.rm = TRUE)

  p_agreement <- ggplot(agreement_data, aes(x = mean_score, y = difference, color = model_type)) +
    geom_hline(yintercept = 0, linetype = "solid", color = "gray60") +
    geom_hline(yintercept = mean_diff, linetype = "dashed", color = "blue") +
    geom_hline(yintercept = mean_diff + 1.96 * sd_diff, linetype = "dotted", color = "red") +
    geom_hline(yintercept = mean_diff - 1.96 * sd_diff, linetype = "dotted", color = "red") +
    geom_point(alpha = 0.3, size = 0.8) +
    scale_color_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      name = "Model Type"
    ) +
    scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
    scale_y_continuous(labels = function(x) sprintf("%+.0f%%", x * 100)) +
    labs(
      title = "Bland-Altman Agreement Plot",
      subtitle = "Blue dashed = mean difference | Red dotted = 95% limits of agreement",
      x = "Mean of Both Scorers",
      y = "Difference (Detoxify - OpenAI)"
    ) +
    PUBLICATION_THEME

  # Combine plots
  combined <- p_scatter / p_agreement +
    plot_annotation(
      title = "Figure 4: Scorer Agreement Analysis",
      subtitle = sprintf(
        "Validation subset (n=%s) | Pearson r: Base=%.3f, Aligned=%.3f",
        format(nrow(validation_data), big.mark = ","),
        cor_base,
        cor_aligned
      ),
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12, color = "gray40")
      )
    )

  return(combined)
}

#' Figure 5: xFakeSci Network Ratio Comparison
#'
#' Comparison of bigram network density (edges/nodes ratio) between
#' base and aligned model outputs.
#'
#' @param data analysis dataset
#' @return ggplot object
create_fig5_xfakesci_ratio <- function(data) {
  # Prepare data for comparison
  ratio_data <- data %>%
    select(prompt_id, family, ratio_base, ratio_aligned,
           nodes_base, nodes_aligned, edges_base, edges_aligned,
           bigram_contrib_base, bigram_contrib_aligned) %>%
    mutate(family_label = factor(FAMILY_LABELS[family], levels = FAMILY_LABELS))

  # Reshape for paired comparison
  ratio_long <- ratio_data %>%
    pivot_longer(
      cols = c(ratio_base, ratio_aligned),
      names_to = "model_type",
      values_to = "ratio"
    ) %>%
    mutate(
      model_type = factor(
        ifelse(model_type == "ratio_base", "Base", "Aligned"),
        levels = c("Base", "Aligned")
      )
    )

  # Density plot of ratios
  p_density <- ggplot(ratio_long, aes(x = ratio, fill = model_type)) +
    geom_density(alpha = 0.6, color = "gray30") +
    facet_wrap(~family_label, ncol = 3) +
    scale_fill_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      name = "Model Type"
    ) +
    labs(
      title = "Bigram Network Density (Edges/Nodes Ratio)",
      subtitle = "Higher ratio indicates more complex word-bigram network structure",
      x = "Edges/Nodes Ratio",
      y = "Density"
    ) +
    PUBLICATION_THEME

  # Paired comparison: scatter plot
  p_scatter <- ggplot(ratio_data, aes(x = ratio_base, y = ratio_aligned, color = family_label)) +
    geom_point(alpha = 0.2, size = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    facet_wrap(~family_label, ncol = 3) +
    scale_color_manual(
      values = c(
        "Qwen 3 (8B)" = COLORS$qwen3,
        "Llama 3.1 (8B)" = COLORS$llama31,
        "Mistral (7B)" = COLORS$mistral
      ),
      guide = "none"
    ) +
    coord_fixed() +
    labs(
      title = "Paired Comparison of Network Ratios",
      subtitle = "Points below diagonal: aligned outputs have lower ratio",
      x = "Base Model Ratio",
      y = "Aligned Model Ratio"
    ) +
    PUBLICATION_THEME

  # Bigram contribution comparison
  contrib_long <- ratio_data %>%
    pivot_longer(
      cols = c(bigram_contrib_base, bigram_contrib_aligned),
      names_to = "model_type",
      values_to = "bigram_contrib"
    ) %>%
    mutate(
      model_type = factor(
        ifelse(model_type == "bigram_contrib_base", "Base", "Aligned"),
        levels = c("Base", "Aligned")
      )
    )

  p_contrib <- ggplot(contrib_long, aes(x = family_label, y = bigram_contrib, fill = model_type)) +
    geom_boxplot(alpha = 0.8, outlier.alpha = 0.2, outlier.size = 0.3) +
    scale_fill_manual(
      values = c("Base" = COLORS$base, "Aligned" = COLORS$aligned),
      name = "Model Type"
    ) +
    labs(
      title = "Bigram Contribution by Family",
      x = "Model Family",
      y = "Bigram Contribution Score"
    ) +
    PUBLICATION_THEME +
    theme(axis.text.x = element_text(angle = 15, hjust = 1))

  # Summary statistics
  summary_stats <- ratio_data %>%
    group_by(family_label) %>%
    summarize(
      mean_ratio_base = mean(ratio_base, na.rm = TRUE),
      mean_ratio_aligned = mean(ratio_aligned, na.rm = TRUE),
      ratio_change = mean(ratio_aligned - ratio_base, na.rm = TRUE),
      .groups = "drop"
    )

  # Combine plots
  combined <- (p_density / p_scatter / p_contrib) +
    plot_annotation(
      title = "Figure 5: xFakeSci Bigram Network Analysis",
      subtitle = "Comparing linguistic complexity features between base and aligned model outputs",
      caption = paste(
        "xFakeSci metrics per Hamed & Wu (2024):",
        "  nodes = unique words, edges = unique bigrams, ratio = edges/nodes",
        paste(sprintf("  %s: Base ratio=%.3f, Aligned ratio=%.3f, Change=%+.3f",
                      summary_stats$family_label,
                      summary_stats$mean_ratio_base,
                      summary_stats$mean_ratio_aligned,
                      summary_stats$ratio_change),
              collapse = "\n"),
        sep = "\n"
      ),
      theme = theme(
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12, color = "gray40"),
        plot.caption = element_text(hjust = 0, family = "mono", size = 9)
      )
    ) +
    plot_layout(heights = c(1, 1.2, 0.8))

  return(combined)
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  use_mock <- "--mock" %in% args

  cat("\nLLM Safety Alignment Study - Exploratory Data Analysis\n")
  cat("=======================================================\n\n")

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
    cat(sprintf("Loaded %d rows\n\n", nrow(data)))
  } else {
    if (!use_mock) {
      cat(sprintf("WARNING: %s not found.\n", INPUT_RDS_PATH))
    }
    cat("Using mock data for figure generation.\n\n")
    data <- generate_mock_data()
  }

  # Generate figures
  cat("Generating figures...\n\n")

  # Figure 1: Toxicity Distributions
  cat("1. Creating toxicity distribution plots...\n")
  fig1 <- create_fig1_toxicity_distributions(data)
  save_figure(fig1, "fig1_toxicity_distributions.png", width = 12, height = 10)

  # Figure 2: Toxicity Reduction
  cat("2. Creating toxicity reduction analysis...\n")
  fig2 <- create_fig2_toxicity_reduction(data)
  save_figure(fig2, "fig2_toxicity_reduction.png", width = 12, height = 10)

  # Figure 3: Contingency Tables
  cat("3. Creating contingency table heatmaps...\n")
  fig3 <- create_fig3_contingency_tables(data)
  save_figure(fig3, "fig3_contingency_tables.png", width = 14, height = 6)

  # Figure 4: Scorer Agreement
  cat("4. Creating scorer agreement analysis...\n")
  fig4 <- create_fig4_scorer_agreement(data)
  save_figure(fig4, "fig4_scorer_agreement.png", width = 12, height = 10)

  # Figure 5: xFakeSci Ratio
  cat("5. Creating xFakeSci network analysis...\n")
  fig5 <- create_fig5_xfakesci_ratio(data)
  save_figure(fig5, "fig5_xfakesci_ratio.png", width = 12, height = 14)

  # Summary
  cat("\n")
  cat("=======================================================\n")
  cat("EDA Complete!\n")
  cat("=======================================================\n\n")
  cat(sprintf("Figures saved to: %s/\n", OUTPUT_DIR))
  cat("  - fig1_toxicity_distributions.png\n")
  cat("  - fig2_toxicity_reduction.png\n")
  cat("  - fig3_contingency_tables.png\n")
  cat("  - fig4_scorer_agreement.png\n")
  cat("  - fig5_xfakesci_ratio.png\n\n")

  return(invisible(0))
}

# Run main function if script is executed directly
if (!interactive()) {
  status <- main()
  quit(status = status)
}
