# Install required R packages for LLM Safety Alignment Study

packages <- c(
  "tidyverse",    # Data manipulation and visualization
  "exact2x2",     # McNemar's exact test
  "coin",         # Permutation tests including Wilcoxon
  "DescTools",    # Cochran's Q test
  "patchwork",    # Combining ggplots
  "knitr",        # Report generation
  "rmarkdown",    # R Markdown documents
  "jsonlite",     # JSON handling
  "broom"         # Tidy model outputs
)

install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

invisible(sapply(packages, install_if_missing))

cat("All R packages installed successfully.\n")
