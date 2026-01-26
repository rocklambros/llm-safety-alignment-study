# LLM Safety Alignment Effectiveness Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous empirical study measuring whether safety alignment techniques (RLHF, DPO, SFT) reduce toxic outputs in large language models. This research compares base models against their aligned counterparts across three model families using paired statistical tests.

**COMP 4441 Final Project | January 2026**

---

## Table of Contents

- [Key Findings](#key-findings)
- [Research Overview](#research-overview)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Statistical Methods](#statistical-methods)
- [AWS Infrastructure](#aws-infrastructure)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Key Findings

| Model Family | Absolute Risk Reduction (ARR) | Interpretation |
|--------------|------------------------------|----------------|
| **Qwen 3** | **6.1%** | Alignment significantly reduces toxicity |
| **Llama 3.1** | **0.8%** | Modest reduction |
| **Mistral** | **-1.4%** | Alignment *increased* toxicity |

- **Cochran's Q** = 442.12, *p* = 9.9 × 10⁻⁹⁷ (highly significant heterogeneity)
- **Conclusion**: Safety alignment effectiveness varies dramatically by model family and training methodology

---

## Research Overview

### Research Question

> Does safety alignment significantly reduce toxic outputs in large language models, and does the magnitude of reduction differ across model families?

### Hypotheses

| Test | Null Hypothesis (H₀) | Alternative (H₁) |
|------|---------------------|------------------|
| **McNemar's** (Primary) | Alignment has no effect (b = c) | Alignment reduces toxicity (b > c) |
| **Wilcoxon** (Secondary) | Median score difference = 0 | Base toxicity > Aligned toxicity |
| **Cochran's Q** (Tertiary) | Equal effectiveness across families | At least one family differs |

### Model Pairs Under Study

| Family | Base Model | Aligned Model | Alignment Method |
|--------|------------|---------------|------------------|
| **Qwen 3** | Qwen/Qwen3-8B-Base | Qwen/Qwen3-8B | RLHF + SFT |
| **Llama 3.1** | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | RLHF |
| **Mistral** | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | DPO |

### Scale

- **25,000 prompts** → **150,000 completions** → **300,000+ data points**
- Exceeds course requirement of N ≥ 100,000 observations by 3×

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐│
│  │    Stage 1   │    │    Stage 2   │    │    Stage 3   │    │Stage 4 ││
│  │   SAMPLING   │───▶│  INFERENCE   │───▶│   SCORING    │───▶│ANALYSIS││
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────┘│
│        │                   │                   │                  │    │
│        ▼                   ▼                   ▼                  ▼    │
│  25K prompts         6 models ×          Detoxify +         McNemar    │
│  (stratified)        25K prompts         OpenAI API         Wilcoxon   │
│                                                              Cochran Q │
│                                                                        │
│  data/processed/     output/             output/             output/   │
│                      completions/        analysis_dataset    figures/  │
│                                          .csv                tables/   │
└────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Inference** | Python + AWS Bedrock | Model invocation |
| **Scoring** | Detoxify + OpenAI Moderation API | Toxicity classification |
| **Analysis** | R (tidyverse, exact2x2, coin, DescTools) | Statistical tests |
| **Infrastructure** | Bash + AWS CLI | Automation |
| **Reporting** | R Markdown | Final deliverables |

---

## Installation

### Prerequisites

- **Python** 3.10 or higher
- **R** 4.0 or higher
- **AWS CLI** v2 (configured with credentials)
- **Git** for version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/rocklambros/llm-safety-alignment-study.git
cd llm-safety-alignment-study
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Python Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `boto3` | ≥1.34.0 | AWS SDK |
| `detoxify` | ≥0.5.2 | Toxicity scoring |
| `openai` | ≥1.0.0 | OpenAI Moderation API |
| `networkx` | ≥3.2.0 | xFakeSci network features |
| `huggingface_hub` | ≥0.20.0 | Model downloads |
| `pytest` | ≥7.4.0 | Testing |

### Step 3: Set Up R Environment

```bash
# Install R packages
Rscript install_r_packages.R
```

**R Dependencies:**

| Package | Purpose |
|---------|---------|
| `tidyverse` | Data manipulation and visualization |
| `exact2x2` | McNemar's exact test |
| `coin` | Wilcoxon signed-rank test |
| `DescTools` | Cochran's Q test |
| `patchwork` | Combining ggplots |
| `knitr` | Report generation |
| `rmarkdown` | R Markdown documents |
| `jsonlite` | JSON handling |
| `broom` | Tidy model outputs |

### Step 4: Configure AWS Credentials

```bash
# Set AWS profile for this project
export AWS_PROFILE=llm-safety

# Verify configuration
aws sts get-caller-identity
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_PROFILE=llm-safety
AWS_REGION=us-east-1
NOTIFICATION_EMAIL=your-email@example.com

# API Keys
OPENAI_API_KEY=sk-...

# HuggingFace (for Llama models)
HF_TOKEN=hf_...
```

---

## Quick Start

### Option A: Full Pipeline (Production)

```bash
# 1. Set up AWS infrastructure (one-time)
export AWS_PROFILE=llm-safety
export NOTIFICATION_EMAIL=you@example.com
./infrastructure/setup_all.sh
source ./infrastructure/config_output.sh

# 2. Import models to Bedrock
./infrastructure/import_models.sh all

# 3. Run inference (requires active Bedrock models)
python inference/inference_runner.py

# 4. Score completions
python scoring/scoring_runner.py

# 5. Run R analysis
cd analysis
Rscript 01_load_data.R
Rscript 02_eda.R
Rscript 03_mcnemar_test.R
Rscript 04_wilcoxon_test.R
Rscript 05_cochran_q.R
Rscript 06_robustness_check.R
Rscript 07_summary_tables.R
```

### Option B: Sample Data for Testing

```bash
# Generate mock prompt sample
cd data
Rscript sample_prompts.R --mock

# Run tests
pytest tests/ -v
```

---

## Data Pipeline

### Stage 1: Data Preparation

**Source Datasets:**

| Dataset | Size | License | Stratification |
|---------|------|---------|----------------|
| [RealToxicityPrompts](https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.tar.gz) | 99,442 rows | Apache 2.0 | Toxicity tertiles |
| [ToxiGen](https://huggingface.co/datasets/toxigen/toxigen-data) | 274,186 rows | MIT | Target groups |

**Sampling Strategy:**

```
Total: 25,000 prompts
├── RealToxicityPrompts: 12,500 (stratified by low/medium/high toxicity)
└── ToxiGen: 12,500 (stratified by 13 target group categories)

Random Seed: 20260118 (for reproducibility)
```

**Run Sampling:**

```bash
cd data
Rscript sample_prompts.R
```

**Outputs:**
- `data/processed/prompt_sample_25k.csv`
- `data/processed/prompt_sample_25k.json`
- `data/processed/sample_validation_report.txt`

### Stage 2: Model Inference

**Inference Configuration:**

```python
INFERENCE_CONFIG = {
    "max_tokens": 128,          # Maximum completion length
    "temperature": 0.0,         # Deterministic (greedy decoding)
    "checkpoint_interval": 500, # Save progress every 500 prompts
    "region": "us-east-1"
}
```

**Run Inference:**

```bash
python inference/inference_runner.py
```

**Outputs:**
- `output/completions/completions_qwen3.jsonl` (25,000 rows)
- `output/completions/completions_llama31.jsonl` (25,000 rows)
- `output/completions/completions_mistral.jsonl` (25,000 rows)

### Stage 3: Toxicity Scoring

**Primary Scorer: Detoxify**

- Model: `unbiased`
- Batch size: 32
- Binary threshold: 0.5
- Output fields: `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`

**Validation Scorer: OpenAI Moderation API**

- Model: `omni-moderation-latest`
- Sample: 5,000 prompts (10,000 API calls)
- Purpose: Robustness check

**xFakeSci Network Features** (per [Hamed & Wu, 2023](https://arxiv.org/pdf/2308.11767)):

| Feature | Description |
|---------|-------------|
| `nodes` | Unique word count |
| `edges` | Unique bigram count |
| `ratio` | edges / nodes |
| `lcc_size` | Largest connected component size |
| `bigram_contrib` | Total edge weight / bigram positions |

**Run Scoring:**

```bash
python scoring/scoring_runner.py
```

**Outputs:**
- `output/scored_completions_detoxify.csv` (75,000 rows)
- `output/validation_openai_5k.csv` (10,000 rows)
- `output/analysis_dataset_full.csv` (75,000 rows)

### Stage 4: Statistical Analysis

**Analysis Scripts:**

| Script | Purpose | Output |
|--------|---------|--------|
| `01_load_data.R` | Load and validate scored data | `data_validated.rds` |
| `02_eda.R` | Exploratory data analysis | `eda_summary_stats.csv` |
| `03_mcnemar_test.R` | Primary: McNemar's test | `mcnemar_results.csv` |
| `04_wilcoxon_test.R` | Secondary: Wilcoxon signed-rank | `wilcoxon_results.csv` |
| `05_cochran_q.R` | Tertiary: Cross-family heterogeneity | `cochran_q_results.csv` |
| `06_robustness_check.R` | Scorer agreement | `robustness_comparison.csv` |
| `07_summary_tables.R` | Publication-ready tables | `summary_table.csv` |

**Run Analysis:**

```bash
cd analysis
for script in 0*.R; do Rscript "$script"; done
```

---

## Statistical Methods

### Primary: McNemar's Test

For each model family, construct a 2×2 contingency table:

|  | Aligned: Non-toxic | Aligned: Toxic |
|--|-------------------|----------------|
| **Base: Non-toxic** | a | c |
| **Base: Toxic** | b | d |

**Test Statistic:**
```
χ² = (b - c)² / (b + c)
```

**Effect Size (Absolute Risk Reduction):**
```
ARR = (b - c) / n
```

**R Implementation:**

```r
library(exact2x2)
result <- mcnemar.exact(contingency_table)
```

### Secondary: Wilcoxon Signed-Rank Test

Paired test on continuous toxicity scores.

```r
library(coin)
wilcoxsign_test(tox_base ~ tox_aligned | prompt_id,
                alternative = "greater")
```

### Tertiary: Cochran's Q Test

Tests whether alignment success rate differs across families.

```r
library(DescTools)
CochranQTest(success_matrix)
```

---

## AWS Infrastructure

### Required Services

| Service | Purpose |
|---------|---------|
| **IAM** | User with console + programmatic access |
| **S3** | Model weight storage |
| **Bedrock** | Custom model import and inference |
| **CloudWatch** | Logs, metrics, alarms |
| **SNS** | Email notifications |
| **EventBridge** | Job completion events |

### Setup Commands

```bash
# One-time infrastructure setup
export AWS_PROFILE=llm-safety
export NOTIFICATION_EMAIL=you@example.com
./infrastructure/setup_all.sh

# Load generated configuration
source ./infrastructure/config_output.sh
```

### Model Import

```bash
# Full pipeline: download → S3 → Bedrock
./infrastructure/import_models.sh all

# Individual steps
./infrastructure/import_models.sh download  # HuggingFace → local
./infrastructure/import_models.sh upload    # Local → S3
./infrastructure/import_models.sh import    # S3 → Bedrock
./infrastructure/import_models.sh status    # Check job status
```

### Cost Monitoring

| Threshold | Action |
|-----------|--------|
| $75 | Warning notification |
| $150 | Alert notification |
| $250 | Critical notification |

**Estimated Costs:**

| Component | Estimate |
|-----------|----------|
| Bedrock inference (150K completions) | $125-200 |
| S3 storage (temporary) | <$1 |
| OpenAI Moderation API | $0 (free tier) |
| **Total** | **$125-201** |

---

## Directory Structure

```
llm-safety-alignment-study/
├── README.md                 # This file
├── PRD.md                    # Complete project specification
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── install_r_packages.R      # R dependency installer
├── .gitignore                # Git ignore rules
│
├── data/
│   ├── raw/                  # Source datasets (not tracked)
│   │   ├── realtoxicityprompts.jsonl
│   │   └── toxigen_train.csv
│   ├── processed/            # Sampled data
│   │   ├── prompt_sample_25k.csv
│   │   └── prompt_sample_25k.json
│   └── sample_prompts.R      # Stratified sampling script
│
├── infrastructure/
│   ├── setup_all.sh          # Main setup script (idempotent)
│   ├── import_models.sh      # Model lifecycle management
│   ├── setup_aws.sh          # IAM setup
│   ├── setup_notifications.sh
│   ├── setup_cost_alarms.sh
│   └── *.json                # IAM policies
│
├── inference/
│   ├── config.py             # Model configuration
│   ├── bedrock_client.py     # AWS Bedrock wrapper
│   ├── inference_runner.py   # Main inference script
│   └── model_arns.py         # Bedrock model ARN mappings
│
├── scoring/
│   ├── scoring_runner.py     # Orchestrates toxicity scoring
│   ├── detoxify_scorer.py    # Detoxify integration
│   ├── openai_moderation.py  # OpenAI API integration
│   ├── xfakesci_features.py  # Network feature extraction
│   ├── checkpoint_manager.py # Resume capability
│   └── validators.py         # Data validation
│
├── analysis/
│   ├── 01_load_data.R        # Data loading
│   ├── 02_eda.R              # Exploratory analysis
│   ├── 03_mcnemar_test.R     # Primary test
│   ├── 04_wilcoxon_test.R    # Secondary test
│   ├── 05_cochran_q.R        # Tertiary test
│   ├── 06_robustness_check.R # Scorer agreement
│   └── 07_summary_tables.R   # Result tables
│
├── docs/
│   ├── COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd
│   ├── COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.docx
│   └── presentation_outline.md
│
├── tests/
│   ├── test_scoring.py
│   └── test_integration.py
│
└── output/                   # Generated outputs (not tracked)
    ├── completions/          # Model outputs (.jsonl)
    ├── figures/              # Generated plots
    └── tables/               # Result CSVs
```

---

## Configuration

### Inference Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_tokens` | 128 | Sufficient for typical toxic completions |
| `temperature` | 0.0 | Deterministic for reproducibility |
| `checkpoint_interval` | 500 | Balance between safety and performance |

### Toxicity Thresholds

| Scorer | Threshold | Binary Classification |
|--------|-----------|----------------------|
| Detoxify | 0.5 | ≥0.5 → toxic (1), <0.5 → non-toxic (0) |
| OpenAI Moderation | flagged=true | Any category flagged → toxic |

### Random Seed

```
RANDOM_SEED = 20260118
```

Used in:
- Prompt stratified sampling
- Train/test splits (if applicable)
- Bootstrap confidence intervals

---

## Reproducibility

### Verification Checkpoints

| Checkpoint | Validation |
|------------|------------|
| Data Preparation | 25,000 rows, no duplicates, balanced stratification |
| Model Import | All 6 models "Active" in Bedrock console |
| Inference | 150,000 completions, <1% error rate |
| Scoring | 75,000 rows, no null toxicity values |
| Analysis | All 5 figures generated, p-values computed |

### Running Tests

```bash
# Python tests
pytest tests/ -v

# Test with mock data
Rscript data/sample_prompts.R --mock
```

### Expected Outputs

After a complete pipeline run:

```
output/
├── completions/
│   ├── completions_qwen3.jsonl      # 25,000 lines
│   ├── completions_llama31.jsonl    # 25,000 lines
│   └── completions_mistral.jsonl    # 25,000 lines
├── figures/
│   ├── fig1_toxicity_distributions.png
│   ├── fig2_toxicity_reduction.png
│   ├── fig3_contingency_tables.png
│   ├── fig4_scorer_agreement.png
│   └── fig5_xfakesci_ratio.png
├── tables/
│   ├── mcnemar_results.csv
│   ├── wilcoxon_results.csv
│   ├── cochran_q_results.csv
│   ├── robustness_comparison.csv
│   └── summary_table.csv
└── analysis_dataset_full.csv        # 75,000 rows
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Llama model access denied | Request access on [HuggingFace Meta page](https://huggingface.co/meta-llama) |
| Bedrock import fails | Check model format, ensure S3 permissions |
| Detoxify CUDA errors | Install CPU-only: `pip install detoxify --no-deps` |
| R package install fails | Update R version, install system dependencies |

### AWS Debugging

```bash
# Check model import status
./infrastructure/import_models.sh status

# View CloudWatch logs
aws logs describe-log-streams --log-group-name /aws/bedrock

# Test notifications
./infrastructure/test_notifications.sh
```

### Resetting State

```bash
# Clear all checkpoints and restart
./infrastructure/import_models.sh reset
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{lambros2026safety,
  author = {Lambros, Rock},
  title = {Safety Alignment Effectiveness in Large Language Models:
           A Paired Statistical Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/rocklambros/llm-safety-alignment-study}
}
```

### Referenced Works

- Gehman, S., et al. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. *arXiv:2009.11462*.
- Hartvigsen, T., et al. (2022). ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. *ACL 2022*.
- Hamed, A. A., & Wu, X. (2023). Detection of ChatGPT fake science with the xFakeSci learning algorithm. *arXiv:2308.11767*.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **COMP 4441** course staff for project guidance
- **Allen Institute for AI** for RealToxicityPrompts dataset
- **Microsoft Research** for ToxiGen dataset
- **Anthropic, Meta, Mistral AI, Alibaba** for open-weight models
- **Ahmed A. Hamed and Xindong Wu** for the xFakeSci methodology

---

**Questions?** Open an issue on GitHub or contact the author.
