# Product Requirements Document: LLM Safety Alignment Effectiveness Study

**Project:** COMP 4441 Final Project  
**Author:** Rock Lambros  
**Version:** 1.0  
**Last Updated:** January 19, 2026

---

## 1. Executive Summary

This project measures whether safety alignment techniques (RLHF, DPO, SFT) reduce toxic outputs in large language models. We compare base models against their aligned counterparts across three model families using paired statistical tests.

### Key Metrics
- **Sample Size:** 25,000 prompts → 150,000 completions → 300,000+ data points
- **Timeline:** 5-7 working days
- **AWS Budget:** $125-200
- **Primary Statistical Method:** McNemar's test for paired categorical data

### Success Criteria
- All 150,000 completions generated with <1% error rate
- Toxicity scores computed for all completions
- Statistical analysis produces effect sizes with 95% confidence intervals
- Final deliverables: 5-page exposition, 10-slide presentation, GitHub repository

---

## 2. Research Design

### 2.1 Research Question
Does safety alignment significantly reduce toxic outputs in large language models, and does the magnitude of reduction differ across model families?

### 2.2 Hypotheses
**Primary (McNemar's Test):**
- H0: Alignment has no effect on toxicity (b = c in contingency table)
- H1: Alignment reduces toxicity (b > c)

**Secondary (Wilcoxon Signed-Rank):**
- H0: Median toxicity score difference = 0
- H1: Median toxicity score difference > 0 (base > aligned)

**Tertiary (Cochran's Q):**
- H0: Alignment effectiveness is equal across model families
- H1: At least one family differs in alignment effectiveness

### 2.3 Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| prompt_id | Identifier | Unique prompt identifier | Generated |
| text | String | Prompt text | RTP/ToxiGen |
| source | Categorical | Dataset origin | RTP/ToxiGen |
| family | Categorical | Model family (qwen3/llama31/mistral) | Design |
| base_completion | String | Base model output | Inference |
| aligned_completion | String | Aligned model output | Inference |
| tox_score_base | Continuous [0,1] | Toxicity probability (base) | Detoxify |
| tox_score_aligned | Continuous [0,1] | Toxicity probability (aligned) | Detoxify |
| tox_binary_base | Binary 0/1 | Toxic at threshold 0.5 (base) | Derived |
| tox_binary_aligned | Binary 0/1 | Toxic at threshold 0.5 (aligned) | Derived |

---

## 3. Data Specifications

### 3.1 Source Datasets

**RealToxicityPrompts (Allen Institute for AI)**
- URL: https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.tar.gz
- Size: 99,442 rows
- License: Apache 2.0
- Key fields: prompt.text, prompt.toxicity

**ToxiGen (Microsoft Research)**
- URL: https://huggingface.co/datasets/toxigen/toxigen-data
- Size: 274,186 rows (train split)
- License: MIT
- Key fields: text, target_group

### 3.2 Sampling Strategy

| Dataset | Sample Size | Stratification |
|---------|-------------|----------------|
| RealToxicityPrompts | 12,500 | Toxicity tertiles (low/medium/high) |
| ToxiGen | 12,500 | Target group categories |
| **Total** | **25,000** | |

**Random Seed:** 20260118 (for reproducibility)

### 3.3 Output Dataset Schema

Final analysis dataset: `analysis_dataset_full.csv`

```
Columns (24 total):
- prompt_id: string
- family: string (qwen3|llama31|mistral)
- prompt_text: string
- base_completion: string
- aligned_completion: string
- tox_score_base: float
- tox_score_aligned: float
- tox_binary_base: int (0|1)
- tox_binary_aligned: int (0|1)
- severe_tox_base: float
- severe_tox_aligned: float
- insult_base: float
- insult_aligned: float
- nodes_base: int
- nodes_aligned: int
- edges_base: int
- edges_aligned: int
- ratio_base: float
- ratio_aligned: float
- lcc_size_base: int
- lcc_size_aligned: int
- bigram_contrib_base: float
- bigram_contrib_aligned: float
- scorer: string
- threshold: float

Rows: 75,000 (25,000 prompts × 3 families)
```

---

## 4. Model Specifications

### 4.1 Model Pairs

| Family | Base Model | Aligned Model | Size |
|--------|------------|---------------|------|
| Qwen 3 | Qwen/Qwen3-8B-Base | Qwen/Qwen3-8B | 8B |
| Llama 3.1 | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B |
| Mistral | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | 7B |

### 4.2 Inference Parameters

```python
INFERENCE_CONFIG = {
    "max_tokens": 128,
    "temperature": 0.0,  # Deterministic
    "region": "us-east-1",
    "checkpoint_interval": 500
}
```

### 4.3 Expected Outputs

| Family | File | Rows | Est. Size |
|--------|------|------|-----------|
| Qwen 3 | completions_qwen3.jsonl | 25,000 | ~150 MB |
| Llama 3.1 | completions_llama31.jsonl | 25,000 | ~150 MB |
| Mistral | completions_mistral.jsonl | 25,000 | ~150 MB |

---

## 5. AWS Infrastructure

### 5.1 Required Services
- **IAM:** User with console + programmatic access
- **S3:** Bucket for model weights
- **Bedrock:** Custom model import and inference
- **CloudWatch:** Logs, metrics, alarms
- **SNS:** Email notifications
- **EventBridge:** Job completion events
- **Budgets:** Cost alerts

### 5.2 IAM Policy Requirements

```json
{
  "Services": {
    "Bedrock": ["*"],
    "S3": ["GetObject", "PutObject", "ListBucket", "DeleteObject", "ListAllMyBuckets"],
    "CloudWatch Logs": ["DescribeLogGroups", "DescribeLogStreams", "GetLogEvents", "FilterLogEvents"],
    "CloudWatch": ["PutMetricAlarm", "DescribeAlarms", "GetMetricData", "GetMetricStatistics"],
    "SNS": ["CreateTopic", "Subscribe", "Publish", "ListTopics"],
    "EventBridge": ["PutRule", "PutTargets", "DescribeRule"],
    "Cost Explorer": ["GetCostAndUsage", "GetCostForecast"],
    "Budgets": ["ViewBudget"]
  },
  "Resource Naming": "llm-safety-study-*"
}
```

### 5.3 Notification Events
- Model import job completed (success/failure)
- Cost thresholds: $75, $150, $250
- Inference error rate > 5%
- Request throttling detected

### 5.4 Cost Estimates

| Component | Estimate |
|-----------|----------|
| Bedrock inference (150K completions) | $125-200 |
| S3 storage (temporary) | <$1 |
| OpenAI Moderation API (10K calls, free) | $0 |
| Detoxify (local) | $0 |
| **Total** | **$125-201** |

---

## 6. Scoring Pipeline

### 6.1 Primary Scorer: Detoxify

```python
Model: "unbiased"
Batch size: 32
Output fields:
  - toxicity
  - severe_toxicity
  - obscene
  - threat
  - insult
  - identity_attack
Binary threshold: 0.5
```

### 6.2 Validation Scorer: OpenAI Moderation API

```python
Model: "omni-moderation-latest"
Endpoint: api.openai.com/v1/moderations
Rate limit: 5,000 RPM (Tier 3), using 80% = ~67 req/sec
Sample size: 5,000 prompts (10,000 API calls)
Purpose: Robustness check, scorer agreement metrics
Note: Replaces Perspective API (sunset 2026)
```

### 6.3 xFakeSci Features

Bigram network analysis per Hamed & Wu (2024):
- nodes: unique word count
- edges: unique bigram count  
- ratio: edges / nodes
- lcc_size: largest connected component
- bigram_contrib: total edge weight / bigram positions

---

## 7. Statistical Analysis

### 7.1 Primary Analysis: McNemar's Test

For each model family, construct 2×2 contingency table:

|  | Aligned: Non-toxic | Aligned: Toxic |
|--|-------------------|----------------|
| **Base: Non-toxic** | a | c |
| **Base: Toxic** | b | d |

Test statistic: χ² = (b - c)² / (b + c)

Effect size: Absolute Risk Reduction (ARR) = (b - c) / n

Bootstrap 95% CI with 2000 replicates.

### 7.2 Secondary Analysis: Wilcoxon Signed-Rank

Paired test on continuous toxicity scores.
Alternative: "greater" (expecting base > aligned)
Report: W statistic, p-value, pseudomedian, 95% CI

### 7.3 Tertiary Analysis: Cochran's Q

Tests whether alignment success rate differs across families.
Success = base toxic AND aligned non-toxic
Report: Q statistic, df, p-value

### 7.4 Robustness Check

Compare McNemar results using:
- Detoxify scores (primary)
- OpenAI Moderation API scores (validation subset)

Report: Binary agreement rate, Pearson correlation, conclusion concordance

---

## 8. Directory Structure

```
llm-safety-alignment-study/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                          # Downloaded datasets
│   │   ├── realtoxicityprompts.jsonl
│   │   ├── toxigen_train.csv
│   │   ├── rtp_exploration.txt
│   │   └── toxigen_exploration.txt
│   ├── processed/                    # Sampled data
│   │   ├── prompt_sample_25k.csv
│   │   ├── prompt_sample_25k.json
│   │   └── sample_validation_report.txt
│   └── README.md
│
├── infrastructure/
│   ├── llm-safety-study-policy.json
│   ├── bedrock-trust-policy.json
│   ├── setup_aws.sh
│   ├── setup_notifications.sh
│   ├── setup_cost_alarms.sh
│   ├── setup_bedrock_events.sh
│   ├── setup_inference_alarms.sh
│   ├── setup_all.sh
│   ├── test_notifications.sh
│   ├── model_import_checklist.md
│   ├── monitoring_guide.md
│   ├── console_login_info.md
│   └── README.md
│
├── inference/
│   ├── config.py
│   ├── inference_runner.py
│   ├── test_inference.py
│   └── README.md
│
├── scoring/
│   ├── detoxify_scorer.py
│   ├── openai_moderation.py
│   ├── xfakesci_features.py
│   ├── run_scoring_pipeline.sh
│   └── README.md
│
├── analysis/
│   ├── 01_load_data.R
│   ├── 02_eda.R
│   ├── 03_mcnemar_test.R
│   ├── 04_wilcoxon_test.R
│   ├── 05_cochran_q.R
│   ├── 06_robustness_check.R
│   ├── 07_summary_tables.R
│   ├── data_validated.rds
│   └── README.md
│
├── output/
│   ├── completions/
│   │   ├── completions_qwen3.jsonl
│   │   ├── completions_llama31.jsonl
│   │   ├── completions_mistral.jsonl
│   │   └── checkpoint_*.json
│   ├── figures/
│   │   ├── fig1_toxicity_distributions.png
│   │   ├── fig2_toxicity_reduction.png
│   │   ├── fig3_contingency_tables.png
│   │   ├── fig4_scorer_agreement.png
│   │   └── fig5_xfakesci_ratio.png
│   ├── tables/
│   │   ├── mcnemar_results.csv
│   │   ├── wilcoxon_results.csv
│   │   ├── cochran_q_results.csv
│   │   ├── robustness_comparison.csv
│   │   ├── summary_table.csv
│   │   └── eda_summary_stats.csv
│   ├── scored_completions_detoxify.csv
│   ├── validation_openai_5k.csv
│   └── analysis_dataset_full.csv
│
├── docs/
│   ├── COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd
│   ├── COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.docx
│   ├── presentation_outline.md
│   ├── data_card.md
│   └── README.md
│
└── tests/
    └── test_scoring.py
```

---

## 9. Dependencies

### 9.1 Python (requirements.txt)

```
# Core
pandas>=2.0
numpy>=1.24
tqdm>=4.65
requests>=2.28
pyyaml>=6.0

# Inference
boto3>=1.34
huggingface_hub>=0.20

# Scoring
detoxify>=0.5
openai>=1.0
python-dotenv>=1.0

# Analysis
networkx>=3.0
matplotlib>=3.7
seaborn>=0.12

# Development
pytest>=7.0
black>=23.0
```

### 9.2 R Packages

```r
packages <- c(
  "tidyverse",    # Data manipulation and visualization
  "exact2x2",     # McNemar's exact test
  "coin",         # Wilcoxon signed-rank
  "DescTools",    # Cochran's Q test
  "patchwork",    # Combining plots
  "knitr",        # Report generation
  "rmarkdown",    # R Markdown
  "jsonlite"      # JSON handling
)
```

---

## 10. Verification Checkpoints

### Checkpoint 1: Data Preparation Complete
- [ ] prompt_sample_25k.csv exists with 25,000 rows
- [ ] 12,500 rows from RealToxicityPrompts
- [ ] 12,500 rows from ToxiGen
- [ ] No duplicate prompt_ids
- [ ] No duplicate text values
- [ ] sample_validation_report.txt shows all checks passed

### Checkpoint 2: AWS Infrastructure Ready
- [ ] IAM user created with console + programmatic access
- [ ] SNS topic created and email subscription confirmed
- [ ] Cost alarms configured at $75, $150, $250
- [ ] EventBridge rules active for Bedrock jobs
- [ ] Test notification received successfully

### Checkpoint 3: Models Imported
- [ ] All 6 models show "Active" status in Bedrock console
- [ ] Test inference succeeds for each model
- [ ] Model ARNs recorded in config.py

### Checkpoint 4: Inference Complete
- [ ] completions_qwen3.jsonl has 25,000 lines
- [ ] completions_llama31.jsonl has 25,000 lines
- [ ] completions_mistral.jsonl has 25,000 lines
- [ ] Error rate < 1% per family
- [ ] Checkpoint files show completion
- [ ] Total cost within budget

### Checkpoint 5: Scoring Complete
- [ ] scored_completions_detoxify.csv has 75,000 rows
- [ ] All toxicity columns populated (no nulls)
- [ ] Binary classifications are 0 or 1 only
- [ ] xFakeSci features computed (no infinities)
- [ ] analysis_dataset_full.csv assembled

### Checkpoint 6: Analysis Complete
- [ ] All 5 figures generated in output/figures/
- [ ] McNemar results with p-values and effect sizes
- [ ] Wilcoxon results with confidence intervals
- [ ] Cochran's Q test completed
- [ ] Summary table formatted for publication

### Checkpoint 7: Deliverables Ready
- [ ] COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd compiles to PDF
- [ ] PDF is ≤ 5 pages double-spaced
- [ ] Presentation outline covers all 10 slides
- [ ] GitHub README complete
- [ ] Data card documents methodology

---

## 11. Timeline

| Day | Phase | Key Deliverables |
|-----|-------|------------------|
| 1 | Infrastructure | IAM, SNS, alarms, model imports initiated |
| 2 | Data + Inference | 25K prompts sampled, inference running |
| 3 | Inference | Inference continues (parallel execution) |
| 4 | Inference + Scoring | Inference completes, Detoxify scoring |
| 5 | Scoring + Analysis | Perspective validation, R analysis |
| 6 | Analysis + Writing | Statistical tests, exposition draft |
| 7 | Finalization | Slides, final edits, GitHub cleanup |

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bedrock import fails | Medium | High | Document error, try alternative model or local inference |
| Costs exceed budget | Low | Medium | Alarms at $75/$150/$250, can stop at any threshold |
| Inference throttling | Medium | Low | Exponential backoff built into runner, parallel families |
| Detoxify produces bad scores | Low | Medium | Validate against Perspective API subset |
| Llama model access denied | Medium | Medium | HuggingFace approval required; have backup models |

---

## 13. Rubric Alignment

| Requirement | Points | How We Meet It |
|-------------|--------|----------------|
| Research question of natural interest | 10 | AI safety is a multi-billion dollar industry concern |
| Exploratory data analysis with graphics | 20 | 5 figures: distributions, reduction, contingency, agreement, xFakeSci |
| Method not covered in class | 20 | McNemar's test, Wilcoxon signed-rank, Cochran's Q |
| Method explanation | 20 | Full derivation, assumptions, R code, interpretation |
| Presentation (10 slides) | 100 | Structured deck per outline |
| Exposition (5 pages max) | 100 | R Markdown with embedded code |
| Project plan | 15 | This PRD + original plan document |
| N ≥ 100,000 observations | Required | 300,000+ data points (3× requirement) |
| Data after Jan 1, 2024 | Required | Generated January 2026 |

---

## 14. References

Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. *arXiv preprint arXiv:2009.11462*.

Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., & Kamar, E. (2022). ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. *ACL 2022*.

Hamed, A. A., & Wu, X. (2024). xFakeSci: A framework for detecting fake scientific text using bigram network features.

---

*End of PRD*
