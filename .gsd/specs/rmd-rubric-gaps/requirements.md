# Requirements: rmd-rubric-gaps

**Status: APPROVED**
**Created**: 2026-03-06
**Feature**: Address rubric gaps in COMP4441 final project Rmd

## Summary

Edit `docs/COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd` to add EDA, assumption checks, visible R code, and a more casual writing tone. Do NOT change any existing results, table values, figure references, or statistical conclusions.

## Target File

`docs/COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd` (141 lines currently)

## Data Landscape (discovered during context gathering)

| Asset | Path | Notes |
|-------|------|-------|
| Pre-cleaned paired data | `analysis/data_validated.rds` | 20MB RDS, primary data source for new code |
| Full inference CSV | `output/analysis_dataset_full.csv` | 341K rows, cols: `prompt_id, family, tox_score_base, tox_score_aligned, tox_binary_base, tox_binary_aligned, scorer, threshold` |
| Prompt metadata | `data/processed/prompt_sample_25k.csv` | Has `source` (rtp/toxigen), `target_group` (13 demographic groups), `stratum` |
| Pre-computed tables | `output/tables/mcnemar_results.csv`, `wilcoxon_results.csv`, `cochran_q_results.csv` | Already loaded by existing Rmd |
| Existing R pipeline | `analysis/01_load_data.R` through `07_summary_tables.R` | Reference for data conventions |
| Prompt ID convention | `rtp_XXXXX` = RealToxicityPrompts, `tg_XXXXX` = ToxiGen | |
| Families | `qwen3`, `llama31`, `mistral` | 25,000 prompts per family |
| Scorer | `detoxify_unbiased` only | Threshold = 0.5 |
| ToxiGen groups | asian, black, chinese, jewish, latino, lgbtq, mental_dis, mexican, middle_east, muslim, native_american, physical_dis, women | 13 groups |

## Change 1: EDA Section

**Location**: Insert `# Exploratory Data Analysis` between `# Data and Methods` (line 43) and `# Statistical Methods` (line 55).

**Data loading**: Load `analysis/data_validated.rds` into a data frame. Run `head()` and `str()` to inspect. If structure differs from expectations, print diagnostics and adapt. Also join with `data/processed/prompt_sample_25k.csv` on `prompt_id` to get `source` and `target_group`.

**Required elements** (all chunks `echo=TRUE`):

1. **Density plot of raw Detoxify scores** — base models only, colored/faceted by `family`. Use ggplot2. Show the distribution of `tox_score_base` across the three families.

2. **Prompt counts by dataset** — bar chart or kable table showing:
   - Total prompts from RealToxicityPrompts vs ToxiGen
   - For ToxiGen: breakdown by all 13 demographic `target_group` values

3. **Base model toxicity rate comparison** — side-by-side bar chart or table of binary toxicity rates (`tox_binary_base`) per family. Purpose: show that all three families start at roughly similar baselines (~18-20%), which motivates cross-family comparison.

**Tone**: Casual intro paragraph. Something like: "Before running the tests, let's get a feel for what the data actually look like."

## Change 2: Assumption Checks Section

**Location**: Insert `# Assumption Checks` immediately before `# Results` (line 63).

**Required elements** (each: one short paragraph + one `echo=TRUE` code chunk):

### 2a. McNemar's Test Assumptions
- Print the `b` (base toxic → aligned clean) and `c` (base clean → aligned toxic) discordant cell counts for each family.
- These are already in `mcnemar_results.csv` columns `cell_b` and `cell_c`, but recompute or display from the raw paired data for transparency.
- Note in prose: b + c must be large enough (rule of thumb: > 25) for the chi-squared approximation. Print actual b + c values.

### 2b. Wilcoxon Signed-Rank Assumptions
- Compute paired differences `d_i = tox_score_base - tox_score_aligned` for each family.
- Plot the distribution of `d_i` (histogram or density) per family.
- Note in prose: the key assumption is that the distribution of differences is roughly symmetric around zero (or at least around its center). Comment on whether the plots support this.

### 2c. Cochran's Q Assumptions
- Confirm the binary coding: each observation is strictly 0 or 1, no NAs.
- Print a quick validation: `table()` of the binary success variable, count of NAs.
- Note in prose: Cochran's Q requires binary responses and complete cases.

## Change 3: Expose Existing Test Code

**Current state**: Global setup (line 21) has `echo = FALSE`. The three results chunks (`table1`, `table2`, and Cochran's Q prose) that build `mcnemar_display`, `wilcoxon_display`, and Q results are invisible.

**Required change**: Add `echo=TRUE` as a chunk-level option to these specific chunks:
- `{r table1}` → `{r table1, echo=TRUE}`
- `{r table2}` → `{r table2, echo=TRUE}`
- Add a new visible chunk for Cochran's Q computation (currently inline prose only) with `echo=TRUE`

Do NOT change the global `echo=FALSE` — other chunks (setup, load-data, figure includes) should stay hidden.

## Change 4: Loosen Prose Tone

**Scope**: Rewrite ALL stiff/formal connective prose throughout the Rmd. Keep statistical terminology precise — only soften the "connective tissue" between ideas.

**Style guide** (examples, not exhaustive):
- "empirical evidence quantifying alignment effectiveness across model families remains limited" → "but we don't really have solid numbers on how well this works across different model families"
- "This study asks:" → "The main question I'm trying to answer here is:"
- "Despite widespread adoption" → "Even though everyone's using these techniques"
- Sound like a grad student writing a class paper, not a journal submission

**Constraints**:
- Do NOT change statistical method descriptions (McNemar's formula, Wilcoxon description, Cochran's Q definition)
- Do NOT change any numeric values, p-values, effect sizes, or CI ranges
- Do NOT change figure captions or table captions
- Do NOT alter the References section
- Sections affected: Introduction, Data and Methods, Statistical Methods, Discussion, and any new EDA/Assumptions prose

## Non-Functional Requirements

- All new code chunks must use `echo=TRUE` so grader can see R code
- All plots must use ggplot2 (already loaded in setup)
- New sections must not break existing figure/table numbering
- The Rmd must still knit to both `word_document` and `pdf_document` without errors
- Data paths must be relative (the Rmd is in `docs/`, so use `../analysis/`, `../data/`, `../output/`)

## Implementation Notes

- The `analysis/data_validated.rds` is the recommended data source — load with `readRDS("../analysis/data_validated.rds")`
- If the RDS structure is unexpected, print `head()` and `str()` first and adapt
- The prompt metadata join uses: `prompt_sample <- read_csv("../data/processed/prompt_sample_25k.csv")` then join on `prompt_id`
- Existing R packages in setup: tidyverse, exact2x2, coin, DescTools, knitr, patchwork — these should suffice
- The existing `analysis/02_eda.R` (26KB) may contain reusable plot patterns — reference but don't copy wholesale

## Acceptance Criteria

1. New `# Exploratory Data Analysis` section exists between Data/Methods and Statistical Methods
2. EDA contains: density plot, prompt count breakdown (with 13 ToxiGen groups), base toxicity rate comparison
3. New `# Assumption Checks` section exists before Results
4. Assumptions section covers McNemar (discordant counts), Wilcoxon (symmetry plots), Cochran (binary validation)
5. All new chunks and the three existing results chunks have `echo=TRUE`
6. Prose throughout reads casual/grad-student, not journal-formal
7. No existing results, table values, figures, or references are altered
8. Rmd knits without errors

## Open Questions

None — all resolved during elicitation.

## Dependencies

- R with tidyverse, ggplot2, knitr installed
- Access to `analysis/data_validated.rds` and `data/processed/prompt_sample_25k.csv`

## Risk Assessment

- **Low**: RDS file might have different column names than the CSV → mitigated by printing `str()` first
- **Low**: Large dataset (341K rows) might be slow in knitr → the RDS is likely pre-filtered to 75K rows (25K × 3 families)
- **Medium**: Tone changes are subjective → implementer should err toward "slightly too casual" rather than "still too formal"
