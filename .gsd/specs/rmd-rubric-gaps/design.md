# Technical Design: rmd-rubric-gaps

## Metadata
- **Feature**: rmd-rubric-gaps
- **Status**: APPROVED
- **Created**: 2026-03-06
- **Author**: Factory Design Mode

---

## 1. Overview

### 1.1 Summary
Edit the existing R Markdown paper (`docs/COMP4441-FinalProject-SafetyAlignmentEffectiveness-LLM-RockLambros.Rmd`) to address grading rubric gaps: add EDA visualizations, assumption check sections, expose R code via `echo=TRUE`, and soften the prose tone. All changes are additive or stylistic -- no existing results, table values, or statistical conclusions are modified.

### 1.2 Goals
- Add EDA section with density plot, prompt counts, and base toxicity comparison
- Add assumption checks for McNemar, Wilcoxon, and Cochran's Q
- Make R code visible in results chunks (`echo=TRUE`)
- Rewrite connective prose in casual grad-student tone

### 1.3 Non-Goals
- Changing any statistical results, p-values, effect sizes, or CI ranges
- Modifying figure captions, table captions, or the References section
- Altering the global `echo=FALSE` default in the setup chunk
- Changing the analysis pipeline scripts in `analysis/`

---

## 2. Architecture

### 2.1 High-Level Design

Single-file modification with 4 logical change sets applied sequentially:

```
Rmd (141 lines)
  |
  +-- [INSERT] EDA section (after line 53, before Statistical Methods)
  |     +-- load-eda chunk: readRDS + join prompt metadata
  |     +-- density plot chunk: tox_score_base by family
  |     +-- prompt counts chunk: RTP vs ToxiGen breakdown
  |     +-- base toxicity rates chunk: binary rates per family
  |
  +-- [INSERT] Assumption Checks section (before Results, line 63+offset)
  |     +-- McNemar assumptions chunk: discordant counts
  |     +-- Wilcoxon assumptions chunk: paired difference distributions
  |     +-- Cochran assumptions chunk: binary validation
  |
  +-- [MODIFY] Existing chunks: add echo=TRUE to table1, table2, new Cochran chunk
  |
  +-- [MODIFY] Prose throughout: casual tone rewrite
```

### 2.2 Component Breakdown

| Component | Responsibility | Location |
|-----------|---------------|----------|
| EDA section | Exploratory visualizations and data summaries | New section between Data/Methods and Statistical Methods |
| Assumption Checks | Pre-test validation of statistical assumptions | New section before Results |
| Code Exposure | Make existing analysis code visible | Chunk options on table1, table2, new Cochran chunk |
| Prose Rewrite | Soften formal academic tone | Throughout Introduction, Data/Methods, Statistical Methods, Discussion |

### 2.3 Data Flow

```
data_validated.rds ($data tibble, 75000x25)
  |
  +-- prompt_sample_25k.csv (join on prompt_id for source, target_group)
  |
  +-- EDA plots: tox_score_base density, prompt counts, binary rates
  +-- Assumption checks: discordant counts from paired data, d_i distributions

mcnemar_results.csv, wilcoxon_results.csv, cochran_q_results.csv
  |
  +-- Already loaded by existing load-data chunk (no change)
  +-- echo=TRUE applied to display chunks
```

---

## 3. Detailed Design

### 3.1 Data Model (RDS Structure)

```r
# readRDS("../analysis/data_validated.rds") returns a list:
# $data: tibble 75000 x 25
#   Key columns:
#     prompt_id      - character (rtp_XXXXX or tg_XXXXX)
#     family         - character (qwen3, llama31, mistral)
#     tox_score_base - numeric (continuous toxicity score, base model)
#     tox_score_aligned - numeric (continuous toxicity score, aligned model)
#     tox_binary_base   - integer (0/1 binary at threshold 0.5)
#     tox_binary_aligned - integer (0/1 binary at threshold 0.5)
#     scorer         - character (detoxify_unbiased)
#     threshold      - numeric (0.5)
#
# prompt_sample_25k.csv columns:
#   prompt_id, source (rtp/toxigen), target_group (13 groups or NA), stratum
```

### 3.2 New Chunks Design

#### EDA Data Loading (`load-eda`)
```r
validated <- readRDS("../analysis/data_validated.rds")
df <- validated$data
str(df)
prompt_meta <- read_csv("../data/processed/prompt_sample_25k.csv", show_col_types = FALSE)
df <- df %>% left_join(prompt_meta %>% select(prompt_id, source, target_group), by = "prompt_id")
```

#### Density Plot (`eda-density`)
```r
ggplot(df, aes(x = tox_score_base, fill = family)) +
  geom_density(alpha = 0.4) +
  labs(title = "Distribution of Base Model Toxicity Scores",
       x = "Detoxify Toxicity Score", y = "Density", fill = "Family") +
  theme_minimal()
```

#### Prompt Counts (`eda-prompt-counts`)
```r
# RTP vs ToxiGen totals + ToxiGen breakdown by 13 target groups
```

#### Base Toxicity Rates (`eda-base-rates`)
```r
# Side-by-side bar chart of tox_binary_base mean per family
```

#### McNemar Assumptions (`assume-mcnemar`)
```r
# Compute discordant pairs b, c from raw paired data per family
# Print b + c values, check > 25 rule of thumb
```

#### Wilcoxon Assumptions (`assume-wilcoxon`)
```r
# Compute d_i = tox_score_base - tox_score_aligned per family
# Plot density of d_i, comment on symmetry
```

#### Cochran Assumptions (`assume-cochran`)
```r
# table() of binary success variable, count NAs
# Confirm strict 0/1 coding with no missing values
```

### 3.3 Chunk Option Changes

| Chunk | Current | After |
|-------|---------|-------|
| `{r table1}` | `echo` inherits FALSE | `{r table1, echo=TRUE}` |
| `{r table2}` | `echo` inherits FALSE | `{r table2, echo=TRUE}` |
| Cochran's Q | Inline prose only | New `{r cochran-q-display, echo=TRUE}` chunk |

---

## 4. Key Decisions

### Decision: Sequential Task Execution

**Context**: All 4 changes target a single file. The zerg framework optimizes for parallel file-level execution.

**Options Considered**:
1. Split into parallel tasks with merge step: Each task outputs a patch, final task merges
2. Strictly sequential tasks: Each task modifies the Rmd in order
3. Single monolithic task: One worker does everything

**Decision**: Strictly sequential tasks (option 2)

**Rationale**: A single file cannot be safely edited in parallel. Sequential tasks provide clear checkpoints and reviewability. A monolithic task would be too large for a single worker context window and harder to debug if something goes wrong.

**Consequences**: Max parallelization is 1. Total duration equals sum of all task estimates. However, each task is focused and verifiable independently.

### Decision: Load RDS Once in EDA, Reuse in Assumptions

**Context**: Both the EDA and Assumption Checks sections need the raw paired data from `data_validated.rds`.

**Decision**: Load in the EDA section's `load-eda` chunk. The Assumption Checks section reuses the `df` object already in the knitr environment.

**Rationale**: Avoids double-loading a 20MB file. Keeps data flow clean.

### Decision: Insertion Order

**Context**: Inserting new sections shifts line numbers for subsequent edits.

**Decision**: Insert from top to bottom: EDA first (after Data/Methods), then Assumption Checks (before Results). Then modify existing chunks. Prose rewrite last.

**Rationale**: Top-to-bottom insertion means each task can reference stable line numbers relative to its starting state.

---

## 5. Implementation Plan

### 5.1 Phase Summary

| Phase | Tasks | Parallel | Est. Time |
|-------|-------|----------|-----------|
| Foundation (L1) | 1 (EDA section) | N/A | 20 min |
| Core (L2) | 1 (Assumption Checks) | No | 20 min |
| Integration (L3) | 1 (Code exposure + Cochran chunk) | No | 10 min |
| Polish (L4) | 1 (Prose tone rewrite) | No | 15 min |
| Testing (L5) | 1 (Knit validation) | No | 10 min |
| Quality (L6) | 1 (CHANGELOG update) | No | 5 min |

### 5.2 File Ownership

| File | Task ID | Phase | Operation |
|------|---------|-------|-----------|
| docs/...Rmd | TASK-001 | foundation | modify (insert EDA) |
| docs/...Rmd | TASK-002 | core | modify (insert assumptions) |
| docs/...Rmd | TASK-003 | integration | modify (echo=TRUE + Cochran chunk) |
| docs/...Rmd | TASK-004 | polish | modify (prose rewrite) |
| docs/...Rmd | TASK-005 | testing | read (knit test) |
| CHANGELOG.md | TASK-006 | quality | create/modify |

**Note**: File ownership is sequential -- each task hands off the file to the next. No concurrent modifications.

### 5.3 Dependency Graph

```
TASK-001 (EDA) --> TASK-002 (Assumptions) --> TASK-003 (echo=TRUE)
                                                    |
                                                    v
                                              TASK-004 (Prose)
                                                    |
                                                    v
                                              TASK-005 (Knit)
                                                    |
                                                    v
                                              TASK-006 (CHANGELOG)
```

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RDS column names differ from expected | Low | Medium | Print `str()` and `head()` first; adapt code |
| Large dataset slow in knitr | Low | Low | RDS is 75K rows (pre-filtered), manageable |
| Prose tone too casual or not casual enough | Medium | Low | Err toward "slightly too casual" per requirements |
| Line number drift between tasks | Medium | Medium | Each task reads current file state; no hardcoded line numbers |
| New chunks break figure/table numbering | Low | High | New sections use `echo=TRUE` code only; no new figures in auto-numbering |
| Rmd fails to knit after changes | Medium | High | TASK-005 validates knitting; fix issues before marking complete |

---

## 7. Testing Strategy

### 7.1 Per-Task Verification
- TASK-001: Rmd parses (no syntax errors in new chunks)
- TASK-002: Rmd parses (no syntax errors in assumption chunks)
- TASK-003: Grep confirms `echo=TRUE` on table1, table2, cochran-q-display
- TASK-004: Manual review of prose tone
- TASK-005: Full knit to both word_document and pdf_document

### 7.2 Acceptance Criteria Verification
1. EDA section exists between Data/Methods and Statistical Methods
2. EDA contains density plot, prompt counts (with 13 ToxiGen groups), base rates
3. Assumption Checks section exists before Results
4. Assumptions cover McNemar (discordant), Wilcoxon (symmetry), Cochran (binary)
5. table1, table2, cochran-q-display all have `echo=TRUE`
6. Prose reads casual/grad-student throughout
7. No existing results altered
8. Rmd knits without errors

---

## 8. Parallel Execution Notes

### 8.1 Safe Parallelization
- **None possible**: Single target file means strictly sequential execution
- Each task depends on the prior task's output state of the Rmd

### 8.2 Recommended Workers
- Minimum: 1 worker (sequential)
- Optimal: 1 worker (single file constraint)
- Maximum: 1 worker (no benefit from more)

### 8.3 Estimated Duration
- Single worker: ~80 minutes
- No parallelization speedup available

---

## 9. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Architecture | | | PENDING |
| Engineering | | | PENDING |
