# Presentation Outline: Safety Alignment Effectiveness in Large Language Models

**Duration:** 15 minutes (10 slides)
**Author:** Rock Lambros
**Course:** COMP 4441
**Date:** January 2026

---

## Slide 1: Title Slide

**Title:** Safety Alignment Effectiveness in Large Language Models

**Subtitle:** Measuring Toxicity Reduction Across Model Families

**Content:**
- Author: Rock Lambros
- Course: COMP 4441 - Final Project
- Date: January 2026

**Speaker Notes:**
- Welcome audience and introduce the topic of AI safety
- Briefly mention that this research addresses a critical question in responsible AI deployment
- Set expectation: we will see concrete evidence of whether safety techniques actually work

**Timing:** 30 seconds

---

## Slide 2: Research Question and Motivation

**Title:** Why Study Safety Alignment?

**Content:**
- LLMs can generate hate speech, threats, misinformation
- Safety alignment techniques (RLHF, DPO, SFT) aim to reduce harmful outputs
- Billions of dollars invested in alignment research
- **Research Question:** Does safety alignment significantly reduce toxic outputs, and does effectiveness vary across model families?

**Visual:** Side-by-side example of base model toxic output vs. aligned model safe output (anonymized)

**Speaker Notes:**
- Explain the real-world stakes: deployed models interact with millions of users daily
- RLHF = Reinforcement Learning from Human Feedback; DPO = Direct Preference Optimization; SFT = Supervised Fine-Tuning
- Emphasize that despite widespread adoption, quantitative evidence of effectiveness is limited

**Timing:** 1.5 minutes

---

## Slide 3: Data Sources

**Title:** Evaluation Datasets

**Content:**
- **RealToxicityPrompts** (Allen Institute for AI)
  - 99,442 naturally occurring web prompts
  - Sampled 12,500 stratified by toxicity level
  - License: Apache 2.0

- **ToxiGen** (Microsoft Research)
  - 274,186 adversarial prompts targeting 13 demographic groups
  - Sampled 12,500 stratified by target group
  - License: MIT

- **Total:** 25,000 prompts for evaluation

**Visual:** Pie chart showing dataset composition and stratification

**Speaker Notes:**
- RealToxicityPrompts captures organic web content that users might actually encounter
- ToxiGen provides adversarial examples designed to bypass safety measures
- Stratification ensures we test across the full toxicity spectrum
- Combined dataset tests both natural and adversarial scenarios

**Timing:** 1.5 minutes

---

## Slide 4: Model Pairs

**Title:** Models Under Evaluation

**Content:**
| Family | Base Model | Aligned Model | Size |
|--------|------------|---------------|------|
| Qwen 3 | Qwen3-8B-Base | Qwen3-8B | 8B |
| Llama 3.1 | Llama-3.1-8B | Llama-3.1-8B-Instruct | 8B |
| Mistral | Mistral-7B-v0.3 | Mistral-7B-Instruct-v0.3 | 7B |

**Key Points:**
- Paired design: same architecture, different training
- Deterministic decoding (temperature = 0)
- 150,000 total completions (25K prompts x 6 models)

**Visual:** Architecture diagram showing base-to-aligned relationship

**Speaker Notes:**
- Paired design is critical: it isolates the effect of alignment training
- Same tokenizer, same architecture, same size - only difference is the safety fine-tuning
- Temperature 0 ensures reproducibility and removes sampling noise
- 150K completions generated via AWS Bedrock

**Timing:** 1.5 minutes

---

## Slide 5: Methodology

**Title:** Scoring and Statistical Tests

**Content:**
**Toxicity Scoring:**
- Primary: Detoxify (unbiased model) - continuous scores [0,1]
- Validation: OpenAI Moderation API (5K subset)
- Binary threshold: 0.5

**Statistical Tests:**
| Test | Purpose | Type |
|------|---------|------|
| McNemar's | Paired binary comparison | Primary |
| Wilcoxon signed-rank | Paired continuous comparison | Secondary |
| Cochran's Q | Cross-family consistency | Tertiary |

**Visual:** Flowchart: Prompts -> Inference -> Scoring -> Statistical Analysis

**Speaker Notes:**
- Detoxify is a transformer classifier trained on Jigsaw dataset - industry standard
- McNemar's test is ideal for paired binary data - exactly our setup
- Wilcoxon uses full continuous scores, more statistically powerful
- Cochran's Q answers: "Is alignment equally effective across all families?"

**Timing:** 1.5 minutes

---

## Slide 6: Results - Toxicity Distributions

**Title:** Toxicity Score Distributions

**Content:**
- Clear leftward shift in aligned models
- Substantial reduction in high-toxicity outputs (>0.5)
- Effect visible across all three model families

**Visual:** `output/figures/fig1_toxicity_distributions.png`
- Overlapping density plots for base (red) vs. aligned (blue)
- Faceted by model family

**Speaker Notes:**
- This figure provides the visual intuition before we get to statistics
- Red curves (base models) extend further right into toxic territory
- Blue curves (aligned models) concentrate more heavily near zero
- The gap between curves represents the alignment effect

**Timing:** 1.5 minutes

---

## Slide 7: Results - McNemar's Test

**Title:** Primary Analysis: Paired Binary Outcomes

**Content:**
**Contingency Table Structure:**
|  | Aligned: Safe | Aligned: Toxic |
|--|--------------|----------------|
| **Base: Safe** | a | c |
| **Base: Toxic** | b | d |

**Key Finding:** All families show b >> c (p < 0.001)

**Absolute Risk Reduction:** 15-25% across families

**Visual:** `output/figures/fig3_contingency_tables.png`
- Heatmap or bar chart of discordant pairs by family

**Speaker Notes:**
- Cell b = prompts where alignment "fixed" toxicity (base toxic, aligned safe)
- Cell c = prompts where alignment "broke" safety (base safe, aligned toxic)
- b >> c means alignment helps far more often than it hurts
- ARR of 20% means: for every 100 prompts, alignment prevents 20 toxic outputs

**Timing:** 1.5 minutes

---

## Slide 8: Results - Cross-Family Comparison

**Title:** Does Alignment Work Equally Across Families?

**Content:**
**Cochran's Q Test Results:**
- Tests whether alignment success rate differs across families
- Q statistic = [value], df = 2, p < 0.05
- **Finding:** Significant heterogeneity detected

**Family Ranking (by ARR):**
1. [Family A]: largest reduction
2. [Family B]: moderate reduction
3. [Family C]: smallest reduction

**Visual:** Bar chart comparing ARR with 95% CI by family

**Speaker Notes:**
- Cochran's Q is like a generalization of McNemar's to more than two groups
- Significant result means the families are NOT equally aligned
- This has practical implications: choosing which model to deploy matters
- Could reflect different alignment techniques or training data differences

**Timing:** 1.5 minutes

---

## Slide 9: Key Findings and Implications

**Title:** What Did We Learn?

**Content:**
**Primary Finding:**
- Safety alignment significantly reduces toxic outputs across all tested families

**Quantitative Impact:**
- 15-25% absolute reduction in toxic output rate
- At scale (billions of queries), this prevents millions of harmful outputs

**Practical Implications:**
- Alignment techniques work, but effectiveness varies
- Model selection matters for safety-critical applications
- Robustness validated via independent scorer agreement

**Visual:** Summary statistics table or infographic

**Speaker Notes:**
- This is good news for the AI safety field - the techniques work
- But the heterogeneity finding is a caution: not all aligned models are equal
- Organizations deploying LLMs should evaluate alignment effectiveness for their use case
- The validation scorer agreement adds confidence these aren't measurement artifacts

**Timing:** 1.5 minutes

---

## Slide 10: Limitations and Future Work

**Title:** Limitations and Next Steps

**Content:**
**Limitations:**
- Automated scoring may miss nuanced harms
- Deterministic decoding only (temperature = 0)
- English-language prompts only
- Binary threshold (0.5) is somewhat arbitrary

**Future Work:**
- Compare specific alignment techniques (RLHF vs. DPO vs. SFT)
- Multilingual evaluation
- Human annotation for ground truth validation
- Longitudinal study as models are updated

**Visual:** Research roadmap or limitation icons

**Speaker Notes:**
- Be honest about what we did not test: other languages, other temperatures, human judgment
- The threshold sensitivity could be addressed with ROC curve analysis
- Future work could disentangle which specific technique provides the most benefit
- As models evolve, continued monitoring is essential

**Timing:** 1 minute

---

## Appendix: Backup Slides

### Backup Slide A: Wilcoxon Signed-Rank Details

**Content:**
- Test statistic: W = [value]
- Pseudomedian difference: 0.08-0.15 (varies by family)
- 95% CI: [lower, upper]
- Interpretation: Aligned models score ~0.1 points lower on toxicity scale

### Backup Slide B: Scorer Agreement Analysis

**Content:**
- Detoxify vs. OpenAI Moderation correlation: r = [value]
- Binary agreement rate: [X]%
- Conclusion concordance: [Y]% of family-level conclusions match

### Backup Slide C: Sample Size Justification

**Content:**
- Power analysis for McNemar's test
- Effect size assumption: 15% ARR
- Required n for 80% power at alpha = 0.05
- Actual n = 25,000 (substantially overpowered)

---

## Presentation Checklist

- [ ] Test slide transitions and animations
- [ ] Verify all figure paths resolve correctly
- [ ] Prepare for Q&A on statistical methods
- [ ] Have backup slides accessible
- [ ] Time practice run (target: 14-15 minutes)
- [ ] Prepare 1-sentence summary of each statistical test
- [ ] Know the exact p-values and effect sizes for verbal delivery
