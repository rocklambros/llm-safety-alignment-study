# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Exploratory Data Analysis section with density plot of base model toxicity scores, prompt count breakdown (RTP vs ToxiGen with 13 demographic groups), and base toxicity rate comparison across families
- Assumption Checks section covering McNemar discordant pair counts, Wilcoxon paired-difference symmetry plots, and Cochran's Q binary validation
- Visible Cochran's Q display chunk with kable output in Results section

### Changed
- Exposed R code in McNemar (table1) and Wilcoxon (table2) results chunks via `echo=TRUE`
- Rewrote connective prose throughout paper in casual grad-student tone while preserving all statistical terminology, numeric values, and references
