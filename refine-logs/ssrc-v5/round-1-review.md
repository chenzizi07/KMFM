# Round 1 Local Method Audit

External GPT-5.5 reviewer tooling required by `research-refine` was unavailable in this environment. This is an explicitly labeled local fallback audit, not an external review.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 9.0 |
| Contribution Quality | 8.5 |
| Frontier Leverage | 7.0 |
| Feasibility | 9.0 |
| Validation Focus | 8.5 |
| Venue Readiness | 7.0 |
| **Overall** | **8.53** |

**Verdict: REVISE pending empirical development replay**

## Blocking Concerns

1. Pavia test has already informed the direction, so it cannot confirm the method.
2. Immediate internal backbone OOF multiplies training cost before the low-capacity selector has shown any value.
3. The source checkpoint used validation for early stopping; reusing the same validation for selector fitting limits confirmatory strength.
4. Class features may overfit unless the class-aware candidate is compared with a score-only deletion check.

## Required Revision

- Insert a checkpoint-only development gate using validation-fold OOF selector scores.
- Keep all features generic; do not encode observed Pavia class identities.
- Require spatial fallback and a positive Wilson lower bound before any correction.
- Treat a Pavia pass only as authorization for formal train-region OOF on a fresh dataset.

## Simplification Opportunities

- Use ridge utility regression rather than an MLP gate.
- Use separate spatial/spectral class one-hot vectors rather than 81 class-pair interactions.
- Limit correction coverage to four fixed candidates.

## Drift Warning

NONE. The revision still targets conversion of measured branch complementarity into stable OA gain.
