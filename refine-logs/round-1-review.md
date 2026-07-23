# Round 1 Method Review

> Reviewer note: the configured external GPT-5.5 review tool was unavailable in this environment. This is an explicitly labeled local fallback audit, not an independent external review.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10 |
| Method Specificity | 9 |
| Contribution Quality | 7 |
| Frontier Leverage | 8 |
| Feasibility | 10 |
| Validation Focus | 9 |
| Venue Readiness | 6 |
| **Overall** | **8.45** |

**Verdict: REVISE**

## Findings

1. **CRITICAL - validation degrees of freedom**: two branch temperatures, `alpha0`, `rho`, and another output temperature would be too much for 90 validation samples. Remove the final temperature and keep the search grids fixed before test evaluation.
2. **IMPORTANT - checkpoint dependence**: a concat-trained source checkpoint is not a neutral two-branch learner. Keep it fixed before replay, describe replay only as a feasibility gate, and require a later dual-branch training run if the gate passes.
3. **IMPORTANT - novelty risk**: confidence-weighted decision fusion is established broadly. The defensible distinction is the explicit global anchor, disagreement-only bounded residual, and spatially disjoint HSI evidence. A formal novelty search remains mandatory before manuscript claims.
4. **MINOR - metric semantics**: routing AUC alone can again be misleading. Report disagreement-set gain, fraction improved/harmed, and OA relative to both global and spatial references.

## Simplification Opportunities

- Remove class-wise weights and neural routers from the pilot.
- Remove final output temperature; use only independently fitted branch temperatures shared by all replay variants.
- Fix the source checkpoint and parameter grids in code rather than accepting arbitrary search ranges.

## Modernization Opportunities

NONE. A foundation-model component is not natural for this small-label, fixed-scene diagnostic and would create contribution sprawl.

## Drift Warning

NONE. The proposal still targets the observed mismatch between branch reliability and final fused decisions.
