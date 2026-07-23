# Refinement Report

The third-round failure was traced to a semantic mismatch: auxiliary branch entropy routed features, while a separate classifier produced the final decision. The refined pilot directly mixes branch logits, anchors dynamic routing to a validation-optimal global weight, and restricts sample-specific changes to disagreement cases with radius at most 0.15.

## Outputs

- `PROBLEM_ANCHOR.md`
- `round-0-initial-proposal.md`
- `round-1-review.md`
- `round-1-refinement.md`
- `FINAL_PROPOSAL.md`
- `REVIEW_SUMMARY.md`

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 10 | 9 | 7 | 8 | 10 | 9 | 6 | 8.45 | REVISE |

## Remaining Weaknesses

- The replay can validate the decision interface but cannot prove a final trainable method.
- Formal novelty checking is required before writing a contribution claim.
- A passing replay must be followed by neutral dual-branch training and cross-fitted router supervision.
