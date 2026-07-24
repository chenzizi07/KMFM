# Spectral Teacher V7 OOF Audit

This is a training-region diagnostic. It does not load or use test labels to
select an encoder or unlock the next method.

## Why V7 exists

OASD-v6 trained its OOF model with `spatial_only` as the primary task and used
the auxiliary spectral head as the teacher. V7 removes that asymmetry: every
OOF fold trains one spatial-primary model and one spectral-primary model from
the same fold and random seed.

## Colab command

```bash
cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
python scripts/audit_spectral_teacher_v7.py \
  --dataset pavia_university \
  --protocol spatial_block \
  --experiment pavia_spectral_teacher_audit_v7 \
  --seeds 0,1,2,3,4 \
  --encoders mlp,conv1d \
  --repeats 3 \
  --folds 3 \
  --epochs 60 \
  --recover-incomplete
```

Successful encoder/seed audits are immutable and skipped on restart. With
`--recover-incomplete`, an interrupted directory is archived under
`audits/_incomplete/` before it is recomputed.

## Fixed gate

- A class is stable when spectral advantage is positive in at least two of
  three repeats, its mean advantage is at least 5 percentage points, and its
  worst repeat is no worse than -5 percentage points.
- A seed passes with at least three stable classes and positive stable
  instance-level net corrections.
- An encoder passes with at least four of five passing seeds, positive mean
  stable net corrections, and mean global spectral OOF OA no more than 5
  percentage points below spatial.
- `DEVELOPMENT_GO` unlocks instance-level residual distillation.
- `DEVELOPMENT_NO_GO` terminates teacher distillation and redirects development
  to spatial-primary spectral self-supervision.

## Outputs

```text
audits/pavia_spectral_teacher_audit_v7/
reports/pavia_spectral_teacher_audit_v7/
  per_seed.csv
  per_class.csv
  spectral_teacher_audit_decision.json
  spectral_teacher_audit_decision.md
```
