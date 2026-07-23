# KMFM: Colab + Google Drive Setup

This project follows the local -> GitHub -> Colab workflow used by
`Evidence-Mamba-HSI` (SASM-Mamba).

## Fixed Google Drive Layout

```text
/content/drive/MyDrive/Colab/
|-- Unsupervised/
|   `-- KMFM/
|       |-- configs/
|       |-- notebooks/
|       |-- scripts/
|       |-- src/
|       |-- splits/       # generated, not committed
|       |-- results/      # generated, not committed
|       `-- reports/      # generated, not committed
`-- Datasets/
    |-- PaviaU.mat
    |-- PaviaU_gt.mat
    |-- Houston_data.mat
    |-- Houston_gt.mat
    |-- KSC.mat
    |-- KSC_gt.mat
    |-- Botswana.mat
    `-- Botswana_gt.mat
```

## First-Time Install

Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone and install from the public GitHub repository:

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised
!REPO_URL=https://github.com/chenzizi07/KMFM.git bash -c "$(curl -fsSL https://raw.githubusercontent.com/chenzizi07/KMFM/main/scripts/colab_install.sh)"
```

Alternatively, open `notebooks/KMFM_LASSF_Colab.ipynb` and run its first two
cells. The notebook can read the same Colab Secret names already used for the
SASM-Mamba workflow if repository authentication is required.

## Update Existing Copy

After code changes are pushed from the local machine, run:

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh
```

The update is fast-forward only. Generated `splits/`, `results/`, and
`reports/` remain on Google Drive and are excluded from Git.

## Verify Before Real Experiments

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!python scripts/smoke_test.py
!pytest -q
```

Then run `notebooks/KMFM_LASSF_Colab.ipynb`. Start with the two-dataset,
two-seed pilot described in the notebook before launching the full four-dataset
experiment matrix.

## Local Push

The local repository uses the same GitHub SSH key configuration as
`E:\codex\Evidence-Mamba-HSI`:

```powershell
git add .
git commit -m "Describe the KMFM change"
git push
```

Do not put a GitHub token in a notebook cell, remote URL, or committed file.
