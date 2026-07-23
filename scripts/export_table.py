from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = str(value)
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an aggregate metric table without manual values")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--protocol", default="spatial_block")
    parser.add_argument("--metric", choices=("oa", "aa", "kappa"), default="oa")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    frame = frame[frame["protocol"] == args.protocol].copy()
    if frame.empty:
        raise ValueError(f"No rows for protocol {args.protocol!r}")
    mean_column = f"{args.metric}_mean_percent"
    sd_column = f"{args.metric}_sd_percent"
    required = {"dataset", "model", mean_column, sd_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")

    datasets = sorted(frame["dataset"].unique())
    models = sorted(frame["model"].unique())
    formatted = pd.DataFrame(index=datasets, columns=models, dtype=object)
    numeric = pd.DataFrame(index=datasets, columns=models, dtype=float)
    for _, row in frame.iterrows():
        mean = float(row[mean_column])
        sd = float(row[sd_column])
        text = f"{mean:.2f} ± {sd:.2f}" if np.isfinite(sd) else f"{mean:.2f}"
        formatted.loc[row["dataset"], row["model"]] = text
        numeric.loc[row["dataset"], row["model"]] = mean

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(prefix.with_suffix(".csv"), index_label="dataset")
    prefix.with_suffix(".md").write_text(formatted.to_markdown(index=True) + "\n", encoding="utf-8")

    latex_rows = []
    for dataset in datasets:
        best = numeric.loc[dataset].max(skipna=True)
        cells = [_latex_escape(dataset)]
        for model in models:
            value = formatted.loc[dataset, model]
            if pd.isna(value):
                cells.append("--")
            elif np.isclose(numeric.loc[dataset, model], best, rtol=0, atol=1e-12):
                cells.append(r"\textbf{" + str(value).replace("±", r"$\pm$") + "}")
            else:
                cells.append(str(value).replace("±", r"$\pm$"))
        latex_rows.append(" & ".join(cells) + r" \\")
    column_spec = "l" + "c" * len(models)
    header = "Dataset & " + " & ".join(_latex_escape(model) for model in models) + r" \\"
    latex = "\n".join(
        [
            r"\begin{tabular}{" + column_spec + "}",
            r"\toprule",
            header,
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    prefix.with_suffix(".tex").write_text(latex, encoding="utf-8")
    print(formatted.to_string())
    print(f"Saved {prefix.with_suffix('.csv')}, .md and .tex")


if __name__ == "__main__":
    main()
