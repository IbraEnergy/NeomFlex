"""Plot PV reactive power utilisation from evaluation artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PV_PREFIX = "pv_bus_"


def _filter(df: pd.DataFrame, run_label: str, days: Iterable[int] | None) -> pd.DataFrame:
    df = df[df["run"] == run_label]
    if days is not None:
        df = df[df["day"].isin(set(days))]
    return df.sort_values(["day", "step"])


def plot_reactive_power(
    step_metrics_path: Path,
    run_label: str = "controlled",
    days: Iterable[int] | None = None,
    output_path: Path | None = None,
) -> None:
    if not step_metrics_path.exists():
        raise FileNotFoundError(f"Step metrics file not found at {step_metrics_path}")

    df = pd.read_csv(step_metrics_path)
    df = _filter(df, run_label, days)

    q_columns = [col for col in df.columns if col.startswith(PV_PREFIX) and col.endswith("_q_mvar")]
    if not q_columns:
        raise ValueError(
            "The step metrics file does not contain PV reactive power columns."
            " Re-run evaluation with the latest tooling."
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    x_values = np.arange(len(df))

    for col in q_columns:
        bus_id = col.removeprefix(PV_PREFIX).removesuffix("_q_mvar")
        ax.plot(x_values, df[col], linewidth=0.8, label=f"PV bus {bus_id}")

    ax.set_xlabel("Timestep (30-minute intervals)")
    ax.set_ylabel("Reactive power (MVAR)")
    ax.set_title(f"PV reactive power set-points – {run_label}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", ncol=min(len(q_columns), 3))
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step_metrics",
        type=Path,
        default=Path("results/evaluation/step_metrics.csv"),
        nargs="?",
        help="CSV file produced by scripts/evaluate_agents.py (default: results/evaluation/step_metrics.csv).",
    )
    parser.add_argument(
        "--run",
        default="controlled",
        help="Run label to visualise (default: controlled).",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        help="Optional subset of day indices to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure.",
    )
    args = parser.parse_args()

    plot_reactive_power(args.step_metrics, args.run, args.days, args.output)


if __name__ == "__main__":
    main()
