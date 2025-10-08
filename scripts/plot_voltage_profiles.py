"""Visualise voltage envelope across the network from evaluation artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VOLTAGE_PREFIX = "voltage_bus_"


def _filter(df: pd.DataFrame, run_label: str, days: Iterable[int] | None) -> pd.DataFrame:
    df = df[df["run"] == run_label]
    if days is not None:
        df = df[df["day"].isin(set(days))]
    return df.sort_values(["day", "step"])


def plot_voltage_envelope(
    step_metrics_path: Path,
    run_label: str = "controlled",
    days: Iterable[int] | None = None,
    output_path: Path | None = None,
) -> None:
    if not step_metrics_path.exists():
        raise FileNotFoundError(f"Step metrics file not found at {step_metrics_path}")

    df = pd.read_csv(step_metrics_path)
    df = _filter(df, run_label, days)

    voltage_cols = [col for col in df.columns if col.startswith(VOLTAGE_PREFIX) and col.endswith("_pu")]
    if not voltage_cols:
        raise ValueError(
            "The step metrics file does not contain per-bus voltage columns. "
            "Re-run evaluation with the latest tooling."
        )

    values = df[voltage_cols].to_numpy()
    maxima = values.max(axis=1)
    minima = values.min(axis=1)
    means = values.mean(axis=1)

    x_values = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x_values, maxima, minima, alpha=0.4, color="lightblue", label="Envelope")
    ax.plot(x_values, means, color="navy", linewidth=0.8, label="Mean voltage")

    ax.axhline(1.05, color="red", linestyle="--", linewidth=0.8, label="Upper limit")
    ax.axhline(0.95, color="red", linestyle="--", linewidth=0.8, label="Lower limit")

    ax.set_xlabel("Timestep (30-minute intervals)")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage envelope – {run_label}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
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

    plot_voltage_envelope(args.step_metrics, args.run, args.days, args.output)


if __name__ == "__main__":
    main()
