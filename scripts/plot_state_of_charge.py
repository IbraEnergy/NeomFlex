"""Plot battery state of charge trajectories from evaluation artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOC_COLUMN_PREFIX = "battery_soc_bus_"


def _filter_days(df: pd.DataFrame, days: Iterable[int] | None) -> pd.DataFrame:
    if days is None:
        return df
    days_set = set(days)
    return df[df["day"].isin(days_set)]


def plot_state_of_charge(
    step_metrics_path: Path,
    run_label: str = "controlled",
    days: Iterable[int] | None = None,
    output_path: Path | None = None,
) -> None:
    if not step_metrics_path.exists():
        raise FileNotFoundError(f"Step metrics file not found at {step_metrics_path}")

    df = pd.read_csv(step_metrics_path)
    df = df[df["run"] == run_label]
    df = _filter_days(df, days)
    df = df.sort_values(["day", "step"])  # Ensure chronological order

    soc_columns = [col for col in df.columns if col.startswith(SOC_COLUMN_PREFIX)]
    if not soc_columns:
        raise ValueError(
            "The step metrics file does not contain battery SOC columns. "
            "Ensure the evaluation was executed with a version that records SOC data."
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    total_steps = len(df)
    x_values = np.arange(total_steps)

    for col in soc_columns:
        bus_id = col.removeprefix(SOC_COLUMN_PREFIX)
        ax.plot(x_values, df[col], label=f"Bus {bus_id}")

    ax.set_xlabel("Timestep (30-minute intervals)")
    ax.set_ylabel("State of charge (MWh)")
    ax.set_title(f"Battery SOC trajectories – {run_label}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", ncol=min(len(soc_columns), 3))
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
        help="Name of the run to visualise (e.g. 'controlled' or 'baseline').",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        help="Optional list of day indices to include in the plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot instead of showing it interactively.",
    )
    args = parser.parse_args()

    plot_state_of_charge(args.step_metrics, args.run, args.days, args.output)


if __name__ == "__main__":
    main()
