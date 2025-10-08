"""Generate diagnostic plots for recorded NeomFlex training episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_episode_analysis(episode_num: int, save_dir: str | Path = "results/data") -> plt.Figure:
    """Create a multi-panel analysis figure for a recorded training episode."""

    save_dir = Path(save_dir)
    csv_path = save_dir / f"episode_{episode_num}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Episode data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

    voltage_cols = [col for col in df.columns if col.startswith("voltage_bus_")]
    voltage_data = df[voltage_cols].values
    mean_voltage = np.mean(voltage_data, axis=0)
    min_voltage = np.min(voltage_data, axis=0)
    max_voltage = np.max(voltage_data, axis=0)

    ax1.plot(range(len(voltage_cols)), mean_voltage, "b-", label="Mean", linewidth=2)
    ax1.fill_between(
        range(len(voltage_cols)),
        min_voltage,
        max_voltage,
        color="b",
        alpha=0.2,
        label="Min/Max Range",
    )
    ax1.axhline(y=0.95, color="r", linestyle="--", label="Lower Limit")
    ax1.axhline(y=1.05, color="r", linestyle="--", label="Upper Limit")
    ax1.set_xlabel("Bus Number")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title(f"Voltage Profile - Episode {episode_num}")
    ax1.grid(True)
    ax1.legend()

    time_steps = range(len(df))
    load = df["load"]
    net_load = load - df["solar"]

    ax2.plot(time_steps, net_load, "k-", label="Net Load", linewidth=2)
    pv_cols = [col for col in df.columns if col.startswith("pv_") and col.endswith("_q")]
    for col in pv_cols:
        bus = col.split("_")[1]
        ax2.plot(time_steps, df[col], ":", label=f"PV {bus} VAR")

    ax2.set_xlabel("Time Step (30-min intervals)")
    ax2.set_ylabel("Power (MW/MVAR)")
    ax2.set_title("Net Load and Control Actions")
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1))

    ax3.plot(time_steps, net_load, "k-", label="Net Load", linewidth=2)
    ax3.plot(time_steps, load, "b-", label="Load", linewidth=2)

    soc_cols = [col for col in df.columns if col.startswith("bat_") and col.endswith("_soc")]
    for col in soc_cols:
        bus = col.split("_")[1]
        ax3.plot(time_steps, df[col], "--", label=f"Battery {bus} SOC")

    ax3.set_xlabel("Time Step (30-min intervals)")
    ax3.set_ylabel("Power (MW) / SOC (MWh)")
    ax3.set_title("Net Load and Battery SOC Profiles")
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=int, help="Episode number to visualise")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/data"),
        help="Directory containing recorded episode CSV files (default: results/data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated figure instead of displaying it.",
    )
    args = parser.parse_args()

    fig = plot_episode_analysis(args.episode, args.data_dir)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
