"""Evaluate trained NeomFlex agents on hold-out demand/solar profiles."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from neomflex import DistributionGridEnv
from neomflex.agents import SACAgent

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    data_path: Path
    checkpoint_dir: Path
    output_dir: Path
    days: Iterable[int] | None
    include_baseline: bool


def _resolve_steps_per_day(env: DistributionGridEnv) -> int:
    """Infer the number of time steps in a day from the dataset."""

    if env.total_days == 0:
        msg = "Dataset must contain at least one full day of measurements"
        raise ValueError(msg)
    return len(env.data) // env.total_days


def _load_agents(env: DistributionGridEnv, checkpoint_dir: Path) -> Dict[str, SACAgent]:
    """Instantiate agents with the correct network shapes and load checkpoints."""

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")

    agents: Dict[str, SACAgent] = {}
    for agent_id in env.territories:
        state_dim = env.observation_spaces[agent_id].shape[0]
        action_dim = env.action_spaces[agent_id].shape[0]
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

        checkpoint_path = checkpoint_dir / f"{agent_id}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for {agent_id} not found at {checkpoint_path}."
                " Ensure the training run produced per-agent '.pt' files."
            )

        logger.info("Loading checkpoint for %s from %s", agent_id, checkpoint_path)
        agent.load(checkpoint_path)
        agents[agent_id] = agent

    return agents


def _agent_action_fn(agents: Dict[str, SACAgent]) -> Callable[[DistributionGridEnv, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Create an action function that queries the trained agents."""

    def action_fn(env: DistributionGridEnv, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            agent_id: agents[agent_id].select_action(observations[agent_id], evaluate=True)
            for agent_id in agents
        }

    return action_fn


def _baseline_action_fn(env: DistributionGridEnv, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Always return zero actions, representing no control intervention."""

    return {
        agent_id: np.zeros(env.action_spaces[agent_id].shape[0], dtype=np.float32)
        for agent_id in env.territories
    }


def _simulate_day(
    env: DistributionGridEnv,
    day_index: int,
    run_label: str,
    action_fn: Callable[[DistributionGridEnv, Dict[str, np.ndarray]], Dict[str, np.ndarray]],
    steps_per_day: int,
) -> pd.DataFrame:
    """Roll out one day in the environment using the provided action function."""

    env.current_day = day_index % env.total_days
    observations = env.reset()
    records = []

    for step in range(steps_per_day):
        current_step_index = env.time_step
        actions = action_fn(env, observations)
        observations, rewards, done, info = env.step(actions)

        voltages = env.net.res_bus.vm_pu.values.astype(float)
        losses = float(env.net.res_line.pl_mw.sum())

        load = float(env.data.iloc[current_step_index]["Demand"])
        solar = float(env.data.iloc[current_step_index]["Solar"])
        voltage_min = float(np.nanmin(voltages))
        voltage_max = float(np.nanmax(voltages))
        voltage_mean = float(np.nanmean(voltages))
        violation_count = int(
            np.sum((voltages < env.v_limits[0]) | (voltages > env.v_limits[1]) | np.isnan(voltages))
        )

        record = {
            "run": run_label,
            "day": int(day_index),
            "step": int(step),
            "dataset_index": int(current_step_index),
            "load_factor": load,
            "solar_factor": solar,
            "losses_mw": losses,
            "voltage_min_pu": voltage_min,
            "voltage_max_pu": voltage_max,
            "voltage_mean_pu": voltage_mean,
            "voltage_violations": violation_count,
        }

        for agent_id, reward in rewards.items():
            record[f"reward_{agent_id}"] = float(reward)

        for bus, soc in env.bat_soc.items():
            record[f"battery_soc_bus_{bus}"] = float(soc)

        for bus, idx in env.bat_sgen_indices.items():
            record[f"battery_power_bus_{bus}_mw"] = float(env.net.sgen.at[idx, "p_mw"])

        for bus, idx in env.pv_sgen_indices.items():
            record[f"pv_bus_{bus}_p_mw"] = float(env.net.sgen.at[idx, "p_mw"])
            record[f"pv_bus_{bus}_q_mvar"] = float(env.net.sgen.at[idx, "q_mvar"])

        for bus_index, value in enumerate(voltages):
            record[f"voltage_bus_{bus_index}_pu"] = float(value)

        if any(done.values()):
            record["terminated_early"] = True
            records.append(record)
            logger.debug("Run %s terminated after %s steps", run_label, step + 1)
            break

        records.append(record)

    return pd.DataFrame.from_records(records)


def run_evaluation(config: EvaluationConfig) -> dict[str, pd.DataFrame]:
    """Execute evaluation and return raw and aggregated metrics."""

    env = DistributionGridEnv(data_path=config.data_path)
    steps_per_day = _resolve_steps_per_day(env)

    agents = _load_agents(env, config.checkpoint_dir)
    days = list(config.days) if config.days is not None else list(range(env.total_days))

    logger.info("Evaluating %s days using dataset %s", len(days), config.data_path)

    step_frames: list[pd.DataFrame] = []

    for day in days:
        logger.info("Simulating day %s with control", day)
        step_frames.append(
            _simulate_day(env, day, run_label="controlled", action_fn=_agent_action_fn(agents), steps_per_day=steps_per_day)
        )

        if config.include_baseline:
            logger.info("Simulating day %s with baseline actions", day)
            baseline_env = DistributionGridEnv(data_path=config.data_path)
            baseline_env.current_day = day % baseline_env.total_days
            step_frames.append(
                _simulate_day(
                    baseline_env,
                    day,
                    run_label="baseline",
                    action_fn=_baseline_action_fn,
                    steps_per_day=steps_per_day,
                )
            )

    step_metrics = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()

    if step_metrics.empty:
        raise RuntimeError("No evaluation data was generated. Check the provided day indices.")

    agent_reward_cols = [col for col in step_metrics.columns if col.startswith("reward_")]
    summary = (
        step_metrics.groupby(["run", "day"], as_index=False)
        .agg(
            min_voltage_pu=("voltage_min_pu", "min"),
            max_voltage_pu=("voltage_max_pu", "max"),
            mean_voltage_pu=("voltage_mean_pu", "mean"),
            total_losses_mw=("losses_mw", "sum"),
            voltage_violations=("voltage_violations", "sum"),
            **{f"total_reward_{col.split('_', 1)[1]}": (col, "sum") for col in agent_reward_cols},
        )
    )

    overall = (
        step_metrics.groupby("run", as_index=False)
        .agg(
            min_voltage_pu=("voltage_min_pu", "min"),
            max_voltage_pu=("voltage_max_pu", "max"),
            mean_voltage_pu=("voltage_mean_pu", "mean"),
            total_losses_mw=("losses_mw", "sum"),
            voltage_violations=("voltage_violations", "sum"),
            **{f"total_reward_{col.split('_', 1)[1]}": (col, "sum") for col in agent_reward_cols},
        )
    )

    return {"steps": step_metrics, "daily_summary": summary, "run_summary": overall}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("results/final_results"),
        help="Directory containing per-agent '.pt' checkpoint files (default: results/final_results).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/testing_profiles.csv"),
        help="CSV file with demand and solar profiles used for evaluation (default: data/testing_profiles.csv).",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        help="Specific day indices to evaluate. When omitted all available days are simulated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory where evaluation artefacts will be written (default: results/evaluation).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also simulate a zero-action baseline for comparison.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the evaluation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    config = EvaluationConfig(
        data_path=args.data,
        checkpoint_dir=args.checkpoints,
        output_dir=args.output,
        days=args.days,
        include_baseline=args.baseline,
    )

    logger.info("Starting evaluation with configuration: %s", config)
    results = run_evaluation(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results["steps"].to_csv(config.output_dir / "step_metrics.csv", index=False)
    results["daily_summary"].to_csv(config.output_dir / "daily_summary.csv", index=False)
    results["run_summary"].to_csv(config.output_dir / "run_summary.csv", index=False)

    logger.info("Evaluation complete. Artefacts written to %s", config.output_dir)


if __name__ == "__main__":
    main()
