# NeomFlex architecture overview

This document summarises the main components of the NeomFlex research codebase.

## Environment

`src/neomflex/grid_environment.py` implements `DistributionGridEnv`, a Gym-compatible
Pandapower environment derived from the IEEE 33-bus radial distribution system. The
environment exposes territories for three agents, each controlling a subset of PV and
battery assets. Observations include territorial voltages, demand/solar forecasts, local
state-of-charge, and a normalised time feature. Actions span continuous set-points for
battery power and PV reactive power.

Key responsibilities:

- Load half-hourly demand and solar profiles from CSV files located in `data/`.
- Maintain battery state-of-charge with configurable bounds and charge/discharge limits.
- Execute power flow simulations via Pandapower and expose resulting voltages/losses.
- Provide detailed logging hooks to monitor constraint violations and asset behaviour.

## Agents

`src/neomflex/agents/sac.py` contains a decentralised Soft Actor-Critic implementation
(`SACAgent`). The module also defines helper classes for experience replay, checkpoint
management, and optional data recording during training. Agents act independently but
share a consistent network architecture.

## Evaluation workflow

1. Train agents and persist checkpoints to `results/final_results/agent_<id>.pt`.
2. Use `scripts/evaluate_agents.py` to run the agents (and an optional baseline) on the
   testing dataset. The script records per-step metrics, daily aggregates, and overall
   summaries in CSV format.
3. Visualise the outputs with the plotting scripts under `scripts/`, all of which accept
   the generated `step_metrics.csv` as input.

## Data artefacts

- `data/training_profiles.csv` – training dataset.
- `data/testing_profiles.csv` – evaluation dataset.
- `results/evaluation/` – created by the evaluation script to store outputs.

Extend this document as the project evolves (e.g., to capture new agent types, reward
structures, or benchmarking methodologies).
