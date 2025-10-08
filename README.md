# NeomFlex

NeomFlex is a research playground for studying multi-agent reinforcement learning
strategies that operate active assets on a distribution grid. The project contains a
Pandapower-based simulation environment, a Soft Actor-Critic (SAC) implementation for
training decentralised agents, and tooling for analysing the resulting behaviour.

## Repository structure

```
├── data/                     # CSV demand/solar profiles for training and testing
├── docs/                     # Additional documentation
├── requirements.txt          # Python dependencies used by the tooling
├── scripts/                  # Command line utilities for evaluation and plotting
├── src/neomflex/             # Python package with environments and agents
└── README.md                 # You are here
```

Key modules:

- `src/neomflex/grid_environment.py` – defines `DistributionGridEnv`, a Gym-compatible
  environment that models the 33-bus distribution grid with PV and battery assets.
- `src/neomflex/agents/sac.py` – decentralised SAC agent implementation and support
  utilities such as data recording and training helpers.
- `scripts/evaluate_agents.py` – roll out trained agents against the testing dataset,
  optionally computing a zero-action baseline and exporting rich metrics for further
  analysis.
- `scripts/plot_*.py` – plotting helpers that consume the evaluation artefacts.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train agents**

   Training scripts are provided inside `src/neomflex/agents/sac.py` (see the
   `Trainer` class). Typical experiments will serialise model checkpoints to
   `results/final_results/agent_<id>.pt`. Ensure that each agent's checkpoint is
   saved under its identifier (e.g., `agent_1.pt`).

3. **Evaluate agents**

   ```bash
   python scripts/evaluate_agents.py \
       --checkpoints results/final_results \
       --data data/testing_profiles.csv \
       --output results/evaluation \
       --baseline
   ```

   The command generates three CSV files in the chosen output directory:

   - `step_metrics.csv` – per-timestep measurements including voltages, losses,
     battery state-of-charge, PV set-points, and agent rewards.
   - `daily_summary.csv` – aggregated statistics for each simulated day.
   - `run_summary.csv` – headline metrics comparing control and baseline runs.

4. **Visualise the results**

   The plotting utilities read the generated CSV files. For example:

   ```bash
   # Battery state of charge over time
   python scripts/plot_state_of_charge.py results/evaluation/step_metrics.csv --run controlled

   # Voltage envelope across the feeder
   python scripts/plot_voltage_profiles.py results/evaluation/step_metrics.csv --run controlled

   # PV reactive power usage
   python scripts/plot_reactive_power.py results/evaluation/step_metrics.csv --run controlled
   ```

   Use the `--output` flag on each command to save plots instead of displaying them
   interactively, and `--days` to focus on a subset of the evaluation horizon.

## Data

The repository ships with two datasets located under `data/`:

- `training_profiles.csv` – demand and solar trajectories used during training.
- `testing_profiles.csv` – out-of-sample profiles for evaluation.

Each CSV contains half-hourly measurements with columns `Demand` and `Solar`.

## Documentation

Additional guidance, design decisions, and experiment notes can be added to the
`docs/` directory. Start with `docs/architecture.md` to capture high-level ideas.

## License

The repository currently has no explicit licence. Add one before sharing results
publicly.
