# File Guide

This guide summarises where to find the code that was added during the recent
refactor.

## Core package (`src/neomflex`)

All of the Python source files that implement the environment and agents now
live inside the `src/neomflex/` package. On GitHub you can browse to these
modules by opening the `src` directory and then the `neomflex` folder. Key
entries include:

- [`src/neomflex/grid_environment.py`](../src/neomflex/grid_environment.py)
  – contains the `DistributionGridEnv` class that wraps the Pandapower model.
- [`src/neomflex/agents/sac.py`](../src/neomflex/agents/sac.py) – holds the SAC
  implementation, replay buffer, and training utilities.

If you do not see the `src` directory on GitHub, make sure you are looking at
the branch where the refactor was committed.

## Command line tooling (`scripts`)

Evaluation and plotting scripts now live in the top-level `scripts/` folder.
When browsing on GitHub select the `scripts` directory to inspect files such as:

- [`scripts/evaluate_agents.py`](../scripts/evaluate_agents.py) – runs rollouts
  for trained checkpoints and exports metrics.
- [`scripts/analyze_results.py`](../scripts/analyze_results.py) – aggregates the
  metrics into daily and run-level summaries.
- [`scripts/plot_state_of_charge.py`](../scripts/plot_state_of_charge.py),
  [`scripts/plot_voltage_profiles.py`](../scripts/plot_voltage_profiles.py), and
  [`scripts/plot_reactive_power.py`](../scripts/plot_reactive_power.py) – helper
  commands for visualisation.

## Additional documentation (`docs`)

Architecture notes and this file guide live under `docs/`. Browse to that
folder to view [`docs/architecture.md`](architecture.md) and any future design
write-ups.

## Datasets (`data`)

The CSV demand and solar profiles are located in the `data/` directory at the
repository root. These files were not changed by the refactor but are required
when running the new scripts.

## Troubleshooting

If the GitHub web interface still does not display the new files, double check
that:

1. The latest commit has been pushed to the remote repository.
2. You are viewing the correct branch in the GitHub branch picker.
3. Your browser cache is not serving a stale version of the file tree.

Alternatively, clone the repository locally and run `git status` or `git log`
to confirm that the refactor commits are present.
