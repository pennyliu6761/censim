# censim

Code for the paper *Spiral-down and lock-in of data-driven newsvendor policies under endogenous demand censoring: A closed-loop simulation analysis*.

The code runs the full study in one command: data preparation, the synthetic experiment (E1), the real-data experiment on FreshRetailNet-50K (E2), the survival analysis (E3), and all figures and tables of the paper.

## Requirements

- CPU only; no GPU is needed.
- 4 or more cores and 16 GB of RAM are recommended (the smoke configuration runs with 8 GB).
- About 3 GB of free disk space.
- Python 3.13, managed by [uv](https://docs.astral.sh/uv/). Tested on Windows 11 and Linux.

## Setup

Install uv once per machine:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, in the project folder:

```bash
uv sync           # creates .venv with the exact package versions in uv.lock
uv run pytest -q  # 8 verification tests; should report "8 passed"
```

Without uv: `python -m venv .venv`, activate it, then `pip install -r requirements.txt` and `pip install -e .`.

## Data

Download `data/train.parquet` (about 100 MB) of [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) and save it as `data/train.parquet` in this project.

## Running

Check the pipeline on a small sample first (6 stores, about 2 minutes):

```bash
uv run python run_all.py --config configs/smoke.yaml
```

Full run used for the paper:

```bash
uv run python run_all.py --config configs/default.yaml
```

A single step can be run on its own, for example `uv run python scripts/s10_make_paper_figures.py --config configs/default.yaml`. To resume from a step, add `--from s03_run_e3` to the `run_all.py` command. The E2 steps save each job as it finishes, so an interrupted run skips the completed jobs when restarted.

| Step | Content | Approximate time (full run) |
| --- | --- | --- |
| s00 | Sample 200 stores, build latent demand (versions A and B), hourly profile | < 1 min |
| s01 | E1: synthetic experiment | 5–15 min |
| s02 | E2: real-data experiment | 2–4 h |
| s03 | E3: failure and recovery, Cox models | < 1 min |
| s04 | Tables | < 1 min |
| s05 | Hourly profile and trajectory figures | < 1 min |
| s06 | Flowcharts (Figs. 1 and 2) | < 1 min |
| s07 | E1: window, refit interval and exploration step | 10–30 min |
| s08 | E2: exploration (QNTX, MSEX) | about 1/3 of s02 |
| s09 | E2: gradient boosting hyperparameters | about 40% of s02 |
| s10 | Remaining figures and tables | < 2 min |
| s11 | Dataset profile (Fig. 4) | about 1 min |

`record_environment` runs first and writes the Python and package versions to `ENVIRONMENT.txt`.

## Outputs

All outputs go to `results/` (`results_smoke/` for the smoke configuration). Each figure is saved as PNG (300 dpi) and PDF, 6.2 inches wide with 10 pt text; insert it at that width.

**Figures**

| Paper | File in `results/figures/` | Step |
| --- | --- | --- |
| Fig. 1 | `fig01_closed_loop` | s06 |
| Fig. 2 | `fig02_ground_truth` | s06 |
| Fig. 3 | `fig03_hourly_profile` | s05 |
| Fig. 4 | `fig04_dataset_profile` | s11 |
| Fig. 5 | `fig04_synthetic_trajectories` | s05 |
| Fig. 6 | `fig05_e1_cost_heatmap` | s10 |
| Fig. 7 | `fig06_exploration_tradeoff` | s10 |
| Fig. 8 | `fig07_real_trajectories` | s05 |
| Fig. 9 | `fig08_cost_decomposition` | s10 |
| Fig. 10 | `fig09_sku_order_ratio` | s10 |
| Fig. 11 | `fig10_recovery_curves` | s05 |
| Fig. 12 | `fig11_cox_hazard_ratios` | s10 |
| Fig. S1 | `figA1_example_series` | s10 |
| Fig. S2 | `fig13_robustness` | s10 |
| Fig. S3 | `fig12_real_exploration` | s10 |

**Tables**

| Paper | File in `results/tables/` |
| --- | --- |
| Table 4 | `tableA1_prop2_theory_vs_simulation.csv` |
| Table 5, Table S1 | `tableA2_e1_cost_ratio.csv` |
| Table 6 | `table4_e2_cost_decomposition.csv` |
| Table 7 | `tableA6_e2_store_sku_distribution.csv` |
| Table 8, Table S5 | `tableA7_failure_thresholds.csv` |
| Table 9 | `table5_e2_exploration.csv` |
| Table 10 | `tableA9_robustness.csv` |
| Table S2 | `tableA3_e1_order_ratio.csv` |
| Table S3 | `tableA4_e1_window_refit.csv` |
| Table S4 | `tableA5_e1_exploration.csv` |
| Table S6 | `tableA8_cox.csv` |
| Dataset profile (Section 4.2) | `dataset_profile_summary.json`, `dataset_profile_series.csv` |

Tables 1–3 of the paper are descriptive. The learner settings in Table S7 are those in `configs/default.yaml`.

## Project layout

```
configs/      default.yaml (paper), smoke.yaml (quick check)
data/         place train.parquet here (not tracked)
scripts/      pipeline steps s00–s11 and shared helpers
src/censim/   newsvendor cost, learners, simulator, ground truth, theory, analysis, plotting
tests/        verification tests (Section 4.4 of the paper)
run_all.py    runs all steps in order
uv.lock       exact package versions used for the paper
```

## Reproducibility notes

- All learners within a demand setting see the same demand paths (common random numbers).
- Features for day t use only data from before day t.
- XGBoost results can differ in the last digits across CPUs and thread counts; the conclusions are not affected.
- Do not run `uv lock --upgrade` if you want to reproduce the published numbers.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

To be added.
