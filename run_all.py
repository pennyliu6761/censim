"""Runs the whole pipeline in order. Usage: python run_all.py --config configs/default.yaml"""
import subprocess
import sys
from pathlib import Path

STEPS = ["record_environment", "s00_prepare_data", "s01_run_e1", "s02_run_e2", "s03_run_e3", "s04_make_tables",
         "s05_make_figures", "s06_make_flowcharts", "s07_run_e1_extra", "s08_run_e2_explore", "s09_run_e2_hyper",
         "s10_make_paper_figures", "s11_dataset_profile"]

if __name__ == "__main__":
    # usage: python run_all.py --config configs/default.yaml [--from s03_run_e3]
    args = sys.argv[1:] or ["--config", "configs/default.yaml"]
    start = None
    if "--from" in args:
        i = args.index("--from"); start = args[i + 1]; args = args[:i] + args[i + 2:]
    steps = STEPS if start is None else ["record_environment"] + STEPS[STEPS.index(start):]
    root = Path(__file__).resolve().parent
    for s in steps:
        print(f"\n===== {s} =====", flush=True)
        r = subprocess.run([sys.executable, str(root / "scripts" / f"{s}.py"), *args], cwd=root)
        if r.returncode != 0:
            sys.exit(f"step {s} failed (exit code {r.returncode})")
    print("\nAll steps finished.")
