"""Writes ENVIRONMENT.txt: the machine and package versions that produced the results.
Called automatically by run_all.py; the file is overwritten on every full run."""
import datetime
import importlib.metadata as md
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGS = ["numpy", "scipy", "pandas", "pyarrow", "xgboost", "statsmodels", "matplotlib", "joblib", "PyYAML"]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "not a git repository"


def main(config="(not given)"):
    lines = [
        f"Run date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Config: {config}",
        f"Git commit: {git_commit()}",
        f"OS: {platform.platform()}",
        f"CPU: {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores",
        f"Python: {platform.python_version()}",
        "",
        "Packages:",
    ]
    for p in PKGS:
        try:
            lines.append(f"  {p}=={md.version(p)}")
        except md.PackageNotFoundError:
            lines.append(f"  {p}: not installed")
    (ROOT / "ENVIRONMENT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    main(sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "(not given)")
