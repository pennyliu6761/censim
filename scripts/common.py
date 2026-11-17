"""Shared helpers for the pipeline scripts."""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    out = ROOT / cfg["output_dir"]
    for sub in ["data", "e1", "e2", "e3", "tables", "figures"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    cfg["_out"] = out
    cfg["_root"] = ROOT
    return cfg


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
