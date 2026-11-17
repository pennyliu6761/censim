"""Step 5: data figures that need only steps 0-4: hourly profile, synthetic
trajectories, real-data trajectories and recovery curves (tau = 0.9).
The flowcharts are drawn by step 6 and the remaining figures by steps 10 and 11 (see README.md)."""
import json
import pandas as pd
from common import load_config, log
from censim import plotting as pl


def main():
    cfg = load_config()
    o = cfg["_out"]
    f = o / "figures"
    pl.fig_hourly_profile(pd.read_csv(o / "data" / "hourly_profile.csv"), f / "fig03_hourly_profile.png")
    traj = json.load(open(o / "e1" / "e1_trajectories.json"))
    e1 = cfg["e1"]
    cv = 0.4 if 0.4 in e1["cvs"] else e1["cvs"][0]
    pl.fig_synthetic_trajectories(traj, e1["learners"], [t for t in e1["taus"] if t in (0.5, 0.9)] or e1["taus"],
                                  f / "fig04_synthetic_trajectories.png", start=min(e1["starts"]), cv=cv)
    wk = json.load(open(o / "tables" / "fig07_weekly_input.json"))
    weekly = {(float(k.split("|")[0]), k.split("|")[1]): v for k, v in wk.items()}
    taus = [t for t in cfg["e2"]["taus"] if t in (0.5, 0.9)] or cfg["e2"]["taus"]
    pl.fig_real_trajectories(weekly, taus, f / "fig07_real_trajectories.png")
    km = pd.read_csv(o / "e3" / "e3_km_escape.csv")
    pl.fig_escape_curves(km[km["tau"] == max(cfg["e2"]["taus"])], f / "fig10_recovery_curves.png")
    log("figures written to " + str(f))


if __name__ == "__main__":
    main()
