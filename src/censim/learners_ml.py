"""Gradient boosting learners for the real-data experiment (E2).

  MSE  reg:squarederror on log(Y + eps), normal residual sd  -> parametric, censoring-blind
  QNT  reg:quantileerror at tau on Y                          -> quantile, censoring-blind
  AFT  survival:aft (normal, log scale), censored intervals    -> parametric, censoring-aware
  QNTX:eps / MSEX:eps  the same model, but the order is raised by (1 + eps) for a series whose
       stockout share over the last 7 days exceeds 1 - tau (exploration; same rule as SAAX in E1)
Each learner also runs as its own full-information benchmark (same loss, trained on true demand).
All learners share the same features and boosting settings; only the training objective differs.
"""
import numpy as np
import xgboost as xgb
from scipy import optimize, stats

EPS = 0.1


class FeatureBuilder:
    """Features for day t, using only information available before t (no leakage)."""

    def __init__(self, panel):
        self.p = panel

    def __call__(self, Y, C, t):
        p = self.p
        lag7 = Y[t - 7] if t >= 7 else Y[:t].mean(0)
        return np.column_stack([
            p["store"], p["product"],
            Y[max(0, t - 7):t].mean(0), Y[max(0, t - 28):t].mean(0), lag7,
            C[max(0, t - 7):t].mean(0), C[max(0, t - 28):t].mean(0),
            p["discount"][t], p["holiday_flag"][t], p["activity_flag"][t], p["precpt"][t], p["dow"][t],
        ])


def make_fit_predict(kind, panel, params, first_train_day=14):
    if ":" in kind:                                   # exploration variant, e.g. "QNTX:0.1"
        base_kind, eps = kind.split(":")[0].rstrip("X"), float(kind.split(":")[1])
        base = make_fit_predict(base_kind, panel, params, first_train_day)

        def fit_predict_x(Y, C, t_end, tau, seed, exact_labels=False):
            pred = base(Y, C, t_end, tau, seed, exact_labels)

            def predict(t, Yc, Cc):
                q = pred(t, Yc, Cc)
                share = Cc[max(0, t - 7):t].mean(0)
                return np.where(share > 1.0 - tau, q * (1.0 + eps), q)
            return predict
        return fit_predict_x

    feats = FeatureBuilder(panel)

    def training_set(Y, C, t_end):
        X = np.vstack([feats(Y, C, t) for t in range(first_train_day, t_end)])
        y = np.concatenate([Y[t] for t in range(first_train_day, t_end)])
        c = np.concatenate([C[t] for t in range(first_train_day, t_end)])
        return X, y, c

    def fit_predict(Y, C, t_end, tau, seed, exact_labels=False):
        X, y, c = training_set(Y, C, t_end)
        if exact_labels:
            c = np.zeros_like(c)
        z = stats.norm.ppf(tau)
        base = dict(params, seed=int(seed), verbosity=0)
        if kind == "MSE":
            dm = xgb.DMatrix(X, label=np.log(y + EPS))
            b = xgb.train(dict(base, objective="reg:squarederror"), dm, params["num_boost_round"])
            sd = float(np.std(np.log(y + EPS) - b.predict(dm)))
            return lambda t, Yc, Cc: np.maximum(np.exp(b.predict(xgb.DMatrix(feats(Yc, Cc, t))) + z * sd) - EPS, 0.0)
        if kind == "QNT":
            dm = xgb.DMatrix(X, label=y)
            b = xgb.train(dict(base, objective="reg:quantileerror", quantile_alpha=tau), dm, params["num_boost_round"])
            return lambda t, Yc, Cc: np.maximum(b.predict(xgb.DMatrix(feats(Yc, Cc, t))), 0.0)
        if kind == "AFT":
            lo = y + EPS
            hi = np.where(c == 1, np.inf, lo)
            dm = xgb.DMatrix(X)
            dm.set_float_info("label_lower_bound", lo)
            dm.set_float_info("label_upper_bound", hi)
            b = xgb.train(dict(base, objective="survival:aft", aft_loss_distribution="normal",
                               aft_loss_distribution_scale=params["aft_scale"]), dm, params["num_boost_round"])
            f = np.log(b.predict(dm))
            ly, cen = np.log(lo), np.isinf(hi)

            def nll(p):
                zz = (ly - f) / np.exp(p[0])
                return -np.where(cen, stats.norm.logsf(zz), stats.norm.logpdf(zz) - p[0]).sum()

            sd = float(np.exp(optimize.minimize(nll, [np.log(params["aft_scale"])], method="L-BFGS-B",
                                                bounds=[(-3, 1.5)]).x[0]))
            return lambda t, Yc, Cc: np.maximum(np.exp(np.log(b.predict(xgb.DMatrix(feats(Yc, Cc, t)))) + z * sd) - EPS, 0.0)
        raise ValueError(kind)

    return fit_predict
