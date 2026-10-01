"""Fuel-intensity prediction on real ships (EU-MRV 2023), three models compared on held-out ships.

Target  annual CO2 per tonne of cargo per mile [g CO2 / t / n mile]  (operational carbon intensity)
Inputs  ship type, design efficiency (EEDI, EEXI or EIV, g CO2 / t / n mile), average speed, time at sea
        (speed = distance / time at sea, distance = total fuel / fuel per mile; no input uses the target's denominator)

Models
  physics      log y = a_type + b log(EEDI) + c log(v)        (admiralty law: intensity ~ design efficiency * v^2)
  gbm          gradient-boosted trees on the same inputs
  qkernel      kernel ridge with a quantum fidelity kernel, simulated classically:
               each input angle-encoded on one qubit (RY), k(x, y) = prod_i cos^2((x_i - y_i) / 2)
  qkernel+phys the quantum kernel learns the residual of the physics model
  ensemble     equal-weight average of gbm and qkernel (log space)

Run: python fuel_model.py  -> fuel_results.json
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
SEED = 7


def load(path="data/eu_mrv_2023.xlsx"):
    raw = pd.read_excel(path, header=2)
    cols = {c: re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns}
    raw = raw.rename(columns=cols)
    pick = lambda frag: [c for c in raw.columns if frag in c][0]
    df = pd.DataFrame({
        "imo": raw["IMO Number"],
        "type": raw["Ship type"],
        "tech": raw["Technical efficiency"].astype(str),
        "fuel_t": pd.to_numeric(raw[pick("Total fuel consumption")], errors="coerce"),
        "hours": pd.to_numeric(raw[pick("Annual Time spent at sea")], errors="coerce"),
        "kg_nm": pd.to_numeric(raw[pick("Annual average Fuel consumption per distance")], errors="coerce"),
        "co2_dwt": pd.to_numeric(raw[pick("Annual average CO₂ emissions per transport work (mass)")], errors="coerce"),
    })
    m = df["tech"].str.extract(r"(EEDI|EEXI|EIV)\s*\(([\d.]+)")
    df["eff_kind"], df["eff"] = m[0], pd.to_numeric(m[1], errors="coerce")
    df["dist"] = df["fuel_t"] * 1000 / df["kg_nm"]
    df["v"] = df["dist"] / df["hours"]
    keep = (df["eff"] > 0.5) & (df["co2_dwt"] > 0) & df["v"].between(4, 30) & (df["hours"] > 500)
    df = df[keep].dropna(subset=["eff", "v", "co2_dwt", "type"]).copy()
    # drop extreme reporting outliers (outside 1st-99th percentile per type)
    q = df.groupby("type")["co2_dwt"].transform
    df = df[(df["co2_dwt"] >= q(lambda s: s.quantile(0.01))) & (df["co2_dwt"] <= q(lambda s: s.quantile(0.99)))]
    common = df["type"].value_counts()
    df = df[df["type"].isin(common[common >= 150].index)]
    return df.reset_index(drop=True)


def features(df, types):
    X = np.column_stack([np.log(df["eff"]), np.log(df["v"]), np.log(df["hours"]), (df["eff_kind"] == "EIV").astype(float), (df["eff_kind"] == "EEXI").astype(float)])
    T = np.column_stack([(df["type"] == t).astype(float) for t in types])
    return X, T


def qkernel(A, B):
    # fidelity kernel of single-qubit RY angle encodings, one qubit per feature
    k = np.ones((A.shape[0], B.shape[0]))
    for i in range(A.shape[1]):
        k *= np.cos((A[:, i][:, None] - B[:, i][None, :]) / 2) ** 2
    return k


def angles(Xtr, Xte, Ttr, Tte, gamma=1.0):
    """Angle encoding scaled by a bandwidth gamma (Shaydulin & Wild 2022)."""
    lo, hi = Xtr.min(0), Xtr.max(0)
    sc = lambda X: (X - lo) / np.where(hi > lo, hi - lo, 1) * np.pi * gamma
    return np.hstack([sc(Xtr), Ttr * np.pi * gamma]), np.hstack([sc(Xte), Tte * np.pi * gamma])


def krr_fit_predict(Atr, ytr, Ate, lam=1e-2, n_max=8000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Atr), min(n_max, len(Atr)), replace=False)
    K = qkernel(Atr[idx], Atr[idx])
    mu = ytr[idx].mean()
    alpha = np.linalg.solve(K + lam * np.eye(len(idx)), ytr[idx] - mu)
    return qkernel(Ate, Atr[idx]) @ alpha + mu


def metrics(y, p):
    # y, p in log space; report on the original scale
    Y, P = np.exp(y), np.exp(p)
    mape = float(np.mean(np.abs(P - Y) / Y) * 100)
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"mape_pct": mape, "r2_log": r2}


def main(seed=SEED, df=None, save=True):
    df = load() if df is None else df
    types = sorted(df["type"].unique())
    rng = np.random.default_rng(seed)
    imos = df["imo"].unique()
    test_imo = set(rng.choice(imos, int(0.2 * len(imos)), replace=False))
    te = df["imo"].isin(test_imo).values
    X, T = features(df, types)
    y = np.log(df["co2_dwt"].values)
    Xtr, Xte, Ttr, Tte, ytr, yte = X[~te], X[te], T[~te], T[te], y[~te], y[te]

    out = {"ships_train": int((~te).sum()), "ships_test": int(te.sum()), "types": types, "models": {}}

    phys = LinearRegression().fit(np.hstack([Ttr, Xtr[:, :2]]), ytr)
    p_phys_tr = phys.predict(np.hstack([Ttr, Xtr[:, :2]]))
    p_phys = phys.predict(np.hstack([Tte, Xte[:, :2]]))
    out["models"]["physics"] = metrics(yte, p_phys)
    out["physics_speed_exponent"] = float(phys.coef_[len(types) + 1])
    out["physics_eedi_exponent"] = float(phys.coef_[len(types)])

    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED)
    gbm.fit(np.hstack([Xtr, Ttr]), ytr)
    p_gbm = gbm.predict(np.hstack([Xte, Tte]))
    out["models"]["gbm"] = metrics(yte, p_gbm)

    vmask = np.random.default_rng(seed + 1).random(len(Xtr)) < 0.2  # validation split of the training ships
    best = None
    for gamma in [0.25, 0.5, 1.0]:
        Atr_g, _ = angles(Xtr, Xte, Ttr, Tte, gamma)
        for lam in [1e-3, 1e-2, 1e-1, 1.0]:
            pv = krr_fit_predict(Atr_g[~vmask], ytr[~vmask], Atr_g[vmask], lam, n_max=4000)
            e = np.mean((pv - ytr[vmask]) ** 2)
            if best is None or e < best[0]:
                best = (e, lam, gamma)
    _, lam, gamma = best
    Atr, Ate = angles(Xtr, Xte, Ttr, Tte, gamma)
    out["qkernel_gamma"] = gamma
    p_qk = krr_fit_predict(Atr, ytr, Ate, lam)
    out["models"]["qkernel"] = metrics(yte, p_qk)
    p_qkp = p_phys + krr_fit_predict(Atr, ytr - p_phys_tr, Ate, lam)
    out["models"]["qkernel_plus_physics"] = metrics(yte, p_qkp)
    p_ens = 0.5 * (p_gbm + p_qk)  # equal-weight average of gradient boosting and the quantum kernel
    out["models"]["ensemble_gbm_qkernel"] = metrics(yte, p_ens)
    out["qkernel_lambda"] = lam
    out["qkernel_train_subset"] = int(min(8000, len(Atr)))

    # per-type error of the best model
    names = {"physics": p_phys, "gbm": p_gbm, "qkernel": p_qk, "qkernel_plus_physics": p_qkp, "ensemble_gbm_qkernel": p_ens}
    bestm = min(out["models"], key=lambda m: out["models"][m]["mape_pct"])
    out["best_model"] = bestm
    tt = df["type"].values[te]
    out["per_type_best"] = {t: metrics(yte[tt == t], names[bestm][tt == t]) | {"n": int((tt == t).sum())} for t in types}

    if save:
        json.dump(out, open("fuel_results.json", "w"), indent=2)
    print(seed, {m: round(v["mape_pct"], 2) for m, v in out["models"].items()})
    return out
    print("best:", bestm, "| train", out["ships_train"], "test", out["ships_test"],
          "| speed exponent %.2f, EEDI exponent %.2f" % (out["physics_speed_exponent"], out["physics_eedi_exponent"]))


def robustness(seeds=(7, 8, 9, 10, 11)):
    """Same experiment on five different train/test splits of the ships."""
    df = load()
    runs = [main(sd, df, save=(sd == seeds[0])) for sd in seeds]
    summary = {}
    for m in runs[0]["models"]:
        v = np.array([r["models"][m]["mape_pct"] for r in runs])
        r2 = np.array([r["models"][m]["r2_log"] for r in runs])
        summary[m] = {"mape_mean": float(v.mean()), "mape_sd": float(v.std(ddof=1)), "r2_mean": float(r2.mean())}
    q, g = [np.array([r["models"][m]["mape_pct"] for r in runs]) for m in ("qkernel", "gbm")]
    summary["qkernel_beats_gbm_splits"] = int((q < g).sum())
    e = np.array([r["models"]["ensemble_gbm_qkernel"]["mape_pct"] for r in runs])
    summary["ensemble_beats_gbm_splits"] = int((e < g).sum())
    summary["splits"] = len(seeds)
    summary["ships"] = int(runs[0]["ships_train"] + runs[0]["ships_test"])
    summary["eedi_exponent_mean"] = float(np.mean([r["physics_eedi_exponent"] for r in runs]))
    res = json.load(open("fuel_results.json"))
    res["robustness"] = summary
    json.dump(res, open("fuel_results.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    robustness()
