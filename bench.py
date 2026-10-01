"""SagarMitra benchmark. Run: python bench.py  -> results.json

Part A, linear model (no port queues, fixed fuel prices): exact MILP vs quantum-inspired.
Part B, nonlinear model (port congestion + volume-priced green fuel):
  exact, ignoring queues        MILP that knows port capacity but not waiting time
  best classical                MILP re-solved with updated waits (sequential linearisation) + greedy repair
  quantum-inspired alone        QIEA + QPSO from a greedy start, with repair
  hybrid (SagarMitra)           QIEA + QPSO warm-started from the best classical plan
Every plan is scored on the true nonlinear model. Seeds 0-2 for the stochastic methods.
"""
import json
import time
import numpy as np

from fleet import Instance, real_routes, real_ports, synthetic_routes, synthetic_ports
from solvers import solve_milp, solve_milp_seq, solve_qi, repair

SEEDS = [0, 1, 2]
FINE = np.arange(9, 18.001, 0.25)


def cases():
    names, d, D = real_routes()
    yield "8 Indian coastal routes", 8, d, D, real_ports(), 1.0
    for R, seed in [(60, 1), (300, 2)]:
        _, d, D = synthetic_routes(R, seed)
        yield f"{R} routes (synthetic)", R, d, D, synthetic_ports(D, seed), R / 8


def feasible(res):
    return res is not None and not np.isnan(res["obj"]) and res["viol"] <= 1e-9


def pct(a, b):
    return 100.0 * (a - b) / b


def summarise(runs, ref):
    ok = [r for r in runs if feasible(r)]
    out = {"feasible_runs": len(ok), "runs": len(runs), "time_s": float(np.mean([r["time"] for r in runs]))}
    if ok:
        objs = np.array([r["obj"] for r in ok])
        out.update(best=float(objs.min()), mean=float(objs.mean()))
        if ref:
            out.update(best_pct=pct(objs.min(), ref), mean_pct=pct(objs.mean(), ref))
    return out


results = {"linear": [], "nonlinear": []}
for label, R, d, D, ports, scale in cases():
    # ---------- A: linear
    inst = Instance(d, D, carbon_price=0.1, scale_avail=scale, scale_supply=scale)
    m1 = solve_milp(inst, time_limit=120)
    m4 = solve_milp(inst, time_limit=120, grid=FINE)
    ref = m4["obj"] if feasible(m4) else m1["obj"]
    qr = [solve_qi(inst, pop=60, iters=300, seed=s, init="random", repair_every=0) for s in SEEDS]
    qg = [solve_qi(inst, pop=60, iters=300, seed=s, init="greedy") for s in SEEDS]
    row = {"case": label, "routes": R,
           "exact_1kn": {"obj": m1["obj"], "time_s": m1["time"], "vars": m1["n_vars"]},
           "exact_025kn": {"obj": m4["obj"], "time_s": m4["time"], "vars": m4["n_vars"]},
           "qi_textbook": summarise(qr, ref), "qi_repair": summarise(qg, ref)}
    results["linear"].append(row)
    print("LINEAR", json.dumps(row))

    # ---------- B: nonlinear
    inst = Instance(d, D, carbon_price=0.1, scale_avail=1.5 * scale, scale_supply=scale, ports=ports, nonlinear=True)
    m0 = solve_milp(inst, time_limit=120)
    ms = solve_milp_seq(inst, time_limit=120)
    t0 = time.perf_counter()
    classical = ms
    if not feasible(ms):  # give the classical pipeline the same repair operator
        k, f, s = repair(inst, ms["k"], ms["f"], ms["s"])
        obj, cost, emis, viol = inst.evaluate(k, f, s)
        classical = dict(ms, k=k, f=f, s=s, obj=float(obj), cost=float(cost), emis=float(emis), viol=float(viol),
                         time=ms["time"] + time.perf_counter() - t0, repaired=True)
    ref = classical["obj"] if feasible(classical) else None
    qa = [solve_qi(inst, pop=60, iters=300, seed=s, init="greedy") for s in SEEDS]
    hy = [solve_qi(inst, pop=60, iters=300, seed=s, init="warm", start=classical) for s in SEEDS] if feasible(classical) else []
    row = {"case": label, "routes": R,
           "exact_ignoring_queues": {"obj": m0["obj"], "viol": m0["viol"], "time_s": m0["time"],
                                     "valid": feasible(m0), "pct_vs_classical": pct(m0["obj"], ref) if ref else None},
           "best_classical": {"obj": classical["obj"], "viol": classical["viol"], "time_s": classical["time"],
                              "valid": feasible(classical), "rounds": len(ms.get("trace", [])),
                              "repaired": bool(classical.get("repaired", False)), "emis": classical.get("emis")},
           "qi_alone": summarise(qa, ref),
           "hybrid": summarise(hy, ref)}
    if hy:
        bh = min([h for h in hy if feasible(h)], key=lambda h: h["obj"])
        row["hybrid"]["emis_best"] = bh["emis"]
    results["nonlinear"].append(row)
    print("NONLINEAR", json.dumps(row))

json.dump(results, open("results.json", "w"), indent=2)
print("saved results.json")
