"""Solvers for the SagarMitra fleet model.

solve_milp       exact MILP (HiGHS via scipy) on a 1-knot speed grid. With pwl=True the green-fuel
                 volume pricing is modelled exactly by piecewise-linear segments; port waits are taken
                 as fixed inputs W per route.
solve_milp_seq   the best classical approach we could build for the nonlinear model: re-solve the MILP,
                 updating port waits from the last plan each time (sequential linearisation).
greedy           cheapest option per route, then repair.
solve_qi         QIEA (Han & Kim 2002) for vessel type and fuel + QPSO (Sun et al. 2004) for speeds,
                 with a repair operator, greedy or warm-start initialisation, and elitism.
"""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

from fleet import (CAP, FUELS, EF, A, ETA, CII_LIMIT, IS_ALT, ALT_OK_MIN_TYPE, SPEED_MIN, SPEED_MAX,
                   PRICE, GREEN, RHO_MAX)

K, F = len(CAP), len(FUELS)
SPEED_GRID = np.arange(SPEED_MIN, SPEED_MAX + 0.01, 1.0)
PWL_SEGMENTS = 8


def s_cap(k, f):
    """Highest speed that keeps (k, f) under the intensity limit."""
    coef = EF[f] / 1000 * A[k] * 0.0036 / ETA / CAP[k]
    return np.sqrt(CII_LIMIT / coef)


def _options(inst, grid):
    opts = []
    for r in range(inst.R):
        for k in range(K):
            for f in range(F):
                if IS_ALT[f] and k < ALT_OK_MIN_TYPE:
                    continue
                for s in grid:
                    if s <= s_cap(k, f) + 1e-9:
                        opts.append((r, k, f, s))
    o = np.array(opts, dtype=float)
    return o[:, 0].astype(int), o[:, 1].astype(int), o[:, 2].astype(int), o[:, 3]


# ---------------------------------------------------------------- exact
def solve_milp(inst, time_limit=60.0, grid=SPEED_GRID, W=None, pwl=False):
    r_, k_, f_, s_ = _options(inst, grid)
    Wr = np.zeros(inst.R) if W is None else np.asarray(W)
    n, gj, cost, emis = inst.route_terms(k_, f_, s_, Wr[r_], idx=r_)
    c = cost + inst.cp * emis
    use_pwl = pwl and inst.nonlinear
    if use_pwl:  # take green fuel out of the per-option cost; segments carry it instead
        c = c - np.where(GREEN[f_], gj * PRICE[f_], 0.0)
    nv = len(c)

    seg_cols, seg_cost, seg_ub = [], [], []
    if use_pwl:
        for j in np.where(GREEN)[0]:
            L = inst.supply[j] / PWL_SEGMENTS
            for m in range(PWL_SEGMENTS):
                mid = (m + 0.5) * L
                seg_cols.append(j); seg_cost.append(PRICE[j] * (1 + mid / inst.supply[j])); seg_ub.append(L)
    ns = len(seg_cols)
    call = np.concatenate([c, np.array(seg_cost)]) if ns else c

    rows, cols, vals, lo, hi = [], [], [], [], []
    row = 0
    for r in range(inst.R):  # exactly one option per route
        idx = np.where(r_ == r)[0]
        rows += [row] * len(idx); cols += list(idx); vals += [1.0] * len(idx); lo.append(1); hi.append(1); row += 1
    for t in range(K):  # ships available
        idx = np.where(k_ == t)[0]
        rows += [row] * len(idx); cols += list(idx); vals += list(n[idx]); lo.append(-np.inf); hi.append(inst.avail[t]); row += 1
    for j in range(F):  # green fuel supply
        if np.isfinite(inst.supply[j]):
            idx = np.where(f_ == j)[0]
            rows += [row] * len(idx); cols += list(idx); vals += list(gj[idx]); lo.append(-np.inf); hi.append(inst.supply[j]); row += 1
            if use_pwl:  # green fuel used = sum of its segments
                rows += [row] * len(idx); cols += list(idx); vals += list(gj[idx])
                sidx = [nv + i for i, jj in enumerate(seg_cols) if jj == j]
                rows += [row] * len(sidx); cols += sidx; vals += [-1.0] * len(sidx)
                lo.append(0); hi.append(0); row += 1
    if inst.nonlinear:  # port call capacity is linear, so the exact model gets it too
        V = np.ceil(inst.D[r_] / CAP[k_])
        for p in range(inst.P):
            idx = np.where((inst.pa[r_] == p) | (inst.pb[r_] == p))[0]
            mult = (inst.pa[r_[idx]] == p).astype(float) + (inst.pb[r_[idx]] == p).astype(float)
            rows += [row] * len(idx); cols += list(idx); vals += list(V[idx] * mult)
            lo.append(-np.inf); hi.append(RHO_MAX * inst.port_cap[p]); row += 1
    Amat = coo_matrix((vals, (rows, cols)), shape=(row, nv + ns)).tocsr()
    integ = np.concatenate([np.ones(nv), np.zeros(ns)])
    ub = np.concatenate([np.ones(nv), np.array(seg_ub)]) if ns else np.ones(nv)

    t0 = time.perf_counter()
    res = milp(call, constraints=LinearConstraint(Amat, lo, hi), integrality=integ, bounds=Bounds(0, ub),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-6})
    dt = time.perf_counter() - t0
    out = {"time": dt, "status": res.status, "n_vars": nv + ns}
    if res.x is None:
        out.update(obj=np.nan)
        return out
    pick = res.x[:nv] > 0.5
    order = np.argsort(r_[pick])
    kk, ff, ss = k_[pick][order], f_[pick][order], s_[pick][order]
    obj, tc, em, viol = inst.evaluate(kk, ff, ss)
    out.update(obj=float(obj), cost=float(tc), emis=float(em), viol=float(viol),
               model_obj=float(res.fun), bound=float(getattr(res, "mip_dual_bound", np.nan)),
               gap=float(getattr(res, "mip_gap", np.nan)), k=kk, f=ff, s=ss)
    return out


def solve_milp_seq(inst, rounds=6, time_limit=60.0, damping=0.5):
    """Sequential linearisation: fix port waits, solve, recompute waits from the new plan, repeat."""
    W = np.zeros(inst.R)
    best, total_t, trace = None, 0.0, []
    for _ in range(rounds):
        m = solve_milp(inst, time_limit=time_limit, W=W, pwl=True)
        total_t += m["time"]
        if np.isnan(m["obj"]):
            break
        trace.append(m["obj"])
        if best is None or (m["viol"] > 1e-9, m["obj"]) < (best["viol"] > 1e-9, best["obj"]):
            best = m
        Wn, _ = inst.waits(m["k"])
        if np.allclose(Wn, W, atol=0.05):
            break
        W = damping * W + (1 - damping) * Wn
    best = dict(best)
    best["time"], best["trace"] = total_t, trace
    return best


# ---------------------------------------------------------------- repair and greedy
def _fix_fuel(k, f):
    return np.where((k < ALT_OK_MIN_TYPE) & IS_ALT[f], 0, f)


def _clip_speed(k, f, s):
    return np.clip(s, SPEED_MIN, np.minimum(SPEED_MAX, s_cap(k, f)))


def repair(inst, k, f, s, max_moves=None, max_routes=40, big=1e4):
    """Greedy repair of one plan (arrays of length R). Each step tries single-route changes on the routes
    involved in a violation (other ship type at current or max speed, other fuel, faster speed), scores
    every candidate on the full plan as objective + big * violation, and applies the best one.
    Stops when the plan is valid or no change helps."""
    k, f = k.copy(), _fix_fuel(k, f.copy())
    s = _clip_speed(k, f, s.copy())
    max_moves = max_moves or 3 * inst.R

    def score(Kb, Fb, Sb):
        obj, _, _, viol = inst.evaluate(Kb, Fb, Sb)
        return obj + big * viol, viol

    cur, v = score(k[None], f[None], s[None])
    cur, v = float(cur[0]), float(v[0])
    for _ in range(max_moves):
        if v <= 1e-9:
            break
        W, _ = inst.waits(k)
        n, gj, _, _ = inst.route_terms(k, f, s, W)
        involved = np.zeros(inst.R, bool)
        used = np.bincount(k, weights=n, minlength=K)
        for t in np.where(used > inst.avail + 1e-9)[0]:
            involved |= k == t
        for j in range(F):
            if np.isfinite(inst.supply[j]) and gj[f == j].sum() > inst.supply[j] + 1e-6:
                involved |= f == j
        if inst.nonlinear:
            V = inst.voyages(k)
            calls = np.bincount(inst.pa, weights=V, minlength=inst.P) + np.bincount(inst.pb, weights=V, minlength=inst.P)
            for p_ in np.where(calls > RHO_MAX * inst.port_cap + 1e-9)[0]:
                involved |= (inst.pa == p_) | (inst.pb == p_)
        involved |= ~inst.allowed(k, f, s)
        idx = np.where(involved)[0]
        if len(idx) == 0:
            idx = np.arange(inst.R)
        if len(idx) > max_routes:  # the heaviest routes first
            idx = idx[np.argsort(-n[idx])[:max_routes]]
        cand = []
        for r in idx:
            for t2 in range(K):
                f2 = int(_fix_fuel(np.array([t2]), np.array([f[r]]))[0])
                for sp in (s[r], SPEED_MAX):
                    cand.append((r, t2, f2, float(_clip_speed(np.array([t2]), np.array([f2]), np.array([sp]))[0])))
            for f2v in range(F):
                f2 = int(_fix_fuel(np.array([k[r]]), np.array([f2v]))[0])
                cand.append((r, k[r], f2, float(_clip_speed(np.array([k[r]]), np.array([f2]), np.array([s[r]]))[0])))
            cand.append((r, k[r], f[r], float(min(SPEED_MAX, s_cap(k[r], f[r]), s[r] + 1.0))))
        c = np.array(cand)
        m = len(c)
        rr = c[:, 0].astype(int)
        Kb = np.repeat(k[None], m, 0); Fb = np.repeat(f[None], m, 0); Sb = np.repeat(s[None], m, 0)
        Kb[np.arange(m), rr] = c[:, 1].astype(int)
        Fb[np.arange(m), rr] = c[:, 2].astype(int)
        Sb[np.arange(m), rr] = c[:, 3]
        sc, vv = score(Kb, Fb, Sb)
        b = int(np.argmin(sc))
        if sc[b] >= cur - 1e-9:
            break
        k, f, s = Kb[b].copy(), Fb[b].copy(), Sb[b].copy()
        cur, v = float(sc[b]), float(vv[b])
    return k, f, s


def greedy(inst, grid=SPEED_GRID):
    r_, k_, f_, s_ = _options(inst, grid)
    n, gj, cost, emis = inst.route_terms(k_, f_, s_, 0.0, idx=r_)
    c = cost + inst.cp * emis
    k = np.zeros(inst.R, int); f = np.zeros(inst.R, int); s = np.full(inst.R, SPEED_MIN)
    for r in range(inst.R):
        i = np.where(r_ == r)[0]
        j = i[np.argmin(c[i])]
        k[r], f[r], s[r] = k_[j], f_[j], s_[j]
    return repair(inst, k, f, s)


# ---------------------------------------------------------------- quantum-inspired hybrid
def _bits_needed(m):
    return int(np.ceil(np.log2(m)))


def solve_qi(inst, pop=60, iters=400, seed=0, penalty=None, polish=True, init="greedy", start=None,
             repair_every=5, time_budget=None, n_repair=4):
    """QIEA for vessel type and fuel, QPSO for speeds.

    init="random"  plain QIEA/QPSO from uniform Q-bits (the textbook algorithm)
    init="greedy"  Q-bits and particles centred on the greedy plan
    init="warm"    centred on `start` (e.g. the exact solver's plan); that plan is also individual 0,
                   so with elitism the result can never be worse than the start
    Every `repair_every` iterations each individual is repaired and written back (Lamarckian repair).
    """
    rng = np.random.default_rng(seed)
    R = inst.R
    repair_moves = 6  # in-loop repair is a quick fix; the full repair runs at initialisation
    bk, bf = _bits_needed(K), _bits_needed(F)
    wk, wf = 2 ** np.arange(bk)[::-1], 2 ** np.arange(bf)[::-1]
    t0 = time.perf_counter()

    def to_bits(k, f):
        kb = (k[..., None] // wk) % 2
        fb = (f[..., None] // wf) % 2
        return np.concatenate([kb, fb], -1)

    def from_bits(bits):
        k = (bits[..., :bk] @ wk) % K
        f = (bits[..., bk:] @ wf) % F
        return k, _fix_fuel(k, f)

    if init == "random":
        theta = np.full((pop, R, bk + bf), np.pi / 4)
        x = rng.uniform(SPEED_MIN, SPEED_MAX, (pop, R))
        centre = None
    else:
        centre = start if init == "warm" else dict(zip("kfs", greedy(inst)))
        cb = to_bits(np.asarray(centre["k"]), np.asarray(centre["f"]))
        p1 = np.where(cb == 1, 0.85, 0.15)  # P(bit = centre bit) = 0.85
        theta = np.broadcast_to(np.arcsin(np.sqrt(p1)), (pop, R, bk + bf)).copy()
        x = np.asarray(centre["s"], float) + rng.normal(0, 0.8, (pop, R))

    obj0 = inst.evaluate(np.zeros((1, R), int), np.zeros((1, R), int), np.full((1, R), 12.0))[0][0]
    penalty = penalty or max(1.0, float(obj0)) * 0.05

    def fitness(k, f, s):
        obj, _, _, viol = inst.evaluate(k, f, s)
        return obj + penalty * viol

    def observe():
        bits = (rng.random(theta.shape) < np.sin(theta) ** 2).astype(int)
        k, f = from_bits(bits)
        return k, f

    k, f = observe()
    s = _clip_speed(k, f, x)
    if centre is not None:
        k[0], f[0], s[0] = np.asarray(centre["k"]), np.asarray(centre["f"]), np.asarray(centre["s"], float)
    for i in range(min(pop, 2 * n_repair)):
        k[i], f[i], s[i] = repair(inst, k[i], f[i], s[i])
    fit = fitness(k, f, s)
    pb_k, pb_f, pb_s, pb_fit = k.copy(), f.copy(), s.copy(), fit.copy()
    g = int(np.argmin(pb_fit))
    history = [float(pb_fit[g])]
    dmax, dmin = 0.05 * np.pi, 0.005 * np.pi

    it = 0
    for it in range(iters):
        if time_budget and time.perf_counter() - t0 > time_budget:
            break
        frac = it / max(1, iters - 1)
        beta = 1.0 - 0.5 * frac  # QPSO contraction-expansion
        mbest = pb_s.mean(0)
        phi = rng.random((pop, R))
        attractor = phi * pb_s + (1 - phi) * pb_s[g]
        u = rng.random((pop, R))
        sign = np.where(rng.random((pop, R)) < 0.5, -1.0, 1.0)
        x = attractor + sign * beta * np.abs(mbest - x) * np.log(1 / u)
        k, f = observe()
        s = _clip_speed(k, f, x)
        fit = fitness(k, f, s)
        if repair_every and it % repair_every == 0:  # repair the most promising infeasible individuals
            _, _, _, viol = inst.evaluate(k, f, s)
            cand = [i for i in np.argsort(fit) if viol[i] > 0][:n_repair]
            for i in cand:
                k[i], f[i], s[i] = repair(inst, k[i], f[i], s[i], max_moves=repair_moves)
            if cand:
                x[cand] = s[cand]
                fit = fitness(k, f, s)
        better = fit < pb_fit
        pb_k[better], pb_f[better], pb_s[better], pb_fit[better] = k[better], f[better], s[better], fit[better]
        g = int(np.argmin(pb_fit))
        # rotation gate toward each individual's best; every 10 iterations toward the global best
        target = to_bits(pb_k, pb_f) if it % 10 else np.broadcast_to(to_bits(pb_k[g], pb_f[g]), theta.shape)
        dth = dmax - (dmax - dmin) * frac
        theta = np.clip(theta + dth * np.where(target == 1, 1.0, -1.0), 0.03, np.pi / 2 - 0.03)
        history.append(float(pb_fit[g]))

    best = dict(k=pb_k[g].copy(), f=pb_f[g].copy(), s=pb_s[g].copy())
    if polish:  # coordinate search on continuous speeds, other routes fixed
        for _ in range(2):
            for r in range(R):
                cand = np.linspace(SPEED_MIN, min(SPEED_MAX, s_cap(best["k"][r], best["f"][r])), 91)
                S = np.repeat(best["s"][None, :], len(cand), 0); S[:, r] = cand
                Kk = np.repeat(best["k"][None, :], len(cand), 0); Ff = np.repeat(best["f"][None, :], len(cand), 0)
                fv = fitness(Kk, Ff, S)
                if fv.min() < fitness(best["k"][None], best["f"][None], best["s"][None])[0] - 1e-9:
                    best["s"][r] = cand[int(np.argmin(fv))]
    dt = time.perf_counter() - t0
    obj, cost, emis, viol = inst.evaluate(best["k"][None], best["f"][None], best["s"][None])
    return dict(time=dt, obj=float(obj[0]), cost=float(cost[0]), emis=float(emis[0]), viol=float(viol[0]),
                history=history, iters=it + 1, **best)
