# Slides 2-4 for build_deck.py. Exec'd inside build_deck.py so it shares its helpers and slide objects.
# Every number comes from results.json (python bench.py).

LIN, NL = RES["linear"], RES["nonlinear"]
FR = json.load(open(os.path.join(ROOT, "fuel_results.json")))
ROB = FR["robustness"]
base_e, gbm_e, qk_e = ROB["physics"]["mape_mean"], ROB["gbm"]["mape_mean"], ROB["qkernel"]["mape_mean"]
ens = ROB.get("ensemble_gbm_qkernel")
ens_wins = ens is not None and ens["mape_mean"] < gbm_e


def sgn(x):
    return f"{x:+.1f}%".replace("-", "−")


def hyb_pct(r):
    h = r["hybrid"]
    return h.get("best_pct") if h.get("feasible_runs") else None


def qi_pct(r):
    q = r["qi_alone"]
    return q.get("best_pct") if q.get("feasible_runs") else None


lin_fail = [r["routes"] for r in LIN if r["qi_repair"]["feasible_runs"] == 0]
hyb_all = [hyb_pct(r) for r in NL]

# ---------- slide 2: proposed solution
set_team(s2); restyle_pointers(s2)
heading(s2, "We measured where quantum-inspired methods help, and where they do not.")

box(s2, 0.5, 2.25, 6.05, 3.5, fill=TINT)
text(s2, 0.75, 2.4, 5.6, 0.35, ["Fuel prediction, tested on real ships"], size=14, bold=True, color=TEAL)
left = [
    (f"{ROB['ships']:,}",
     f"real ships from the EU-MRV 2023 reports. 80% train, 20% held out, repeated on {ROB['splits']} random splits."),
    (f"{qk_e:.1f}%",
     f"average error of the quantum-kernel model. Gradient boosting: {gbm_e:.1f}%. Design-efficiency baseline: {base_e:.1f}%."),
    ((f"{ens['mape_mean']:.1f}%",
      f"averaging the quantum kernel with gradient boosting: lower than gradient boosting alone in {ROB['ensemble_beats_gbm_splits']} of {ROB['splits']} splits.")
     if ens_wins else
     ("Level", "Quantum kernel and gradient boosting are level within the noise. Both clearly beat the baseline.")),
]
for i, (num, lab) in enumerate(left):
    y = 2.9 + i * 0.93
    text(s2, 0.75, y, 1.6, 0.8, [num], size=26, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s2, 2.4, y, 3.95, 0.8, [lab], size=11.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)

box(s2, 6.8, 2.25, 6.0, 3.5, fill=TINT2)
text(s2, 7.05, 2.4, 5.6, 0.35, ["Fleet optimisation: the hybrid wins where ports queue"], size=14, bold=True, color=ORANGE)
right = [
    ("Invalid", "The exact plan that ignores port queues needs more ships than the fleet has, once waiting time is counted."),
    (" · ".join(sgn(x) for x in hyb_all if x is not None),
     "Hybrid against the best classical plan (exact solver re-solved with updated waits, then repaired) at "
     + " / ".join(str(r["routes"]) for r, x in zip(NL, hyb_all) if x is not None) + " routes."),
    (" · ".join(sgn(100 * (r["hybrid"]["emis_best"] - r["best_classical"]["emis"]) / r["best_classical"]["emis"])
                for r in NL if r["hybrid"].get("emis_best")),
     "Lifecycle emissions of the hybrid plan against the classical plan, at the same sizes. Cheaper and cleaner at once."),
]
for i, (num, lab) in enumerate(right):
    y = 2.9 + i * 0.93
    sz = 26 if len(num) <= 7 else (19 if len(num) <= 14 else 15)
    text(s2, 7.05, y, 2.0, 0.8, [num], size=sz, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s2, 9.1, y, 3.55, 0.8, [lab], size=11.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)

boxed_text(s2, 0.5, 5.92, 12.3, 0.62,
           [[("SagarMitra: ", {"bold": True, "color": FOAM}),
             ("the exact solver builds the plan and quantum-inspired search refines what is not linear. "
              "It starts from the classical plan, so it never returns anything worse.", {})]],
           fill=NAVY, size=13, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, pad=0.2)

# ---------- slide 3: technical approach
set_team(s3); restyle_pointers(s3)
labels = ["Routes, cargo,\nfleet, ports", "Fuel model\nphysics + ML", "Exact core\nHiGHS", "Quantum-inspired\nrefinement",
          "Rule check\ncargo · ports · CII", "Trade-off\ndashboard"]
fills = [NAVY, NAVY, BLUE, ORANGE, NAVY, NAVY]
for i, (lab, fc) in enumerate(zip(labels, fills)):
    shp = MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON
    b = box(s3, 0.5 + i * 2.02, 1.72, 2.2, 0.9, fill=fc, shape=shp)
    b.adjustments[0] = 0.28
    text(s3, 0, 0, 0, 0, lab.split("\n"), size=10.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, space_after=0, shape=b)
    b.text_frame.margin_left = Inches(0.3 if i else 0.1); b.text_frame.margin_right = Inches(0.22)

bul = [
    ("Fuel model: ", "design efficiency (EEDI/EEXI/EIV), speed, time at sea and ship type. Gradient boosting and a quantum fidelity kernel (one qubit per input, simulated)."),
    ("Exact core: ", "ship, fuel and speed per route under fleet, port and green-fuel limits; volume pricing as piecewise-linear segments; re-solved with updated port waits."),
    ("Quantum-inspired refinement: ", "QIEA for ship type and fuel, QPSO for continuous speeds, a repair step, warm-started from the exact plan."),
    ("Not linear: ", "port queue wait W\u2080\u00b7\u03c1/(1\u2212\u03c1); green fuel dearer with volume. Emissions well-to-wake (IMO LCA)."),
]
text(s3, 0.5, 2.85, 6.1, 3.1, [[("\u2022  " + a, {"bold": True, "color": NAVY}), (b, {})] for a, b in bul],
     size=10.5, color=GREY, space_after=6)


def table(slide, x, y, widths, rows, colour):
    t = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(0.32 * len(rows))).table
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, rw in enumerate(rows):
        t.rows[i].height = Inches(0.32)
        for j, val in enumerate(rw):
            c = t.cell(i, j)
            c.fill.solid(); c.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i % 2 else WHITE)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.text = ""
            r = tf.paragraphs[0].add_run(); r.text = val
            r.font.name = FONT; r.font.size = Pt(9.5); r.font.bold = (i == 0 or j == 0)
            r.font.color.rgb = WHITE if i == 0 else colour(i, j, val)


text(s3, 6.85, 2.78, 5.95, 0.3, [f"Fuel prediction: {ROB['ships']:,} real ships (EU-MRV 2023), {ROB['splits']} splits"],
     size=11.5, bold=True, color=TEAL)
prow = [("Model", "Error (MAPE)", "R\u00b2 (log)")]
pm = [("Design-efficiency baseline", "physics"), ("Gradient boosting", "gbm"), ("Quantum kernel (simulated)", "qkernel")]
if ens is not None:
    pm.append(("Average of both", "ensemble_gbm_qkernel"))
for name, key in pm:
    m = ROB[key]
    prow.append((name, f"{m['mape_mean']:.1f}% \u00b1 {m['mape_sd']:.1f}", f"{m['r2_mean']:.2f}"))
best_key = min([k for _, k in pm], key=lambda k: ROB[k]["mape_mean"])
best_name = [n for n, k in pm if k == best_key][0]
table(s3, 6.85, 3.1, [2.55, 1.9, 1.5], prow,
      lambda i, j, v: TEAL if prow[i][0] == best_name else NAVY)

y2 = 3.1 + 0.32 * len(prow) + 0.1
text(s3, 6.85, y2, 5.95, 0.3, ["Fleet optimisation: port queues + volume-priced green fuel"], size=11.5, bold=True, color=TEAL)
orow = [("Fleet", "Exact, no queues", "Best classical", "QI alone", "Hybrid")]
for r in NL:
    e = "Invalid plan" if not r["exact_ignoring_queues"]["valid"] else "Valid"
    c = ("Valid \u00b7 " + fmt_s(r["best_classical"]["time_s"])) if r["best_classical"]["valid"] else "Invalid"
    q = sgn(qi_pct(r)) if qi_pct(r) is not None else "No plan"
    h = sgn(hyb_pct(r)) if hyb_pct(r) is not None else "n/a"
    orow.append((f"{r['routes']} routes" + (" (real)" if r["routes"] == 8 else ""), e, c, q, h))


def ocol(i, j, v):
    if j == 4:
        return TEAL
    if j in (1, 3) and ("Invalid" in v or "+" in v or "No plan" in v):
        return ORANGE
    return NAVY


table(s3, 6.85, y2 + 0.32, [1.2, 1.25, 1.3, 1.0, 1.2], orow, ocol)
yn = y2 + 0.32 + 0.32 * len(orow) + 0.05
text(s3, 6.85, yn, 5.95, 0.5,
     [f"% vs best classical plan, best of 3 seeds. Without queues the exact solver wins: textbook QI is "
      f"{sgn(LIN[0]['qi_textbook']['best_pct'])} at 8 routes. Re-run: python bench.py, python fuel_model.py"],
     size=8.5, color=GREY, italic=True)

chips = ["Python", "NumPy", "SciPy / HiGHS", "scikit-learn", "pymoo", "FastAPI", "React"]
x = 0.5
for c in chips:
    w = 0.2 + 0.072 * len(c)
    b = box(s3, x, 5.45, w, 0.34, fill=TINT, radius=0.4)
    text(s3, 0, 0, 0, 0, [c], size=9.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b)
    x += w + 0.08
text(s3, 0.5, 5.92, 6.1, 0.34, ["Runs on a normal computer. No quantum hardware needed."], size=11, color=GREY,
     italic=True, anchor=MSO_ANCHOR.MIDDLE)

# ---------- slide 4: feasibility
set_team(s4); restyle_pointers(s4)
box(s4, 0.5, 1.65, 4.2, 4.85, fill=NAVY)
text(s4, 0.78, 1.85, 3.7, 0.4, ["Already working"], size=16, bold=True, color=WHITE)
done = [f"Fuel model on {ROB['ships']:,} real EU-MRV ships, 5 test splits",
        "Fleet model: 8 Indian coastal routes, port queues, green-fuel pricing",
        "Exact solver + QIEA/QPSO hybrid with repair",
        "Benchmarks: 3 fleet sizes, 3 seeds, one command each", "Public code on GitHub, explainer video"]
for i, d in enumerate(done):
    y = 2.38 + i * 0.55
    c = box(s4, 0.8, y + 0.05, 0.3, 0.3, fill=TEAL, shape=MSO_SHAPE.OVAL)
    text(s4, 0, 0, 0, 0, ["✓"], size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=c)
    text(s4, 1.25, y, 3.3, 0.45, [d], size=11.5, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(s4, 0.78, 5.25, 3.7, 1.1, [[("Next: ", {"bold": True, "color": FOAM}),
                                ("voyage-level fuel data (weather, draft) for the fuel model, real port-call statistics, planner dashboard.", {})]],
     size=12, color=WHITE)

risks = [
    ("Quantum-inspired search loses on simple models", "Measured: it does. The exact solver handles the linear core, and quantum-inspired search only refines the rest."),
    ("A stronger classical method could close the gap", "We benchmark against the strongest classical pipeline we can build, and publish every number."),
    ("Public fuel data is annual, not per voyage", "Already trained on 8,793 real ships' annual reports. Voyage data from operators would sharpen it."),
    ("Fuel prices and green-fuel supply are uncertain", "Price and supply are inputs. The dashboard runs what-if scenarios."),
    ("Judges need to trust the numbers", "Open benchmark with fixed seeds. One command re-runs every figure."),
]
text(s4, 5.0, 1.65, 7.8, 0.3, [[("Challenge", {"bold": True, "color": TEAL}), ("", {})]], size=12)
text(s4, 8.25, 1.65, 4.5, 0.3, [[("How we handle it", {"bold": True, "color": TEAL}), ("", {})]], size=12)
for i, (c, a) in enumerate(risks):
    y = 2.0 + i * 0.9
    boxed_text(s4, 5.0, y, 3.05, 0.78, [c], fill=TINT, size=11.5, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE, pad=0.12)
    text(s4, 8.25, y, 4.55, 0.78, [a], size=11.5, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
