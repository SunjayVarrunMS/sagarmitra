"""Prints results.json as two small tables."""
import json

r = json.load(open("results.json"))


def p(x):
    return "   -   " if x is None else f"{x:+6.2f}%"


print("LINEAR (no queues, fixed prices)   % vs exact on 0.25-kn grid")
print(f"{'routes':>6} {'exact 1kn':>12} {'exact .25kn':>12} {'QI textbook':>14} {'QI + repair':>14}")
for x in r["linear"]:
    t, q = x["qi_textbook"], x["qi_repair"]
    print(f"{x['routes']:>6} {x['exact_1kn']['time_s']:>10.2f}s {x['exact_025kn']['time_s']:>10.2f}s "
          f"{p(t.get('best_pct')):>9} {t['feasible_runs']}/3 {p(q.get('best_pct')):>9} {q['feasible_runs']}/3")

print("\nNONLINEAR (port queues + volume-priced green fuel)   % vs best classical")
print(f"{'routes':>6} {'no-queue exact':>16} {'classical':>18} {'QI alone':>14} {'hybrid':>14}")
for x in r["nonlinear"]:
    e, c, q, h = x["exact_ignoring_queues"], x["best_classical"], x["qi_alone"], x["hybrid"]
    print(f"{x['routes']:>6} {('valid' if e['valid'] else 'INVALID') + ' ' + p(e.get('pct_vs_classical')):>16} "
          f"{('valid' if c['valid'] else 'INVALID') + (' rep' if c['repaired'] else '') + f" {c['time_s']:.1f}s":>18} "
          f"{p(q.get('best_pct')):>9} {q['feasible_runs']}/3 {p(h.get('best_pct')):>9} {h['feasible_runs']}/3  "
          f"(mean {p(h.get('mean_pct'))}, {h.get('time_s', 0):.1f}s)")
