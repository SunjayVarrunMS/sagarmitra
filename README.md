# SagarMitra

Prototype for Smart India Hackathon 2026, problem statement **SIH26138**: *Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization* (Egreen Quanta). Team Prometheus 01, BITS Pilani Hyderabad Campus.

The problem statement asks for quantum-inspired methods benchmarked against conventional ones. This repo does that benchmark, on both halves of the problem, and reports what came out, including where quantum-inspired methods lost.

## Results

### 1. Fuel prediction on real ships

8,793 ships from the EU-MRV 2023 public reports. Target: annual CO2 per tonne of cargo per nautical mile. Inputs: design efficiency (EEDI, EEXI or EIV), average speed, time at sea, ship type. Split by ship, 80% train / 20% test, repeated on 5 random splits.

| Model | Error (MAPE), mean ± sd | R² (log) |
|---|---|---|
| Design-efficiency baseline (log-linear) | 30.5% ± 1.3 | 0.79 |
| Gradient boosting | 27.2% ± 0.8 | 0.82 |
| Quantum fidelity kernel (simulated) | 27.7% ± 1.6 | 0.82 |
| **Average of gradient boosting and quantum kernel** | **26.7% ± 1.0** | **0.83** |

The quantum kernel on its own is level with gradient boosting (it wins 1 split of 5). Averaging the two is better than gradient boosting alone in **5 of 5** splits.

The quantum kernel angle-encodes each input on one qubit (RY rotation) and uses the fidelity kernel k(x, y) = Π cos²((xᵢ − yᵢ)/2), simulated exactly on a normal computer, inside kernel ridge regression. Bandwidth and regularisation are tuned on a validation split of the training ships only.

Side finding: operational intensity scales almost 1:1 with the design rating (fitted exponent 1.02). On annual averages, slower average speed goes with *higher* intensity, likely because slow annual averages include idling and manoeuvring.

### 2. Fleet optimisation

One decision per route: ship type, fuel (VLSFO, LNG, grey/bio methanol, grey/green ammonia) and speed. Limits on ships available, green-fuel supply, port capacity and an IMO-CII-style carbon intensity cap. Objective: weekly cost + carbon at $100/t. 8 Indian coastal routes (JNPT–Mundra to Paradip–Haldia) plus synthetic 60- and 300-route networks; 3 seeds for every stochastic method.

**Linear model** (no port queues, fixed fuel prices): the exact solver wins.

| Routes | Exact (HiGHS, 0.25-kn grid) | Textbook QIEA + QPSO | QIEA + QPSO with repair |
|---|---|---|---|
| 8 | proven best, 0.2 s | +3.1% | +0.5% |
| 60 | proven best, 0.7 s | no valid plan | no valid plan |
| 300 | proven best, 4.1 s | no valid plan | no valid plan |

**Nonlinear model** (port queueing W₀·ρ/(1−ρ) at every call, green fuel priced by volume): the hybrid wins.

| Routes | Exact, ignoring queues | Best classical | Quantum-inspired alone | **Hybrid** | Hybrid emissions |
|---|---|---|---|---|---|
| 8 | invalid plan | valid | −0.2% | **−0.8%** | −3.5% |
| 60 | invalid plan | valid | +0.6% | **−2.6%** | −7.9% |
| 300 | invalid plan | valid | +5.3% | **−2.7%** | −7.3% |

Percentages are against the best classical plan: the MILP re-solved six times with updated port waits (sequential linearisation), then given the same greedy repair the hybrid uses. The hybrid is QIEA (ship type, fuel) + QPSO (continuous speed) warm-started from that plan, with elitism, so it cannot return a worse plan than it started from. The same result held for all 3 seeds.

## Run it

```
pip install numpy scipy pandas openpyxl scikit-learn
python bench.py          # fleet optimisation benchmark -> results.json (about 10 minutes)
python summary.py        # prints results.json as tables
python fuel_model.py     # fuel prediction -> fuel_results.json (needs the EU-MRV file, below)
```

`fuel_model.py` expects `data/eu_mrv_2023.xlsx`: the EU-MRV 2023 "Publication of information" file from [mrv.emsa.europa.eu](https://mrv.emsa.europa.eu/#public/emission-report) (Reporting period 2023, download). It is not redistributed here.

`build_deck.py` builds the SIH idea deck from both results files (needs python-pptx and the official SIH template).

## Files

| File | What it is |
|---|---|
| `fleet.py` | Fleet model: routes, ports, ship types, fuels, linear and nonlinear evaluation |
| `solvers.py` | Exact MILP (HiGHS via SciPy), sequential re-solve, greedy repair, QIEA + QPSO |
| `bench.py` | Fleet optimisation benchmark |
| `fuel_model.py` | Fuel-intensity models on EU-MRV data |
| `results.json`, `fuel_results.json` | Outputs used in the deck |

## Limits

- Sea distances are approximate. Fuel prices, charter rates, port capacities and emission factors are indicative, chosen to be the right order of magnitude, not market data.
- The 60- and 300-route networks are synthetic (fixed seeds).
- EU-MRV data is annual per ship, not per voyage, so the fuel model cannot see weather or draft yet.
- The quantum kernel and QIEA/QPSO are quantum-*inspired*: everything runs on classical hardware.
