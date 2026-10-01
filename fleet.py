"""SagarMitra prototype: green fleet deployment model.

One decision per route: vessel type k, fuel f, cruising speed s (knots).
Everything else follows from that:
  voyages/week  V = ceil(D / cap_k)
  energy/voyage E = a_k * s^2 * 2d * 0.0036 / eta   (GJ)   -- admiralty law, P = a s^3
  ships needed  n = ceil(V * (2d/s + port_h + wait) / 168)
  cost/week     = n * (charter_k + alt_premium_f) + fuel cost + carbon_price * emissions
  emissions     = V * E * EF_f / 1000   (t CO2e, well-to-wake)
Side constraints: ships of each type available, weekly supply of green fuels,
and a carbon-intensity limit per (k, f, s) in the spirit of IMO CII.

Linear model (nonlinear=False): no port waiting, fixed fuel prices.
Nonlinear model (nonlinear=True) adds two couplings between routes:
  port congestion   every voyage is one call at each end; wait per call = W0 * rho / (1 - rho),
                    rho = calls at that port / port capacity, so small ships crowd ports
  volume pricing    green fuels get dearer as the fleet buys more of them:
                    marginal price = p0 * (1 + Q / Qref), so cost = p0 * (Q + Q^2 / (2 Qref))

All numbers are indicative, chosen to be the right order of magnitude. They are
not market data.
"""
import numpy as np

ETA = 0.45          # engine efficiency
PORT_H = 36.0       # port hours per round trip, before any queueing
SPEED_MIN, SPEED_MAX = 9.0, 18.0
W0 = 6.0            # hours: base wait per call in the queueing formula
RHO_MAX = 0.97      # port utilisation above this counts as a violation

# vessel types: capacity (t), weekly charter ($k), admiralty coefficient a (kW / kn^3)
CAP = np.array([3000, 6000, 12000, 25000, 40000], float)
CHARTER = np.array([45, 70, 105, 160, 210], float)
A = 0.73 * (CAP / 3000) ** (2 / 3)
AVAIL = np.array([4, 4, 3, 3, 2], float)
ALT_OK_MIN_TYPE = 1  # type 0 (smallest) cannot take alternative fuels

# fuels: price ($k/GJ), well-to-wake EF (g CO2e/MJ), weekly premium per ship ($k), weekly supply (GJ, inf = unlimited)
FUELS = ["VLSFO", "LNG", "Grey methanol", "Bio-methanol", "Grey ammonia", "Green ammonia"]
PRICE = np.array([15, 13, 18, 34, 20, 42], float) / 1000
EF = np.array([92, 79, 98, 16, 118, 9], float)
PREMIUM = np.array([0, 3, 2, 2, 4, 4], float)
IS_ALT = np.array([0, 1, 1, 1, 1, 1], bool)
SUPPLY = np.array([np.inf, np.inf, np.inf, 60000, np.inf, 45000], float)
GREEN = np.isfinite(SUPPLY)  # the volume-priced, supply-limited fuels

CII_LIMIT = 2.2e-5  # t CO2e per tonne-nm at design load (binds on small, fast ships)
REAL_PORT_CAP = 8.0  # ship calls per week each port can take


def real_routes():
    # approximate sea distances (nm) and weekly demand (t) on an Indian coastal network
    legs = [
        ("JNPT", "Mundra", 420, 42000),
        ("JNPT", "Kochi", 580, 30000),
        ("Kochi", "Tuticorin", 300, 16000),
        ("Tuticorin", "Chennai", 420, 20000),
        ("Chennai", "Visakhapatnam", 330, 34000),
        ("Visakhapatnam", "Paradip", 260, 46000),
        ("Paradip", "Haldia", 210, 38000),
        ("Mundra", "Kochi", 950, 22000),
    ]
    names = [f"{a}-{b}" for a, b, _, _ in legs]
    d = np.array([l[2] for l in legs], float)
    D = np.array([l[3] for l in legs], float)
    return names, d, D


def real_ports():
    names, _, _ = real_routes()
    ports = []
    for n in names:
        for p in n.split("-"):
            if p not in ports:
                ports.append(p)
    a = np.array([ports.index(n.split("-")[0]) for n in names])
    b = np.array([ports.index(n.split("-")[1]) for n in names])
    return ports, a, b, np.full(len(ports), REAL_PORT_CAP)


def synthetic_routes(n, seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(180, 1100, n).round()
    D = rng.uniform(8000, 50000, n).round(-2)
    return [f"R{i}" for i in range(n)], d, D


def synthetic_ports(D, seed, per_port=4):
    """Random port pairs, ~per_port routes touching each port, capacity set so mid-size ships run near 60% load."""
    rng = np.random.default_rng(seed + 1000)
    R = len(D)
    P = max(8, (2 * R) // per_port)
    a = rng.integers(0, P, R)
    b = (a + rng.integers(1, P, R)) % P
    calls_mid = np.zeros(P)
    V_mid = np.ceil(D / CAP[2])
    np.add.at(calls_mid, a, V_mid)
    np.add.at(calls_mid, b, V_mid)
    cap = np.maximum(4.0, np.ceil(calls_mid / 0.6))
    return [f"P{i}" for i in range(P)], a, b, cap


class Instance:
    def __init__(self, d, D, carbon_price=0.1, scale_avail=1.0, scale_supply=1.0, names=None,
                 ports=None, nonlinear=False):
        self.d, self.D, self.cp = d, D, carbon_price  # carbon price in $k per t
        self.R = len(d)
        self.avail = AVAIL * scale_avail
        self.supply = SUPPLY * scale_supply
        self.names = names
        self.nonlinear = nonlinear
        if ports is not None:
            self.port_names, self.pa, self.pb, self.port_cap = ports
            self.P = len(self.port_cap)

    # ---- pieces used by every solver
    def voyages(self, k):
        return np.ceil(self.D / CAP[k])

    def waits(self, k):
        """Queueing hours per round trip for each route (two calls), and port overload. Batch (..., R)."""
        if not self.nonlinear:
            z = np.zeros(np.shape(k), float)
            return z, np.zeros(np.shape(k)[:-1])
        V = self.voyages(k)
        onehot_a = np.eye(self.P)[self.pa]  # (R, P)
        onehot_b = np.eye(self.P)[self.pb]
        calls = V @ onehot_a + V @ onehot_b  # (..., P)
        rho = calls / self.port_cap
        over = np.maximum(0, rho - RHO_MAX).sum(-1) * 10
        r = np.minimum(rho, RHO_MAX)
        wport = W0 * r / (1 - r)
        W = wport[..., self.pa] + wport[..., self.pb]
        return W, over

    def route_terms(self, k, f, s, W=0.0, idx=None):
        d = self.d if idx is None else self.d[idx]
        D = self.D if idx is None else self.D[idx]
        V = np.ceil(D / CAP[k])
        E = A[k] * s**2 * 2 * d * 0.0036 / ETA
        n = np.ceil(V * (2 * d / s + PORT_H + W) / 168.0)
        fuel_gj = V * E
        cost = n * (CHARTER[k] + PREMIUM[f]) + fuel_gj * PRICE[f]  # fuel at base price
        emis = fuel_gj * EF[f] / 1000.0
        return n, fuel_gj, cost, emis

    def intensity(self, k, f, s):
        return EF[f] / 1e6 * A[k] * s**2 * 0.0036 / ETA * 1000 / CAP[k] * 1.0

    def allowed(self, k, f, s):
        ok = self.intensity(k, f, s) <= CII_LIMIT
        ok &= ~(IS_ALT[f] & (k < ALT_OK_MIN_TYPE))
        return ok

    def evaluate(self, k, f, s):
        """Returns objective, cost, emissions, violation for a batch (..., R)."""
        W, over = self.waits(k)
        n, gj, cost, emis = self.route_terms(k, f, s, W)
        total_cost = cost.sum(-1)
        viol = over.copy() if np.ndim(over) else np.asarray(over, float)
        for t in range(len(CAP)):
            viol = viol + np.maximum(0, (n * (k == t)).sum(-1) - self.avail[t])
        for j in range(len(FUELS)):
            if GREEN[j]:
                Q = (gj * (f == j)).sum(-1)
                viol = viol + np.maximum(0, Q - self.supply[j]) / 1000.0
                if self.nonlinear:
                    total_cost = total_cost + PRICE[j] * Q**2 / (2 * self.supply[j])
        viol = viol + (~self.allowed(k, f, s)).sum(-1) * 10
        emis_t = emis.sum(-1)
        return total_cost + self.cp * emis_t, total_cost, emis_t, viol
