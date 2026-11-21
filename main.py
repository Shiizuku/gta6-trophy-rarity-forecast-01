"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's PS5 trophies, five quantiles (5/25/50/75/95%)
of its Exophase earn rate at the case's target, from the history of reads in
the case file. Python 3.13, standard library only; no network.

Model idea
----------
An Exophase earn rate is a ratio, rate = earners / players_tracked, and the
page only changes when Exophase refreshes its counts (hours apart). So the
forecast is built from the two counts, not from the ratio directly:

  * players_tracked P(t): fitted with an ensemble of growth curves
    (saturating exponential, power law, log-linear on the last day) and
    extrapolated to the target; the ensemble median is the point forecast
    and the ensemble spread feeds the uncertainty.
  * earners E_i(t) per trophy: a per-player hazard model. New earners of
    trophy i arrive at rate h_i * (P - E_i), i.e. proportional to the
    players who have not earned it yet. h_i is estimated from the case
    history (new earners over person-time at risk, shrunk toward the
    tier mean), then the hazard equation is stepped forward to the target
    under the extrapolated P(t).

The median rate is 100 * E_i(target) / P(target). Quantiles are logit-normal
around it, with a spread that combines the random-walk term, the P-ensemble
spread and the hazard sensitivity (re-running the forward step with h_i
halved and doubled). The 100% trophy (every tracked player holds it) is
forecast to stay at ~100%.

Fallback
--------
If the history never changes (the launch-night practice cases: every read
shows the same page), there is nothing to fit. The page is stale and a
refresh is due before the target, so the forecast applies a dilution prior:
the refreshed page tracks several times more players, and the newly tracked
players hold each trophy at a lower rate than the launch-night cohort, so
each rate falls to roughly a third of its last value, with a wide spread.
The exact dilution is graded by the trophy's current rate (early, common
trophies dilute less) and the spread is wide because the refresh is unseen.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch day anchor (00:00 UTC on release day), used as the origin for the
# growth curves, as the house curve baseline does.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Random-walk spread of a rate's logit move, per square-root day (same
# scale the house persistence baseline uses).
SIGMA_RW = 0.5
# Spread floor / cap, in logit units.
SIGMA_MIN = 0.35
SIGMA_MAX = 2.0
# Dilution prior for a flat (never refreshed) history: median rate becomes
# last_rate * (DILUTE_LOW + DILUTE_SLOPE * rate / max_rate).
DILUTE_LOW = 0.25
DILUTE_SLOPE = 0.20
DILUTE_SIGMA = 0.7


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lg))))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def distinct_states(case):
    """Compress the history to the reads where something changed.

    Between refreshes Exophase prints the same page, so consecutive
    identical reads carry no new information. Returns a list of
    (time_seconds, players_tracked, [rates]) with rates carried forward
    where a read lacks a trophy (null).
    """
    states = []
    last_rates = None
    for read in case["history"]:
        rates = list(read["rates"])
        if last_rates is not None:
            rates = [r if r is not None else last_rates[i]
                     for i, r in enumerate(rates)]
        else:
            rates = [r if r is not None else 0.0 for r in rates]
        p = read["players_tracked"]
        if p is None:
            p = states[-1][1] if states else 0
        if states and states[-1][1] == p and states[-1][2] == rates:
            continue
        states.append((parse_time(read["at"]), p, rates))
        last_rates = rates
    return states


def fit_players(states, t_target):
    """Extrapolate players_tracked to t_target.

    Returns (P_target, P_low, P_high, k_sat): the point forecast from the
    saturating-exponential fit (a launch's tracked players grow fast and
    then level off; a log-linear fit would compound the early growth and
    overshoot badly over a 4.5-day horizon), the ensemble range for the
    spread, and the saturating curve's rate constant for interpolation.
    """
    t0 = states[0][0]
    p_last = states[-1][1]
    if len(states) < 2 or all(s[1] == states[0][1] for s in states):
        return p_last, p_last, p_last, None
    hours = [(s[0] - LAUNCH) / 3600.0 for s in states]
    ps = [float(s[1]) for s in states]
    h_target = (t_target - LAUNCH) / 3600.0
    forecasts = []
    sat = None

    # (a) Saturating exponential P = Pinf * (1 - exp(-k * h)).
    # Grid over k; Pinf by least squares on P for each k; keep the best
    # fit in log space.
    best = None
    for k in [10 ** (e / 20.0) for e in range(-60, 21)]:
        xs = [1.0 - math.exp(-k * h) for h in hours]
        denom = sum(x * x for x in xs)
        if denom <= 0:
            continue
        pinf = sum(x * p for x, p in zip(xs, ps)) / denom
        if pinf <= 0:
            continue
        err = sum((math.log(max(1.0, pinf * x)) - math.log(p)) ** 2
                  for x, p in zip(xs, ps))
        if best is None or err < best[0]:
            best = (err, pinf, k)
    if best is not None:
        _, pinf, k = best
        sat = pinf * (1.0 - math.exp(-k * h_target))
        forecasts.append(sat)

    # (b) Power law P = a * h^b (log-log linear fit).
    ok = [i for i, h in enumerate(hours) if h > 0.5]
    if len(ok) >= 2:
        xs = [math.log(hours[i]) for i in ok]
        ys = [math.log(ps[i]) for i in ok]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            a = my - b * mx
            if 0.0 < b < 5.0:
                forecasts.append(math.exp(a + b * math.log(max(1.0, h_target))))

    # (c) Log-linear on the last 24 hours of states (spread only: it
    # compounds the early growth and overshoots over a long horizon).
    recent = [(h, p) for h, p in zip(hours, ps)
              if h >= hours[-1] - 24.0]
    if len(recent) >= 2:
        xs = [h for h, _ in recent]
        ys = [math.log(p) for _, p in recent]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            inter = my - slope * mx
            forecasts.append(math.exp(inter + slope * h_target))

    forecasts = [f for f in forecasts if f and f > 0]
    if not forecasts:
        return p_last, p_last, p_last, None
    # Point forecast: the saturating fit when it exists, else the
    # ensemble median. Never below the last read, never more than 15x it.
    point = sat if sat is not None else sorted(forecasts)[len(forecasts) // 2]
    point = min(max(point, float(p_last)), float(p_last) * 15.0)
    lo = min(max(min(forecasts), float(p_last)), float(p_last) * 15.0)
    hi = min(max(max(forecasts), float(p_last)), float(p_last) * 15.0)
    k_sat = best[2] if best is not None else None
    return point, lo, hi, k_sat


def players_at(states, p_target, t_last, t_target, t, k_sat):
    """Players tracked at time t, on the fitted growth curve's shape.

    Interpolates from the last read to the target forecast along a
    saturating exponential (the growth curve is concave: a straight line
    between the ends would understate P mid-window and so understate
    how fast earners grow). Falls back to linear when no curve was fit.
    """
    if t <= t_last:
        return float(states[-1][1])
    if t >= t_target:
        return float(p_target)
    p0 = float(states[-1][1])
    if k_sat is None or k_sat <= 0:
        frac = (t - t_last) / (t_target - t_last)
    else:
        num = 1.0 - math.exp(-k_sat * (t - t_last) / 3600.0)
        den = 1.0 - math.exp(-k_sat * (t_target - t_last) / 3600.0)
        frac = num / den if den > 0 else (t - t_last) / (t_target - t_last)
    return p0 + (float(p_target) - p0) * frac


def estimate_hazards(states, n_trophies, tiers):
    """Per-trophy earn hazard (new earners per player-hour at risk).

    h_i = new earners / person-hours at risk, computed over the distinct
    states, then shrunk toward the tier mean (trophies of a tier share
    roughly how hard they are to earn). Returns (hazards, tier_mean).
    """
    # Earners per state per trophy.
    earners = [[r / 100.0 * st[1] for r in st[2]] for st in states]
    risk_time = [0.0] * n_trophies
    new_earners = [0.0] * n_trophies
    for k in range(len(states) - 1):
        a, b = states[k], states[k + 1]
        dt_h = (b[0] - a[0]) / 3600.0
        if dt_h <= 0:
            continue
        for i in range(n_trophies):
            ea, eb = earners[k][i], earners[k + 1][i]
            pa, pb = a[1], b[1]
            risk = 0.5 * ((pa - ea) + (pb - eb))
            risk_time[i] += max(0.0, risk) * dt_h
            new_earners[i] += max(0.0, eb - ea)
    # Tier means (per hour).
    tier_h = {}
    for tier in set(tiers):
        idx = [i for i, t in enumerate(tiers) if t == tier]
        num = sum(new_earners[i] for i in idx)
        den = sum(risk_time[i] for i in idx)
        tier_h[tier] = (num / den) if den > 0 else 0.0
    hazards = []
    for i in range(n_trophies):
        # Shrink toward the tier mean with a 200 person-hour prior.
        prior = tier_h.get(tiers[i], 0.0)
        h = (new_earners[i] + prior * 200.0) / (risk_time[i] + 200.0)
        hazards.append(h)
    return hazards, tier_h


def step_earners(e0, p_fn, hazard, t_from, t_to):
    """Step E forward under dE/dt = h * (P(t) - E), hourly steps."""
    e = e0
    t = t_from
    step = 3600.0
    while t < t_to - 1e-9:
        dt = min(step, t_to - t)
        p_mid = p_fn(t + dt / 2.0)
        e += hazard * max(0.0, p_mid - e) * (dt / 3600.0)
        e = min(e, p_mid)
        t += dt
    return e


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    tiers = [t.get("tier", "bronze") for t in trophies]
    t_target = parse_time(case["target"])
    t_at = parse_time(case["at"])
    days = case["horizon_minutes"] / 1440.0
    states = distinct_states(case)
    last_rates = states[-1][2]
    p_last = states[-1][1]
    max_rate = max(last_rates) if last_rates else 1.0

    flat = all(s[1] == states[0][1] and s[2] == states[0][2] for s in states)

    rows = []
    if flat:
        # Nothing ever changed: the page is stale, a refresh is due.
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate >= 99.99:
                rows.append({"id": trophy["id"],
                             "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
                continue
            d = DILUTE_LOW + DILUTE_SLOPE * (rate / max_rate if max_rate else 0.0)
            med = clamp_pct(rate * d)
            lg = logit(med)
            qs = [round4(clamp_pct(inv_logit(lg + z * DILUTE_SIGMA))) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    p_target, p_lo, p_hi, k_sat = fit_players(states, t_target)
    hazards, _ = estimate_hazards(states, n, tiers)
    t_last = states[-1][0]
    earners_last = [r / 100.0 * p_last for r in last_rates]

    def p_fn_factory(p_t):
        return lambda t: players_at(states, p_t, t_last, t_target, t, k_sat)

    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
            continue
        e0 = earners_last[i]
        h = hazards[i]
        e_med = step_earners(e0, p_fn_factory(p_target), h, t_last, t_target)
        med = clamp_pct(100.0 * e_med / p_target)

        # Spread: random walk + P-ensemble range + hazard sensitivity.
        sig2 = (SIGMA_RW * math.sqrt(days)) ** 2
        if p_hi > p_lo and p_target > 0:
            sig2 += (0.5 * math.log(p_hi / p_lo)) ** 2
        e_lo = step_earners(e0, p_fn_factory(p_target), h * 0.5, t_last, t_target)
        e_hi = step_earners(e0, p_fn_factory(p_target), h * 2.0, t_last, t_target)
        r_lo = 100.0 * e_lo / p_target
        r_hi = 100.0 * e_hi / p_target
        if r_lo > 0 and r_hi > r_lo:
            sig2 += (0.5 * (logit(r_hi) - logit(r_lo))) ** 2
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, math.sqrt(sig2)))

        lg = logit(med)
        qs = [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in Z]
        # Rounding can only preserve order (logit is monotone in z), but
        # enforce it explicitly so the output is always valid.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
