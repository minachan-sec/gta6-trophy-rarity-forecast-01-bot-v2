"""Forecast GTA VI trophy earn rates on Exophase, as five quantiles per trophy.

Model (built from first principles for this competition; see README for the
reasoning in prose):

  * An earn rate is a ratio: earners / players_tracked. The two move for
    different reasons -- players jump when Exophase refreshes its page and
    reveals hours of accumulated signups, while earners grow only as players
    actually play. So the forecast models earners and players separately and
    divides, instead of extrapolating the ratio directly. That is what makes
    the forecast survive a refresh: a refresh dilutes most rates because the
    denominator jumps far more than the numerator.
  * When the case's history shows several distinct page states, both series
    are fitted in log-log time (log count against ln(1 + hours since the
    release day began)), with per-trophy earners slopes shrunk toward the
    cross-trophy median (empirical Bayes), and extrapolated to the target.
    The fits weight recent states more heavily: over launch week the player
    base matures, so the current pace of earning -- not the average pace
    since midnight on release day -- is what the next days resemble.
  * When the history is frozen (one distinct page state), no trend exists to
    fit, so the forecast is a refresh mixture: with some probability the page
    refreshes before the target and reveals the players accumulated since the
    page last changed (a power law in time), with earners growing only a
    fraction of that -- the fraction depends on how quickly a new player can
    earn the trophy, read from its description and tier. Otherwise the page
    stays exactly as it is.
  * Quantiles come from deterministic Monte Carlo over the joint uncertainty
    (shared players shock, per-trophy earners noise), so the spread is honest
    about how little is known, and wider the further the target is.
  * Structural orderings are enforced: story chapters are nested (nobody
    finishes Chapter 2 without Chapter 1), and the platinum / 100%-completion
    trophies cannot exceed any other trophy.
  * The program never crashes and never emits an invalid forecast: any
    failure inside a case falls back to a persistence forecast from the last
    read, and every output row is validated before it is written.

Standard library only, no network, deterministic (its own fixed-seed PRNG).
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes *.out to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The release day begins here; log-time is measured from it, as the house
# curve baseline does.
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Monte Carlo replicates per trophy. Odd, so the median is a real sample.
N_SAMPLES = 2001
# Fixed seed: two runs on the same case must be byte-identical.
SEED = 20261119

# Priors for the flat-history (refresh-mixture) regime. These are judgements
# about launch-week behaviour, not fits to any answer:
#   - Exophase refreshes a game's page every few hours, somewhat regularly
#     (a Weibull renewal with shape 2 and scale ~4.5 h).
#   - Tracked players grow roughly like a power law in time since release,
#     exponent ~1.1 (fast on launch day, slowing later).
#   - At a refresh, earners of a trophy grow only a fraction of the players
#     jump: most newly tracked players have barely played, so only the
#     trophies a fresh player can earn within hours keep pace.
REFRESH_SHAPE = 2.0
REFRESH_SCALE_HOURS = 4.5
PLAYERS_EXPONENT_PRIOR = 1.1
PLAYERS_EXPONENT_SD = 0.35
EARNERS_NOISE_SD = 0.30
# Floor on the trend regime's per-trophy log spread: extrapolating days ahead
# from days of data is uncertain even when the fit looks clean.
TREND_SPREAD_FLOOR = 0.35
# The intro trophy (every tracked player holds it) is forecast at 100%, but
# with a small chance that some newly tracked players have not earned it yet;
# the logit scale punishes a 100% lower quantile brutally if the truth is
# 99.9%, so the hedge is cheap insurance.
INTRO_DILUTION_PROB = 0.10
INTRO_DILUTION_MAX = 0.03


# --------------------------------------------------------------------------
# Deterministic randomness: splitmix64 + Box-Muller, no dependence on the
# platform's `random` module (identical output on any Python).
# --------------------------------------------------------------------------

class Rng:
    """Small self-contained PRNG with uniform and normal draws."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def _next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        # 53 random bits -> [0, 1)
        return (self._next_u64() >> 11) / 9007199254740992.0

    def normal(self):
        # Box-Muller; u1 is kept away from 0 so log() stays finite.
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        return radius * math.cos(2.0 * math.pi * u2)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def clamp_pct(x):
    """Clamp a percent into the scorer's working range."""
    if x != x or x in (float("inf"), float("-inf")):
        return 1.0
    return min(100.0, max(0.01, x))


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp; None when it does not parse."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_since_release(moment):
    """Hours from the release day's start; negative (pre-release) reads are 0."""
    if moment is None:
        return 0.0
    return max(0.0, (moment - RELEASE_START).total_seconds() / 3600.0)


def log_time(moment):
    """The log-time axis: ln(1 + hours since the release day began)."""
    return math.log1p(hours_since_release(moment))


def round4(x):
    """Round to 4 decimals, halves up; whole numbers come back as ints so the
    JSON writes 100 rather than 100.0."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def ols(xs, ys):
    """Ordinary least squares of ys on xs, recent points weighted more.

    Weights ramp from 0.5 at the earliest x to 2.0 at the latest: over launch
    week the process accelerates as the player base matures, so the current
    slope is a better guide to the next days than the average slope since
    release. Returns (slope, intercept, resid_sd, slope_se, x_mean, sxx),
    or None when the fit is impossible (fewer than 2 points or no spread).
    """
    n = len(xs)
    if n < 2:
        return None
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    ws = [0.5 + 1.5 * ((x - lo) / span) ** 2 for x in xs]
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n >= 3:
        rss = sum(w * (y - (intercept + slope * x)) ** 2
                  for w, x, y in zip(ws, xs, ys)) / (sw / n)
        resid_var = rss / (n - 2)
    else:
        resid_var = 0.25  # two points fit exactly; do not trust that
    resid_sd = math.sqrt(max(resid_var, 0.0))
    slope_se = resid_sd / math.sqrt(sxx) if sxx > 0 else 1.0
    return (slope, intercept, resid_sd, slope_se, mx, sxx)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def robust_sd(values):
    """A standard-deviation estimate from the interquartile range."""
    vals = sorted(values)
    n = len(vals)
    if n < 4:
        return 0.0
    q1 = vals[n // 4]
    q3 = vals[(3 * n) // 4]
    return max(0.0, (q3 - q1) / 1.349)


# --------------------------------------------------------------------------
# Reading the case
# --------------------------------------------------------------------------

def distinct_states(history):
    """Collapse the history to distinct page states.

    Exophase's page sits frozen between refreshes, so long runs of identical
    reads are one observation, not many. A state is (time, players, rates).
    """
    states = []
    prev_key = None
    for read in history or []:
        try:
            players = read.get("players_tracked")
            rates = read.get("rates") or []
            key = (players, tuple(rates))
        except AttributeError:
            continue
        if key == prev_key:
            continue
        prev_key = key
        states.append({
            "at": parse_time(read.get("at")),
            "players": players if isinstance(players, (int, float)) and players > 0 else None,
            "rates": list(rates),
        })
    return states


def last_rates(trophies, history):
    """The most recent rate per trophy (None where never seen)."""
    rates = [None] * len(trophies)
    for read in reversed(history or []):
        rs = read.get("rates") or []
        for i in range(len(trophies)):
            if rates[i] is None and i < len(rs) and rs[i] is not None:
                rates[i] = rs[i]
        if all(r is not None for r in rates):
            break
    return rates


# --------------------------------------------------------------------------
# How fast a trophy's earners grow at a refresh, from its description
# --------------------------------------------------------------------------

_EARLY_WORDS = ("chapter 1", "once", "arcade", "fishing", "discover",
                "enjoy a cocktail", "follow-through", "first")
_LATE_WORDS = ("final mission", "every", "all ", "100%", "250", "100 ",
               "50 challenges", "20 different")


def earners_share(trophy):
    """The fraction of a players jump that a trophy's earners keep up with.

    Returns a number in [0, 1]: 1 means earners track players one-for-one
    (the intro trophy every new player earns at once); near 0 means the
    trophy's earners barely move at a refresh (a grind or endgame trophy).
    Read from the trophy's tier and description only.
    """
    tier = (trophy.get("tier") or "").lower()
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    if tier == "platinum" or "100%" in text:
        return 0.02
    share = 0.12
    if "chapter 1" in text:
        share = 0.40
    elif "chapter" in text:
        share = 0.14
    if any(w in text for w in _EARLY_WORDS):
        share = max(share, 0.40)
    if any(w in text for w in _LATE_WORDS):
        share = min(share, 0.06)
    if tier == "gold":
        share = min(share, 0.08)
    elif tier == "silver":
        share = min(share, 0.10)
    return share


def story_chains(trophies):
    """Ordered chains of trophies whose earners are nested by construction.

    Story chapters must be earned in order, and the final mission comes after
    them; the platinum and the 100%-completion trophy need every other trophy.
    Returns a list of index chains (earliest first) plus the indices that sit
    below every other trophy.
    """
    chapters = {}
    final_idx = None
    below_all = []
    for i, trophy in enumerate(trophies):
        desc = (trophy.get("description") or "").lower()
        tier = (trophy.get("tier") or "").lower()
        m = re.search(r"chapter\s+(\d+)", desc)
        if m:
            chapters[int(m.group(1))] = i
        elif "final mission" in desc and "complete the final mission" in desc:
            final_idx = i
        if tier == "platinum" or "100% completion" in desc:
            below_all.append(i)
    chains = []
    if len(chapters) >= 2:
        ordered = [chapters[k] for k in sorted(chapters)]
        if final_idx is not None:
            ordered.append(final_idx)
        chains.append(ordered)
    return chains, below_all


# --------------------------------------------------------------------------
# Quantiles from Monte Carlo samples
# --------------------------------------------------------------------------

def sample_quantiles(samples, levels):
    """Empirical quantiles of a sample list, at the given levels."""
    ordered = sorted(samples)
    n = len(ordered)
    out = []
    for tau in levels:
        pos = tau * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)
    return out


# --------------------------------------------------------------------------
# Regime 1: the history is frozen -- a refresh mixture
# --------------------------------------------------------------------------

def refresh_probability(staleness_h, horizon_h, scale=REFRESH_SCALE_HOURS):
    """P(at least one refresh before the target) under a Weibull renewal.

    The page has already sat unchanged for `staleness_h` hours; the longer it
    has sat, the more due a refresh is (shape 2 = rising hazard).
    """
    s = max(0.0, staleness_h) / scale
    t = max(0.0, staleness_h + horizon_h) / scale
    p = 1.0 - math.exp(-(t ** REFRESH_SHAPE) + (s ** REFRESH_SHAPE))
    return min(0.97, max(0.02, p))


def mixture_forecast(rng, last, shares, horizon_h, staleness_h,
                     last_change_h, target_h, levels,
                     players_mult=None, tau_scale=REFRESH_SCALE_HOURS):
    """Sample the refresh mixture for every trophy.

    Each replicate: with probability p the page refreshes before the target
    and reveals the players accumulated since the page last changed (a power
    law in time, exponent drawn around the prior); earners grow only their
    share of that jump. Without a refresh the page is exactly as it is.
    `players_mult`, when given, is (median, log_sd) for the players jump,
    used when an observed jump calibrates the mixture instead of the prior.
    Returns {trophy_index: [quantiles]}.
    """
    n = len(last)
    p_refresh = refresh_probability(staleness_h, horizon_h, tau_scale)
    # Log-time span the refresh would reveal.
    span = math.log1p(max(0.0, target_h)) - math.log1p(max(0.0, last_change_h))
    span = max(span, 0.0)
    samples = [[] for _ in range(n)]
    for _ in range(N_SAMPLES):
        if rng.uniform() >= p_refresh:
            for i in range(n):
                samples[i].append(last[i])
            continue
        if players_mult is not None:
            log_mult = rng.normal() * players_mult[1] + math.log(players_mult[0])
        else:
            exponent = PLAYERS_EXPONENT_PRIOR + rng.normal() * PLAYERS_EXPONENT_SD
            log_mult = max(0.0, exponent) * span
        for i in range(n):
            base = last[i]
            if base is None:
                base = 1.0
            if base >= 99.99:
                # The intro trophy: earners track players, so the rate holds
                # at 100% -- except a small dilution risk (see header).
                value = 100.0
                if rng.uniform() < INTRO_DILUTION_PROB:
                    value = 100.0 * (1.0 - rng.uniform() * INTRO_DILUTION_MAX)
                samples[i].append(value)
                continue
            log_earners = shares[i] * log_mult + rng.normal() * EARNERS_NOISE_SD
            log_earners = max(0.0, log_earners)  # earners never decrease
            samples[i].append(base * math.exp(log_earners - log_mult))
    rows = {}
    for i in range(n):
        rows[i] = sample_quantiles(samples[i], levels)
    return rows


# --------------------------------------------------------------------------
# Regime 2: the history shows a trend -- fit earners and players, divide
# --------------------------------------------------------------------------

def trend_forecast(rng, trophies, states, last, x_target, levels):
    """Sample the trend forecast for every trophy.

    log players and log earners are each fitted against log-time; the rate at
    the target is their difference. Earners slopes are shrunk toward the
    cross-trophy median. The players draw is shared across trophies within a
    replicate (one common shock).
    """
    n = len(trophies)
    xs = [log_time(s["at"]) for s in states]
    x_last = xs[-1]

    # --- players fit, slope shrunk toward the launch-week prior ---
    p_points = [(x, math.log(s["players"])) for x, s in zip(xs, states)
                if s["players"]]
    players_pred = None  # (mean log P at target, sd)
    players_slope = PLAYERS_EXPONENT_PRIOR
    if len(p_points) >= 2:
        fit = ols([p[0] for p in p_points], [p[1] for p in p_points])
        if fit:
            slope, intercept, resid_sd, slope_se, mx, sxx = fit
            # Shrink the slope toward the prior (a 2-day fit extrapolated to
            # day 7 overstates its own certainty).
            prior_var = 0.5 ** 2
            se2 = max(slope_se, 0.05) ** 2
            slope = (slope / se2 + PLAYERS_EXPONENT_PRIOR / prior_var) / (
                1.0 / se2 + 1.0 / prior_var)
            players_slope = slope
            last_log_p = math.log(states[-1]["players"])
            mean = last_log_p + slope * (x_target - x_last)
            # Cap the implied players growth: launch week is explosive, but
            # a fit must not promise a hundredfold jump.
            mean = min(mean, last_log_p + math.log(60.0))
            var = resid_sd ** 2 + (slope_se * (x_target - x_last)) ** 2
            players_pred = (mean, math.sqrt(max(var, 0.04)))
    if players_pred is None:
        last_p = states[-1]["players"] or 1000
        players_pred = (math.log(last_p) + PLAYERS_EXPONENT_PRIOR *
                        (x_target - x_last), 0.8)

    # --- per-trophy earners fits ---
    earners_pts = []
    for i in range(n):
        pts = []
        for x, s in zip(xs, states):
            if not s["players"] or i >= len(s["rates"]):
                continue
            rate = s["rates"][i]
            if rate is None or rate <= 0:
                continue
            earners = rate * s["players"] / 100.0
            if earners > 0:
                pts.append((x, math.log(earners)))
        earners_pts.append(pts)

    raw_slopes = []
    fits = []
    for pts in earners_pts:
        if len(pts) >= 3:
            fit = ols([p[0] for p in pts], [p[1] for p in pts])
        elif len(pts) == 2 and pts[1][0] > pts[0][0]:
            slope = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
            fit = (slope, pts[0][1] - slope * pts[0][0], 0.3, 0.5, 0.0, 1.0)
        else:
            fit = None
        fits.append(fit)
        if fit:
            raw_slopes.append(fit[0])
    pooled_slope = median(raw_slopes) if raw_slopes else players_slope * 0.3
    tau = max(robust_sd(raw_slopes), 0.15)
    pooled_resid = median([f[2] for f in fits if f]) if any(fits) else 0.3

    # Last earners per trophy (half a player when nothing was ever earned:
    # the log needs something positive, and the spread will be wide).
    last_earners = []
    for i in range(n):
        value = 0.5
        for s in reversed(states):
            if s["players"] and i < len(s["rates"]) and s["rates"][i]:
                value = s["rates"][i] * s["players"] / 100.0
                break
        last_earners.append(max(value, 0.5))

    # Per-trophy shrunk slope + spread at the target.
    trophy_params = []
    for i in range(n):
        fit = fits[i]
        if fit:
            slope_hat, _icept, resid_sd, slope_se, _mx, _sxx = fit
            w = tau * tau / (tau * tau + max(slope_se, 0.05) ** 2)
            slope = w * slope_hat + (1.0 - w) * pooled_slope
            resid = 0.5 * resid_sd + 0.5 * pooled_resid
            se = slope_se
        else:
            slope = pooled_slope
            resid = pooled_resid
            se = 0.5
        slope = min(max(slope, -1.0), players_slope + 2.0)
        mean_log_e = math.log(last_earners[i]) + slope * (x_target - x_last)
        var = resid ** 2 + (se * (x_target - x_last)) ** 2 + TREND_SPREAD_FLOOR ** 2
        trophy_params.append((mean_log_e, math.sqrt(min(var, 4.0))))

    # --- Monte Carlo: shared players draw, per-trophy earners draws ---
    samples = [[] for _ in range(n)]
    mean_log_p, sd_log_p = players_pred
    for _ in range(N_SAMPLES):
        log_p = mean_log_p + rng.normal() * sd_log_p
        for i in range(n):
            if last[i] is not None and last[i] >= 99.99:
                value = 100.0
                if rng.uniform() < INTRO_DILUTION_PROB:
                    value = 100.0 * (1.0 - rng.uniform() * INTRO_DILUTION_MAX)
                samples[i].append(value)
                continue
            mean_e, sd_e = trophy_params[i]
            log_e = mean_e + rng.normal() * sd_e
            samples[i].append(100.0 * math.exp(log_e - log_p))
    rows = {}
    for i in range(n):
        rows[i] = sample_quantiles(samples[i], levels)
    return rows


# --------------------------------------------------------------------------
# Fallback: persistence with a random-walk spread (never invalid, never a crash)
# --------------------------------------------------------------------------

def fallback_rows(trophies, last, horizon_minutes, levels):
    spread = 0.5 * math.sqrt(max(horizon_minutes, 0) / 1440.0)
    # Standard-normal quantiles at the requested levels (the five contest
    # levels have well-known values; anything else interpolates between them).
    known = {0.05: -1.6449, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.95: 1.6449}
    rows = {}
    for i, _t in enumerate(trophies):
        rate = last[i] if last[i] is not None else 1.0
        p = min(99.99, max(0.01, rate)) / 100.0
        centre = math.log(p / (1.0 - p))
        qs = []
        for tau in levels:
            zv = known.get(round(tau, 2))
            if zv is None:
                # Linear interpolation across the known points.
                pts = sorted(known.items())
                if tau <= pts[0][0]:
                    zv = pts[0][1]
                elif tau >= pts[-1][0]:
                    zv = pts[-1][1]
                else:
                    for (t0, z0), (t1, z1) in zip(pts, pts[1:]):
                        if t0 <= tau <= t1:
                            zv = z0 + (z1 - z0) * (tau - t0) / (t1 - t0)
                            break
            qs.append(100.0 / (1.0 + math.exp(-(centre + zv * spread))))
        rows[i] = qs
    return rows


# --------------------------------------------------------------------------
# One case, end to end
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(t) for t in levels]
    horizon_minutes = case.get("horizon_minutes")
    if not isinstance(horizon_minutes, (int, float)):
        horizon_minutes = 90
    horizon_h = max(0.0, horizon_minutes / 60.0)

    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if target is None and at is not None:
        from datetime import timedelta
        target = at + timedelta(minutes=horizon_minutes)
    target_h = hours_since_release(target)
    x_target = log_time(target)

    last = last_rates(trophies, history)
    states = distinct_states(history)
    rng = Rng(SEED)
    shares = [earners_share(t) for t in trophies]

    rows = None
    if len(states) >= 3:
        rows = trend_forecast(rng, trophies, states, last, x_target, levels)
    elif len(states) == 2:
        # One observed jump calibrates the mixture: blend what the jump showed
        # with the priors, half and half.
        first, second = states
        staleness = 0.0
        if at is not None and second["at"] is not None:
            staleness = max(0.0, (at - second["at"]).total_seconds() / 3600.0)
        gap_h = 4.5
        if first["at"] is not None and second["at"] is not None:
            gap_h = max(0.5, (second["at"] - first["at"]).total_seconds() / 3600.0)
        tau_scale = 0.5 * gap_h + 0.5 * REFRESH_SCALE_HOURS
        mult = None
        if first["players"] and second["players"] and second["players"] > first["players"]:
            observed = second["players"] / first["players"]
            span = (math.log1p(hours_since_release(second["at"])) -
                    math.log1p(hours_since_release(first["at"])))
            prior = math.exp(max(0.0, PLAYERS_EXPONENT_PRIOR) * max(span, 0.0))
            mult = (math.sqrt(observed * max(prior, 1.0)), 0.55)
            log_obs = math.log(observed)
            adj_shares = []
            for i in range(len(trophies)):
                e1 = e2 = None
                if i < len(first["rates"]) and first["rates"][i]:
                    e1 = first["rates"][i] * first["players"] / 100.0
                if i < len(second["rates"]) and second["rates"][i]:
                    e2 = second["rates"][i] * second["players"] / 100.0
                if e1 and e2 and e2 > 0 and log_obs > 0.05:
                    obs_share = math.log(e2 / e1) / log_obs
                    adj_shares.append(0.5 * min(max(obs_share, 0.0), 1.0) +
                                      0.5 * shares[i])
                else:
                    adj_shares.append(shares[i])
            shares = adj_shares
        rows = mixture_forecast(
            rng, last, shares, horizon_h, staleness,
            hours_since_release(second["at"]), target_h, levels,
            players_mult=mult, tau_scale=tau_scale)
    elif len(states) == 1:
        first_read = parse_time((history[0].get("at") if history else None))
        staleness = 0.0
        if at is not None and first_read is not None:
            staleness = max(0.0, (at - first_read).total_seconds() / 3600.0)
        last_change_h = hours_since_release(first_read)
        rows = mixture_forecast(rng, last, shares, horizon_h, staleness,
                                last_change_h, target_h, levels)
    if rows is None:
        rows = fallback_rows(trophies, last, horizon_minutes, levels)

    # --- structural orderings: nested trophies, per quantile ---
    chains, below_all = story_chains(trophies)
    for chain in chains:
        for k in range(1, len(chain)):
            earlier, later = chain[k - 1], chain[k]
            rows[later] = [min(a, b) for a, b in zip(rows[later], rows[earlier])]
    if below_all and len(trophies) > 1:
        for idx in below_all:
            others = [j for j in range(len(trophies)) if j != idx]
            floor = [min(rows[j][k] for j in others) for k in range(len(levels))]
            rows[idx] = [min(a, b) for a, b in zip(rows[idx], floor)]

    # --- final validation: finite, in range, non-decreasing, rounded ---
    out_rows = []
    for i, trophy in enumerate(trophies):
        qs = rows.get(i) or [1.0] * len(levels)
        clean = []
        prev = 0.01
        for q in qs:
            v = clamp_pct(q if isinstance(q, (int, float)) else 1.0)
            v = max(v, prev)
            clean.append(round4(v))
            prev = clean[-1]
        out_rows.append({"id": trophy.get("id"), "q": clean})
    return {"trophies": out_rows}


def safe_forecast_case(case):
    """forecast_case with a total safety net: never raises, never invalid."""
    try:
        result = forecast_case(case)
        # Shape-check the result before trusting it.
        rows = result.get("trophies")
        want = len(case.get("trophies") or [])
        n_levels = len(case.get("levels") or DEFAULT_LEVELS)
        if not isinstance(rows, list) or len(rows) != want:
            raise ValueError("row count mismatch")
        for row in rows:
            qs = row.get("q")
            if (not isinstance(qs, list) or len(qs) != n_levels or
                    any(not isinstance(q, (int, float)) or q != q or
                        q in (float("inf"), float("-inf")) or q < 0 or q > 100
                        for q in qs)):
                raise ValueError("bad quantile row")
        return result
    except Exception:
        try:
            trophies = case.get("trophies") or []
            last = last_rates(trophies, case.get("history") or [])
            horizon = case.get("horizon_minutes")
            if not isinstance(horizon, (int, float)):
                horizon = 90
            rows = fallback_rows(trophies, last, horizon, DEFAULT_LEVELS)
            out = []
            for i, trophy in enumerate(trophies):
                qs = [round4(clamp_pct(q)) for q in rows[i]]
                qs = sorted(qs)
                out.append({"id": trophy.get("id"), "q": qs})
            return {"trophies": out}
        except Exception:
            return {"trophies": []}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return
    try:
        names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    except Exception:
        return
    for name in names:
        case_id = name[:-3]
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            if not isinstance(case, dict):
                case = {}
        except Exception:
            case = {}
        try:
            forecast = safe_forecast_case(case)
            payload = json.dumps(forecast, separators=(",", ":")) + "\n"
            if len(payload.encode("utf-8")) > 65536:
                # Should never happen (55 trophies is ~4 KB); shrink by
                # dropping to 2 decimals rather than emitting an oversize file.
                slim = {"trophies": [
                    {"id": r["id"], "q": [round(q, 2) for q in r["q"]]}
                    for r in forecast["trophies"]]}
                payload = json.dumps(slim, separators=(",", ":")) + "\n"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        except Exception:
            # Last resort: still try to leave a file behind.
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write('{"trophies":[]}\n')
            except Exception:
                pass


if __name__ == "__main__":
    main()
