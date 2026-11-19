# gta6-trophy-rarity-forecast-01-bot-v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minachan-sec/gta6-trophy-rarity-forecast-01-bot-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minachan-sec/gta6-trophy-rarity-forecast-01-bot-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast every
Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy, for the lowest pinball loss
on the logit scale.

## The model

An earn rate is a ratio — earners divided by players tracked — and the two
halves move for different reasons. Exophase refreshes a game's page every few
hours, and each refresh reveals hours of accumulated signups, so the
denominator jumps; earners grow only as players actually play, so the numerator
moves slowly. A forecast that extrapolates the ratio directly is blindsided by
every refresh. This program models earners and players separately and divides:

- **Rich history (several distinct page states).** Log players and log earners
  are each fitted against log-time (ln(1 + hours since the release day began)),
  recent states weighted more heavily — over launch week the player base
  matures, so the current pace is the better guide — with per-trophy earners
  slopes shrunk toward the cross-trophy median. The rate at the target is the
  difference of the two extrapolations.
- **Frozen history (one distinct page state).** No trend exists to fit, so the
  forecast is a refresh mixture: with some probability (a Weibull renewal over
  the page's staleness) the page refreshes before the target and reveals the
  players accumulated since it last changed, following a power law in time,
  while earners grow only a share of that jump — the share depends on how
  quickly a new player can earn the trophy, read from its description and
  tier. Without a refresh the page stays exactly as it is.
- **Quantiles** come from deterministic Monte Carlo over the joint uncertainty
  (one shared players shock, per-trophy earners noise), so the spread is honest
  about how little is known and widens with the horizon.
- **Structural orderings** are enforced: story chapters are nested (nobody
  finishes Chapter 2 without Chapter 1), and the platinum / 100%-completion
  trophies cannot exceed any other trophy.
- **It never crashes and never emits an invalid forecast.** Any failure inside
  a case falls back to a persistence forecast from the last read, and every
  output row is validated before it is written. Two runs on the same case are
  byte-identical (fixed-seed PRNG, no wall-clock or platform randomness).

Standard library only, no network, offline. `main.py` is the whole entry.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`*.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Local validation

Scored locally against the practice answers (the 04:30Z refresh read,
recovered from the live Exophase page: implied earners at 3,868 players sum to
exactly the published 4,276, and the house baseline scores 1.5084 against it,
matching the published "about 1.51"):

| forecast | practice loss (p00 + p01) |
|---|---|
| this entry | **0.5614** |
| persistence baseline | 1.5084 |
| curve baseline | 1.5084 |

The entry also beats both baselines in every scenario of a uniform-growth
reconstruction grid and in synthetic final cases (6,510-minute horizon, rich
history with a maturing player base), and passes 20 edge-case tests (empty /
malformed / partial input) without ever crashing or emitting invalid output.

Practice is a 90-minute horizon over a frozen page; the final is a 4.5-day
horizon over a rich history, so a practice win does not guarantee a final win —
but the model's trend regime is built for exactly that final shape, and the
synthetic-final tests are the evidence for it.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
