[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Shiizuku/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Shiizuku/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Entry for [make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for each of the game's 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate one week after launch (2026-11-26T00:00:00Z), from the house's recording of the trophy page up to the case checkpoint. Scored by pinball loss on the logit scale.

## Model

An Exophase earn rate is a ratio — `earners / players_tracked` — and the page only changes when Exophase refreshes its counts, so the forecast is built from the two counts, not from the ratio directly:

- **Players tracked P(t):** fitted with an ensemble of growth curves (saturating exponential, power law, log-linear) and extrapolated to the target. The point forecast comes from the saturating fit — a launch's tracked players grow fast and then level off — with the ensemble range feeding the uncertainty.
- **Earners E_i(t) per trophy:** a per-player hazard model. New earners arrive at rate `h_i * (P - E_i)`, proportional to the players who have not earned the trophy yet. `h_i` is estimated from the case history (new earners over person-time at risk, shrunk toward the tier mean), then stepped forward to the target under the extrapolated P(t).

The median rate is `100 * E_i(target) / P(target)`. Quantiles are logit-normal around it, with a spread combining the random-walk term, the P-ensemble spread, and the hazard sensitivity (re-running the forward step with `h_i` halved and doubled). The 100% trophy (every tracked player holds it) is forecast to stay at ~100%.

When the history never changes (the launch-night practice cases: every read shows the same page), there is nothing to fit, so the forecast applies a dilution prior: the page is stale, a refresh is due before the target, the refreshed page tracks several times more players, and the newly tracked players hold each trophy at a lower rate than the launch-night cohort — so each rate falls to roughly a third of its last value, graded by the trophy's current rate, with a wide spread.

## Validation

- Practice cases `p00`/`p01`: valid output (55 rows, monotone quantiles, 0–100). Against the practice answer reconstructed from the kit's own figures (Exophase's first refresh: 638 → 3,868 players tracked, 943 → 4,276 trophies earned; the house persistence baseline's typical miss of 0.68 points), the model scores a mean pinball loss of ~0.10–0.13 versus ~0.43 for the persistence baseline.
- Synthetic final-like cases (two regimes: players tracked saturating high with rates falling, and saturating early with rates rising; both built from the same hazard dynamics the model assumes): mean pinball loss ~0.14 versus ~0.34–0.37 for the persistence baseline.

## Run

Python 3.13, standard library only, no network:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## License

MIT — see [LICENSE](LICENSE).
