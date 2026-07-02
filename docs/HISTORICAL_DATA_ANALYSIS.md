# Historical Data Analysis

Written 2026-07-02. This document aggregates the findings of six parallel
analyses run against the project's historical data, hunting for exploitable
betting patterns, characterizing where the model's edge actually lives, and
stress-testing several parlay-construction ideas. For the underlying data
inventory (sources, row counts, date ranges) see
[CURRENT_HISTORICAL_DATA.md](CURRENT_HISTORICAL_DATA.md); for the pipeline
scripts see [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md). This document is
about what the data *says*, not how it is built.

## Methodology and guardrails

Six independent analyses were run. Each was held to the same discipline,
because a two-season sample is small enough to manufacture convincing fake
patterns:

- Every hit-rate, ROI, and correlation number is reported with its sample size.
- All payout math uses real quoted odds, never a flat -110 assumption.
- Model-based analyses use only genuinely out-of-sample predictions logged live
  in `data/betting.db` (recorded before outcomes were known). The training
  parquets were never re-scored with the current model, which would be
  in-sample leakage.
- Every promising pattern was split chronologically (early vs late, and by
  season) and only trusted if it survived in both halves. Several headline
  patterns did not survive this and are reported as noise.

Two datasets were used depending on the question:

- `data/processed/classification_training_data.parquet` (4,755 goalie-games,
  2,260 with both goalies present, both seasons) for pure-market and same-game
  pairing work.
- `data/betting.db` (1,789 rows, 566 games, 2026-01-04 to 2026-04-13) for
  anything involving the model's predictions.

A note on the multibook parquet: it was deliberately *not* used for same-game
pairing work. It is 82% home goalies and only 32 of its 1,381 games contain both
goalies, an artifact of how away-goalie names are matched during its build. This
is a separate data-quality issue worth investigating on its own (a home-skewed
training set could bias the model) but it does not affect the analyses here.

## Executive summary

Ranked by how actionable and how well-supported each finding is:

1. **The model's entire real edge is on the UNDER side.** UNDER recommendations
   hit 65.5% for +28.5% ROI and hold up (in fact strengthen) over time; OVER
   recommendations are a persistent -25.0% loser across every month and both
   chronological halves. Suppressing or heavily downweighting OVER calls is the
   single highest-value change suggested by this whole exercise. (Agent B)

2. **A data-integrity bug corrupts market features on ~262 production rows.**
   The odds-averaging step added when the tracker data was bridged in produces
   mathematically impossible odds (and therefore impossible `market_vig` /
   `impl_prob` values) on tracker-era rows. It fabricated fake positive ROI in
   two of the analyses before being caught. Fix recommended below. (Found
   independently by Agents C and D, root-caused here.)

3. **Same-game goalie outcomes are mildly negatively correlated** (Pearson
   -0.146, split-outcome rate 54.5% vs ~50% under independence), stable across
   every cut. Real and structurally sensible, but too weak to be a standalone
   edge. (Agent A1)

4. **None of the parlay strategies beat straight betting.** Model-directed
   same-game parlays are indistinguishable from straight bets and flip sign over
   time (Agent A2). Hedging both split directions is a structural -4.2% loser
   (Agent D, confirmed on a re-run with the corrected data). The market is
   efficient across line values (Agent C, also confirmed on re-run).

5. **On Underdog specifically, avoid 2-leg tickets; prefer 3-leg.** Underdog's
   fixed 3x-for-2-legs multiplier needs ~58% per-leg accuracy to break even and
   the model's top picks only reach ~52%; 3-leg tickets are the first size where
   observed hit rate (~55%) matches breakeven. Moderate confidence, thin sample.
   (Agent E)

The most important synthesis, combining findings 1 and 5, is in the
[cross-cutting section](#cross-cutting-the-under-edge-changes-the-parlay-answer)
below: the parlay backtests were run on the model's picks *including* its
money-losing OVER calls. The UNDER-only picks are exactly the high-hit-rate legs
Underdog parlays need, so the parlay question deserves a dedicated UNDER-only
re-run before it is considered closed.

---

## 1. Data-integrity bug: invalid averaged odds (Agents C and D, root-caused here)

Two agents independently found the same corruption: 164 goalie-games in
`classification_training_data.parquet` (all dated 2026-01-22 onward) have
`odds_over_american` or `odds_under_american` values inside the impossible
(-100, +100) American-odds range, for example -0.5, -39.0, 0.5.

**Root cause.** `load_betting_lines_from_tracker()` in
[scripts/merge_betting_lines.py](../scripts/merge_betting_lines.py) averages
American odds directly across books:
`odds_over_american=('line_over', 'mean')`. American odds are non-linear and
cannot be averaged arithmetically. When a goalie is quoted at, say, +102 at one
book and -103 at another (two nearly identical ~50% probabilities), the mean is
-0.5, which is not a valid odds value at all. `betting.db` itself is clean; the
corruption is created by the averaging step in the pipeline.

**Impact is not cosmetic.** Those invalid odds feed `add_market_features.py`,
which computes `impl_prob_over`, `market_vig`, and `fair_prob_over` from them.
The result: 262 rows in the production `multibook_classification_training_data.parquet`
carry impossible **negative `market_vig`** (a book cannot have negative hold)
and extreme implied probabilities (down to 0.005, up to 1.0). These propagate
into the production dataset via inherited base-feature columns even though
multibook overlays clean per-book raw odds on top. Attribution is unambiguous:
262 of the 266 negative-vig rows are tracker-era (2026-01-22+); only 4 predate
the bug.

**It manufactured fake edges in this very analysis.** Agent D's first pass,
before excluding these rows, showed a spurious +18.7% ROI driven entirely by a
handful of garbage mega-payout rows (a -0.5 "odds" converts to a 201x decimal
multiplier). Agent C's raw blind-UNDER ROI was similarly inflated. Both agents
excluded the rows and reran, which is why their final numbers are trustworthy,
but this is a concrete demonstration that the bug will silently poison any
downstream EV or backtest math.

**Fixed 2026-07-02.** `load_betting_lines_from_tracker()` in
[scripts/merge_betting_lines.py](../scripts/merge_betting_lines.py) now converts
each book's American odds to implied probability, averages the probabilities,
then converts back to decimal/American odds for storage -- immune to the
sign-crossing that produced the invalid values. The classification and
multibook parquets were regenerated. Verified: 0 invalid-odds rows remain in
either parquet (was 164 / 262); `market_vig` across the whole multibook dataset
now sits in a sane -0.009 to +0.095 range (was -0.79 to +0.55). The fix actually
corrected 362 rows, not just the 164 that were visibly broken -- every
multi-book tracker row from 2026-01-22 onward had a somewhat-wrong averaged
price under the old method, and only the worst cases landed in the obviously
impossible zone. Rows before 2026-01-22 and single-book rows were untouched, as
expected (`clean_training_data.parquet`, which has no odds, is byte-identical).

Agents C and D were re-run on the corrected data (2026-07-02, Sonnet 5) to
confirm whether their conclusions -- both reached by *excluding* the corrupted
rows -- would change once those rows had valid odds instead. **Both held.**
See sections 5 and 6 below for the before/after numbers.

---

## 2. Where the model's edge concentrates (Agent B)

Data: 336 graded directional calls in `data/betting.db` (200 UNDER, 136 OVER),
joined to `clean_training_data.parquet` for rest features. Multi-book rows never
disagreed on direction (0 conflicts), so per-book rows are legitimate ROI units.

**The headline is a clean OVER/UNDER split:**

| Direction | n | Hit rate | ROI/bet |
|---|---|---|---|
| UNDER | 200 | 65.5% | +28.5% |
| OVER | 136 | 39.0% | -25.0% |

The UNDER edge holds up cleanly under a chronological split (early 63.0% / +27.3%,
late 68.0% / +29.6% -- it strengthens). The OVER loss is equally persistent: it
loses money in every calendar month from January through April, in both
chronological halves, and is spread across 45 distinct goalies -- not one bad
actor or one bad week.

**The single strongest, most trustworthy cell in the entire dataset:**

| Segment | n | Hit rate | ROI/bet | Chronological check |
|---|---|---|---|---|
| UNDER at 65-70% confidence | 72 | 80.6% | +54.6% | early 79.4% / late 81.6% -- nearly identical |

This cell spans 29 different goalies (max 6 from any one), so it is not a
hot-hand artifact.

**Nearly every other segment is a mirage once you control for direction.**
BetOnline looked like a bad book (37% hit) but its rows are disproportionately
OVER calls (33 OVER vs 13 UNDER). The 25-28 betting-line bucket looked strong but
is 89 UNDER vs 14 OVER. The non-monotonic confidence pattern (65-70% good, 75%+
bad) is the same story: the good buckets are UNDER-heavy, the bad ones OVER-heavy.
Home/away, rest days, and back-to-back showed only small, sample-thin effects
(several cells n<15) not distinguishable from noise once direction is held fixed.

**Bottom line:** the model is not broadly good; its UNDER calls are good and its
OVER calls actively lose money. A rule that only bets UNDER (or heavily
downweights OVER) would have turned this 3.5-month sample from mixed to solidly
profitable. This is the most actionable finding in the analysis.

---

## 3. Same-game goalie correlation (Agent A1)

Data: 2,260 games in `classification_training_data.parquet` with both goalies
carrying a line. Outcomes only, no model predictions.

The two goalies' line margins (actual saves minus line) are **negatively
correlated**: Pearson -0.146 (p ~ 3e-12), Spearman -0.143. Small but highly
significant. The 2x2 contingency:

| | Away OVER | Away UNDER |
|---|---|---|
| **Home OVER** | 20.84% | 28.50% |
| **Home UNDER** | 25.97% | 24.69% |

That is 54.47% split outcomes (exactly one goalie over) vs 45.53% same-direction,
against a ~50/50 expectation under independence -- a real ~4.5-point excess of
splits (chi-square p ~ 2e-5).

The mechanism is intuitive: two goalies split a roughly fixed pool of game shot
volume, so a shot-heavy night for one net tends to mean a lighter night at the
other. The effect is remarkably stable -- the correlation sits between -0.13 and
-0.16 in *every* cut tested (both chronological halves, both seasons, favorite
vs underdog goalies, high vs low combined total). No cut flips the sign.

**Implication for parlays:** because a 2-leg parlay needs both legs to hit, a
negative correlation structurally *disfavors* same-direction parlays (both-over
or both-under occur less than chance) and mildly *favors* split parlays (one
over, one under). But at ~4-5 points off independence this is a base-rate tilt to
layer onto real per-leg edge, not a standalone strategy. Agent D quantifies why
it is nowhere near enough to overcome vig on its own.

---

## 4. Model-directed same-game parlays (Agent A2)

Data: 416 games in `data/betting.db` with both starting goalies gradeable, using
the model's live directional pick for each leg (deduped to one line per goalie,
Underdog-preferred).

| Strategy | n | Win rate | ROI |
|---|---|---|---|
| 2-leg parlay of the model's two picks | 416 | 29.1% | +4.59% |
| Betting both legs straight | 832 legs | 53.1% | +0.73% |

Both ROIs are statistically indistinguishable from zero (bootstrap 95% CIs
straddle zero), and both **flip from positive to negative in the second
chronological half** (parlay +13.8% early / -4.6% late). The parlay's higher
nominal ROI is pure variance from multiplying two uncorrelated legs' odds --
the leg-outcome correlation here is 0.035, essentially zero, so no synergy is
being captured. The subsets that would matter most are unusable: both-legs-
recommended is n=19 (far too small), both-legs-positive-EV is n=229 and slightly
negative (-2.3%).

**Bottom line:** no evidence that model-directed same-game parlays beat straight
bets. Treat as inconclusive noise, not a discovered edge. (See the cross-cutting
section -- this used the model's OVER calls too, which we now know lose money.)

---

## 5. Hedge both directions (Agent D)

The idea: buy *both* split-direction 2-leg parlays every game (home-over/away-under
AND home-under/away-over), betting that the game lands split often enough to
profit. Data: originally 2,111 clean paired games (after excluding the 164
corrupted-odds games from section 1); re-run 2026-07-02 on the corrected data
with those games included, growing the sample to **2,245** paired games
(+134, all from the previously-corrupted 2026-01-22+ window).

**Re-run result: conclusion held, got slightly worse.**

| Metric | Original (excluded) | Re-run (corrected, included) |
|---|---|---|
| Sample | 2,111 games | 2,245 games |
| Split frequency vs independence | 54.71% vs 50.00% (+4.71pp) | 54.43% vs 49.95% (+4.49pp) |
| Buy-both-tickets ROI | -3.89% | **-4.25%** |
| Breakeven (split x payout / 2.0) | 96.1% | 95.75% |
| Ticket1-only "edge" | +1.48% (flagged as noise) | **-0.06%** (resolved to flat) |

- The split-frequency negative correlation is still real and significant
  (p=0.000025) -- confirmed on the larger sample, not weakened.
- The 134 recovered games (previously excluded, now valid) ran **-8.80% ROI**
  on their own -- a losing stretch, not a hidden winner, which is why the
  corrected aggregate is marginally worse than the exclusion-based estimate,
  not better.
- Ticket1-only, the one variant that looked marginally positive before
  (+1.48%), resolved to essentially flat (-0.06%) once the corrupted-odds
  games were replaced with correct data -- confirming it was noise, exactly as
  flagged originally.
- A 5,000-resample bootstrap on the corrected data confirms the both-tickets
  ROI is robustly negative: 95% CI -7.83% to -0.49%, entirely below zero.
- Quarter-by-quarter shape is unchanged: 3 of 4 quarters negative (-4% to
  -9.5%), one positive (+5.41%, was +6.89%).

**Bottom line:** a structural loser, roughly **-4.2% ROI** (was -3.9%),
exactly as the vig math predicts, and if anything confirmed by more data
rather than softened. Buying both sides of a vigged market cannot escape the
hold; the real but modest negative correlation is not enough to bridge the
gap. Full re-run detail: `agent_d_rerun_findings.md` (scratchpad).

---

## 6. Trends by betting-line value (Agent C)

Data: all 4,755 goalie-games in `classification_training_data.parquet`, bucketed
by line value. Pure market, no model. Originally, ROI math excluded 164 rows
(all 2025-2026, dated 2026-01-22+) with invalid odds; re-run 2026-07-02 on the
corrected data included all 4,755 rows with zero odds-based exclusions.

At face value there is a striking pattern: as the line rises, the over-rate falls
well below the ~50% the market prices in (the market applies roughly the same
implied over-probability, ~0.53 including vig, at every line level -- it does not
shade its number for extreme lines). This produces an apparent blind-UNDER edge
climbing to +32% ROI in the >=30 bucket, statistically significant (p<0.02) in
the 24-25.5, 26-27.5, and 28-29.5 buckets in the pooled sample.

**It does not survive the chronological check, and including the corrected
data does not change that.** The entire signal remains concentrated in season
2024-2025:

| Bucket (26-29.5 combined) | Original (excluded) | Re-run (corrected, included) |
|---|---|---|
| Season 2024-2025 | -7.8% (p<0.0001) | -7.8% (p<0.0001) -- unchanged |
| Season 2025-2026 | +4.9% (p=0.113) | **+5.4% (p=0.088)** -- still reversed, still insignificant |

The sign still *reverses* in the current season, and the corrected data if
anything nudges the reversal slightly stronger (p tightened from 0.113 to
0.088, still short of the conventional 0.05 threshold). A bettor acting on the
pooled pattern today would still be betting into an already-reversed trend.
This is a textbook single-season artifact, and it survived a second look with
clean data. An independent chronological (non-season) split confirms the same
story: early half strongly UNDER-significant (diff -7.7%, p=3.4e-6), late half
reversed and insignificant (diff +4.3%, p=0.155). The extreme buckets the user
specifically asked about (<20 at n=15, >=30 at n=17) are too small to conclude
anything and inherit the same 2024-2025-only problem.

**Bottom line:** the market is efficient across the line-value spectrum. Do not
build a standalone "bet UNDER on high lines" rule. If a line-value signal is
wanted, feed it to the trained model as one feature, not as an exploit. This
was confirmed a second time on odds-corrected data, so it can be considered
closed rather than provisional. Full re-run detail: `agent_c_rerun_findings.md`
(scratchpad).

---

## 7. Underdog payout structure and nightly parlay construction (Agent E)

Underdog Fantasy is the user's real venue, and it only allows player-prop
parlays -- so the practical question is how to combine a night's picks into
tickets.

**Part 1 -- the payout-structure question, resolved.** Underdog's stored
`line_over`/`line_under` are genuine per-leg American odds pulled from Underdog's
own API (`UnderdogFetcher.get_goalie_saves()` reads `american_price`), confirmed
by 87 distinct near-continuous values from -139 to +113. They are **not**
Underdog's parlay multiplier table. Underdog actually pays fixed multipliers by
ticket size (Power Play: 2-leg 3x, 3-leg 6x, 4-leg 10x, 5-leg 20x; Flex pays
partial on N-1/N-2 correct), independent of the displayed leg odds. The
implication is important: **any parlay simulation that compounds the stored
American odds models a sportsbook parlay, not a real Underdog ticket, and
overstates realized profit** -- Underdog's multiplier tables bake in more hold.
Both payout models were simulated; only the Underdog-multiplier numbers reflect
real dollars.

(For contrast, PrizePicks rows all carry a single hardcoded -120 -- a researcher
placeholder, not a real price. Underdog's are real per-leg prices but still not
the payout mechanism.)

**Part 2 -- best construction rule.** Data: 213 deduped Underdog picks across 64
game-nights. Under the real Underdog multiplier payout:

| Ticket size | Underdog multiplier | Breakeven per-leg win% | Observed per-leg hit% |
|---|---|---|---|
| 2-leg | 3x | 57.7% | 52.1% |
| 3-leg | 6x | 55.0% | 55.2% |
| 4-leg | 10x | 56.2% | 60.6% |
| 5-leg | 20x | 54.9% | 65.0% |

2-leg Power Plays lose money (-18.8% ROI) because the top-2-EV legs only hit ~52%,
short of the 58% breakeven. 3-leg tickets are the first size where hit rate meets
breakeven, and "bucket every night's positive-EV legs into 3-leg tickets"
(all-pos-EV-multi-3leg) was the most consistently profitable rule -- positive
under both power-play (+17.4%) and flex (+37.0%) payouts, and positive in both
chronological halves, unlike almost every other strategy which flipped sign.
Jamming all of a night's legs into one giant parlay was the one clearly bad rule
(negative in both halves). Larger 4-5 leg tickets show higher nominal ROI but on
16-26 tickets that is variance, not proven edge (top-5's +150% is carried by 2
winning tickets out of 16).

**Bottom line:** with this model's picks, avoid 2-leg Underdog parlays and prefer
3-leg tickets, ideally Flex for the variance reduction. Confidence is
low-to-moderate: no leg-level hit-rate-vs-breakeven gap reaches significance
(all p>0.05) on this ~64-night sample, and the exact multipliers should be
confirmed live in-app.

---

## Cross-cutting: the UNDER edge changes the parlay answer

The parlay analyses (A2, D, E) were all run on the model's picks *as-is*, which
means they included the model's OVER calls -- and Agent B established that those
OVER calls hit only 39% and lose 25% per bet. Every parlay is only as good as its
worst leg, and half the candidate legs were coin-flips-that-lose.

This reframes the parlay question rather than closing it. Agent E found that
Underdog 3-leg tickets need ~55% per-leg accuracy and the blended top-EV picks
only reach ~52-55%. But Agent B found the UNDER-at-65-70%-confidence legs hit
**80.6%**. A parlay built only from high-confidence UNDER legs is composed of
exactly the legs that clear Underdog's breakeven with room to spare. The existing
backtests cannot see this because they average the good UNDER legs together with
the bad OVER legs.

**This is the single most promising unexplored thread from this exercise:**
re-run Agent E's nightly-construction simulation restricted to UNDER
recommendations (and ideally to the 65-70%+ confidence band), under the real
Underdog multiplier payout. The sample will be thin -- there may be only a
handful of nights with 2-3 qualifying UNDER legs -- but it targets the one edge
the data actually supports.

## Recommendations

1. **Suppress or heavily downweight OVER recommendations.** This is the clearest,
   best-supported, most valuable change. The model's OVER calls are a persistent
   money-loser across every slice tested.
2. ~~Fix the odds-averaging bug~~ -- **done 2026-07-02**, see section 1 above.
3. **Re-run the Underdog parlay simulation on UNDER-only, high-confidence legs**
   (the cross-cutting thread above) before concluding anything about parlays.
4. **Do not pursue** hedge-both-directions (structural -4% loser), line-value
   blind betting (single-season artifact), or same-direction same-game parlays
   (disfavored by the negative correlation and not supported by the backtest).
5. **Separately investigate the multibook home-goalie skew** (82% home, 32 paired
   games) noted in the methodology section -- it could be biasing the production
   model and is unrelated to the analyses here.

## Caveats

Everything model-based rests on 336 directional calls over 3.5 months; the
Underdog construction work rests on 64 nights. These samples are large enough to
establish the UNDER-vs-OVER split with real confidence (it survives every
chronological cut) but too thin to pin down finer segment effects or exact parlay
ROIs. The pure-market findings (A1, C, D) use the full two-season sample and are
correspondingly more robust. As more seasons accumulate, the UNDER/OVER finding
in particular is worth re-confirming, and the whole parlay question re-running on
a UNDER-only basis.
