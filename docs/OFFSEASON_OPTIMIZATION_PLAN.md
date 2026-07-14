# Offseason Optimization Plan (2026)

Written 2026-07-07. This is the output of an offseason deep dive across the whole
system -- the live betting record in `data/betting.db`, the production model's
internals and calibration, the training data itself, and the Underdog parlay
question left open in [HISTORICAL_DATA_ANALYSIS.md](HISTORICAL_DATA_ANALYSIS.md).
Every number below was computed fresh from the repo's data during this analysis,
not carried over from earlier docs. Where a finding is statistically thin, it says
so.

The one-paragraph version after the full offseason audit: **the current model
stack has no demonstrated tradable edge.** The old +23.31% backtest was
contaminated by test-set selection and corrupted multibook labels; the clean
retrain failed its single test touch; calibration compressed the model's raw
confidence and found no pre-registered bettable pocket; and venue analysis did
not show that Underdog was simply hanging softer saves totals. The live Feb-Apr
2026 UNDER run remains genuinely odd and worth respecting, but it is not enough
to scale without CLV or a repeatable mechanism. The offseason priority is now
to turn the system into an edge-detection platform: ticket-level tracking, CLV,
market-aware features, better hockey-context features, and a distributional
saves model evaluated through the honest harness. As of 2026-07-09, the
platform pieces (tickets, CLV, snapshots, shadow-run logging) are implemented
and the live DB migration has been applied. The market-anchoring and
distributional-model experiments (roadmap items 7 and 9) both came back clean
negatives on a bettable edge, though each produced a real methodological gain.
The first current-data game-context slice (roadmap item 8) improved
distributional prediction quality but still lost under the pre-registered
probability-edge betting policy. A follow-up push-aware true-EV policy audit
did not rescue the result: validation selection still preferred the old
probability-edge rule, and true-EV policies were mostly too broad and negative.
The next offline model question is timing-safe game-market ingestion, not
another generic retrain or another policy threshold sweep.

## Table of contents

1. [Honest state of the system](#1-honest-state-of-the-system)
2. [The critical new finding: corrupted training labels](#2-the-critical-new-finding-corrupted-training-labels)
3. [Model training improvements, ranked](#3-model-training-improvements-ranked)
4. [Betting strategy for next season](#4-betting-strategy-for-next-season)
5. [Prioritized offseason roadmap](#5-prioritized-offseason-roadmap)
6. [Codex-authored live implementation log](#6-codex-authored-live-implementation-log)
7. [Appendix: what was checked and found sound](#7-appendix-what-was-checked-and-found-sound)

---

## 1. Honest state of the system

### 1.1 The live record (what actually happened with real money)

From `data/betting.db`, bets with `bet_amount > 0`:

| Slice | n bets | Staked | P/L | ROI |
|---|---|---|---|---|
| All recorded bets | 301 | $397 | +$39.31 | +9.9% |
| UNDER selections | 180 | $246 | +$54.18 | +22.0% |
| OVER selections | 121 | $151 | -$14.88 | -9.9% |
| Followed model rec | 236 | $319 | +$44.28 | +13.9% |
| Went against model | 65 | $78 | -$4.97 | -6.4% |

Two things jump out. First, the familiar UNDER/OVER split shows up in the real
money, not just the backtests. Second, **the database cannot represent the real
Underdog economics**: the actual bankroll went from ~$100 to ~$1,800, but the
tracker shows only +$39, because it records parlay legs as individual straight
bets graded at American odds. Underdog pays fixed ticket multipliers, not
compounded leg odds. The tracker is currently blind to the thing that made most
of the money. Fixing this (a `tickets` table, section 4.5) matters for
everything downstream -- you cannot optimize what you cannot measure.

### 1.2 The current model's pure out-of-sample record

The production model deployed 2026-02-01, so `betting.db` rows from Feb 1
onward are a clean, fully out-of-sample test of exactly the model that will run
next season (January rows came from the older Config #4398-era model). Flat-bet
backtest of its recommendations at the stored odds:

| Direction | n | Hit rate | ROI/bet |
|---|---|---|---|
| UNDER | 168 | 69.0% | +32.6% |
| OVER | 81 | 37.0% | -29.7% |

UNDER by confidence bucket (Feb 1+):

| Bucket | n | Hit | ROI |
|---|---|---|---|
| 60-65% | 37 | 56.8% | +12.2% |
| 65-70% | 72 | 80.6% | +54.6% |
| 70-75% | 35 | 68.6% | +30.1% |
| 75%+ | 24 | 54.2% | +2.2% |

The 65-70% cell is the same one Agent B found; it holds in the pure
current-model window and across chronological halves. The falloff at 75%+ is
not noise to shrug at -- it is the calibration problem below.

### 1.3 The model is systematically miscalibrated at the extremes

Binning all 1,755 graded live predictions by the model's own P(over) against
what actually happened:

| Model P(over) | n | Actual over rate |
|---|---|---|
| 0.00-0.30 | 59 | 37.3% |
| 0.30-0.35 | 90 | 26.7% |
| 0.35-0.45 | 294 | 52.6% |
| 0.45-0.55 | 1,033 | ~49% |
| 0.55-0.60 | 144 | 47.9% |
| 0.60-0.65 | 57 | 36.8% |
| 0.65-0.70 | 36 | 41.7% |
| 0.70+ | 42 | 42.9% |

On average the model is centered (mean P(over) 0.489 vs actual 48.4%), but the
tails are broken in a specific, asymmetric way: **when the model says over 60%+,
games actually go over only ~37-43% of the time -- its confident OVER calls are
inverted, not merely weak.** Its confident UNDER calls (low P(over)) are
directionally right but overconfident. This is why the OVER side loses money and
why 75%+ UNDER confidence underperforms 65-70%: the raw probability stops
meaning anything past ~0.65 distance from the market.

A large part of the cause is the next section. A second contributor: an OVER
call at high confidence is, mechanically, the model disagreeing most with the
market on a goalie whose line looks low relative to recent form -- exactly the
spot where the market knows something the model does not (confirmed starter
changes, defensive injuries, schedule spots). The model has no inputs for any of
that; the market does.

One more slice that ties this to the training data: OVER recommendations on
**away** goalies hit 26.1% (n=46) vs 45.6% on home goalies (n=90). UNDERs show
no such split (67.3% away / 63.7% home). The model is at its absolute worst on
away-goalie OVER calls -- and as shown next, away goalies are almost absent from
(and actively corrupted in) its training data.

### 1.4 The headline +23.31% backtest ROI should be retired

`scripts/tune_hyperparameters.py` line 302 ranks all 168 (config x EV threshold)
evaluations by **combined val+test ROI** and ships the argmax. The test set
therefore participated in model selection, which makes the +23.31% figure an
optimistically biased maximum over 168 draws, not an unbiased estimate. This is
not a disaster -- the live record independently confirms a real UNDER edge --
but the honest performance claim for this system is now "**+32.6% ROI on 168
live out-of-sample UNDER recommendations, Feb-Apr 2026**", not the metadata
number. Section 3.3 covers how to select models honestly next time.

---

## 2. The critical new finding: corrupted training labels

### 2.1 What the bug is

`scripts/build_multibook_training_data.py` matches each odds record (one per
game/goalie/book) to a base feature row by trying
`{game_date}_{home_team}` **first**, then the away team, breaking on the first
hit. A goalie-last-name check is supposed to reject wrong-goalie matches -- but
the base parquet (`classification_training_data.parquet`) **has no
`goalie_name` column**, so the check silently never runs (the code path is
`if 'goalie_name' in base_row.index:` ... skip otherwise).

Consequence: an **away** goalie's odds record matches the **home** team's key
first, succeeds, and attaches the away goalie's betting line and odds to the
home goalie's rolling features -- and then recomputes the training label as
`over_hit = (home goalie's actual saves > away goalie's line)`. That is a
corrupted label, not a noisy one: 69.4% of games quote different lines for the
two goalies.

### 2.2 Measured impact

- The multibook parquet's 6,916 rows contain only **1,413 unique goalie-games**
  (the base parquet has 4,755 available), 79.7% of them home goalies; only
  32 of 1,381 games have both goalies present.
- Tracker-era rows (2026-01-04+) are 95.1% home. Tracing 1,996 of those rows
  back to `betting.db` game-by-game: **43.9% carry a betting line that belongs
  only to the opposing goalie.**
- The same matching loop built the pre-tracker rows, so the corruption spans
  the whole dataset -- including everything the production model was trained
  on Feb 1.
- Secondary hygiene issues in the same file: ~7.5% duplicate rows on
  `(game_id, goalie_id, book_key, betting_line)`, and 128 PrizePicks rows whose
  odds are a hardcoded -120 placeholder (they pollute any profitability
  backtest that touches them; `unknown`-book rows, n=148, are similar).

This single bug plausibly explains much of section 1.3: the model effectively
never saw clean away-goalie rows, and nearly half its recent training labels
answered the wrong question. That it still developed a real UNDER edge is
testament to the heavy regularization and to how much signal the clean 56% of
rows carried -- and it means a clean retrain has genuine upside, not just
hygiene value.

### 2.3 The fix (concrete)

1. In `scripts/merge_betting_lines.py` (or `create_clean_features.py`), carry
   `goalie_name` into the base parquet -- `betting.db` and the NHL API both
   map `goalie_id` to a name.
2. In `build_multibook_training_data.py`, make the last-name check mandatory:
   if the name is missing on either side, **reject** the match instead of
   accepting it. A goalie whose name matches neither team's base row should be
   logged and counted, not silently attached to whichever key existed.
3. Deduplicate on `(game_id, goalie_id, book_key, betting_line)` before
   writing, and drop (or flag) `prizepicks`/`manual`/`unknown` book rows.
4. Regenerate, then verify: `is_home` mean should land near 0.50, unique
   goalie-games should roughly double toward ~2,800+, and games with both
   goalies present should jump from 32 into the hundreds.

**Implemented and verified 2026-07-07.** Changes landed in
`scripts/create_clean_features.py` (carries `goalie_name` from the boxscore
JSONs through the whole pipeline) and `scripts/build_multibook_training_data.py`
(mandatory accent-insensitive last-name check, mandatory opponent check, book
filtering, final dedup). Two additional defects were found and fixed during
implementation:

- **A UTC/local date-shift bug** was silently discarding most legitimate
  matches: the odds cache's `commence_time` is UTC, so evening games land on
  the next calendar day relative to the boxscore-derived local `game_date`.
  A +/-1-day tolerance (mirroring `extract_historical_odds.py`) fixed it and
  was the single biggest lever -- matched records went from ~4,800 to ~13,200.
- The date tolerance opened a subtle **wrong-game path** (a goalie starting
  back-to-back nights could have one night's line attached to the other
  night's game, 124 records = 0.94%), closed by also requiring the base row's
  `opponent_team` to equal the odds event's other team.

Verified results (before -> after): `is_home` 0.797 -> 0.514; unique
goalie-games 1,413 -> 4,580; games with both goalies 32/1,381 -> 2,182/2,398;
duplicate rate 7.5% -> 0; placeholder-odds book rows 278 -> 0; tracker-era
misattribution 43.9% -> 0.0%; and an exact-tuple consistency check (every
output row must correspond to a real odds event with the same book, line,
goalie, and both teams within +/-1 day) passes for all 13,192 rows. Effective
training data is ~3.2x larger and clean. Pre-fix parquets are backed up in
`data/processed/backup_20260707/`.

Every training experiment in section 3 should use this regenerated data;
results from before the fix are not a valid baseline to compare against.

---

## 3. Model training improvements, ranked

Ordered by expected value per unit of effort.

### 3.1 Fix the data, then retrain on everything through April 2026

The production model has never seen a row after 2026-01-03 -- the parquets were
extended through 2026-04-13 on 2026-07-02, after it was trained. Combined with
the section 2 fix, a retrain gets: clean labels, ~2x effective sample, balanced
home/away, and three more months of the most recent season. This is the
mandatory first project; everything else layers on top.

**Done 2026-07-07, and the result is the most important honest number in this
repo: there is no demonstrable backtest edge on the clean data.** The full
retrain (same 42-config random search, same seed, same 60/20/20 chronological
split, now with the leak-free selection described in 3.3) was run on the
regenerated 13,192-row parquet. Fold boundaries: train 2024-10-04 to
2025-10-16 (7,915 rows), val 2025-10-16 to 2025-12-04 (2,638), test
2025-12-04 to 2026-04-13 (2,639). The winner (selected on validation only:
depth 3, lr 0.05, mcw 10, gamma 1.0, alpha 10, lambda 40, 1,200 trees, EV
threshold 0.15):

- Validation: +5.66% ROI (872 bets, 33.1% bet rate, 56.2% win rate),
  bootstrap 95% CI [-0.44%, +12.02%] -- **spans zero**.
- Single-touch test: **-7.09% ROI** (844 book-row bets, 32.0% bet rate,
  49.2% win rate), row bootstrap 95% CI [-13.44%, -0.79%]. OVER -8.72%
  (368 bets), UNDER -5.82% (476 bets): both sides lose. Important precision
  caveat: multibook rows on the same goalie-night are correlated. Deduping
  to one bet per goalie-game still loses (-6.8% ROI on 448 bets), but the
  wider bootstrap CI spans zero (about [-15.7%, +1.9%]). The conclusion is
  still bad, just not "statistically proven negative" in the independent-bet
  sense.
- The top 10 validation configs span +2.9% to +5.7% ROI -- all inside one
  CI half-width of each other, so the "winner" is a noise draw, not a
  discovery. The production config ranked 7th on validation (+3.8%).
- Every number above was independently re-derived (raw booster + reimplemented
  odds math + fresh bootstrap seed) and reproduced exactly. Model artifacts:
  `models/trained/tuned_v2_clean_20260707_212023/` (includes the full tuning
  log). **Not deployed** -- the production model dir and `predictor.py` are
  untouched.

What this means: the old +23.31% headline is now fully explained as
selection leakage on corrupted labels. The raw XGBoost-vs-market pipeline, on
clean labels with honest selection, loses at the books' own odds. The live
UNDER edge (section 1.2) was measured on Underdog lines during Feb-Apr 2026
and is NOT reproduced by this backtest, whose test window (Dec 2025-Apr 2026)
overlaps it. The venue analysis (done 2026-07-07, see 4.3) has since ruled
out soft Underdog lines as the explanation, and the calibration layer (done
2026-07-07, see 3.2) found no pre-registered bettable pocket. Model
probabilities remain grossly overconfident: at a 0.15 probability-edge
threshold the model still bets a third of all lines, claiming an average
+22.3% edge on test bets that resolve at a 49% win rate. The remaining
planning assumption is zero proven prior edge.

### 3.2 Add a probability calibration layer

The raw XGBoost score is not a trustworthy probability past ~0.65 (section
1.3). Standard fix: fit isotonic regression (or Platt scaling) on the
validation fold's predictions and pass every live score through it. Two
consequences worth spelling out:

- EV math becomes honest. Today's `EV = f(raw_prob, odds)` inherits the tail
  miscalibration, which is exactly why the 12% EV threshold produces
  recommendations whose extreme-confidence tier underperforms the middle tier.
- The betting policy simplifies: instead of the empirical "65-70% band good,
  75%+ mysteriously bad" rule, you bet calibrated probability above a
  threshold, and the bands stop needing folklore.

Validate it the same way everything else is validated: fit on one chronological
fold, check on a later one.

**Implemented 2026-07-07 -- verdict: no pre-registered +EV pocket is
demonstrated after calibration.** Protocol (pre-registered, leak-free):
validation fold split
chronologically (CAL-FIT rows 7915-9234, CAL-SELECT 9234-10553); isotonic and
Platt fit on CAL-FIT only; Platt chosen by CAL-SELECT Brier (raw 0.26469,
isotonic 0.24918 but with log-loss *worse* than raw -- it overfit 1,319
points -- Platt 0.24844); betting policies pre-registered on CAL-SELECT;
exactly one test-fold touch. Artifacts: `calibrator.pkl`,
`calibration_metadata.json`, run log in the `tuned_v2_clean_20260707_212023`
model dir; repeatable script `scripts/calibrate_model.py`. The repo's global
ignore rules normally hide `*.pkl` and `*.log`, so `.gitignore` now has narrow
exceptions for this calibrator and the two run logs. All headline numbers
were independently re-derived from scratch in a second pass (Platt refit
agrees with the saved artifact to within 0.0007).

The findings, in order of importance:

1. **The model carries almost no information.** Discrimination was only
   weakly positive inside validation: AUC 0.539 on CAL-FIT and 0.542 on
   CAL-SELECT. It did not meaningfully survive the final test fold (test AUC
   0.513; market vig-free implied probability AUC 0.522). Calibration
   compresses the raw 0.16-0.88 probability range down to 0.467-0.620
   (1st-99th percentile). The wild raw-score confidence that drove live
   betting bands was miscalibration, not knowledge. (Market implied-prob AUC
   is a limited comparator because books price saves props mostly by moving
   the *line*, not the odds, so implied probability is nearly constant by
   construction -- beating or losing to it on AUC is not the same as beating
   the market price.)
2. **Calibration makes the probabilities much less wrong, but not good enough
   to bet.** Test Brier improves 0.27579 -> 0.25040 because Platt scaling
   pulls the model back toward base rates. The calibrated reliability tables
   are better than raw, but still noisy; the top calibrated decile on test
   overpredicts badly. The probabilities became more honest, and that honesty
   reveals there is little to bet on.
3. **Pre-registered test results.** Policy A (both sides, calibrated edge >=
   0.05): -8.80% ROI on 238 bets across 149 goalie-nights (row bootstrap CI
   [-21.5%, +3.8%], cluster CI [-27.1%, +9.7%]); the bet mix collapsed to 98%
   OVER and lost -10.6% on that side. Policy B (UNDER-only, edge >= 0.02): 25
   bets on 18 nights in 4+ months, +2.80%, CI roughly +/-45pp -- uninformative.
   Post-hoc autopsy: the CAL-SELECT pocket that selected policy A (+24.1%) was
   133 bets on only 52 goalie-nights, all OVER, cluster CI already spanning
   zero -- fragile before the test fold ever saw it.
4. **The calibrated UNDER pool barely exists**: 8 qualifying UNDER bets in ~7
   weeks of CAL-SELECT, 25 in 4+ months of test. The live "65-70% confidence
   UNDER band" (section 4.2) was a raw-score artifact of the corrupted-data
   model. After honest calibration, that band is nearly empty.

Combined with the venue analysis (4.3), both offseason diagnostics now point
the same way: Underdog's lines are not soft, and the current feature/model
stack -- when scored on the reconstructed historical frame -- has no
demonstrated calibrated edge over the market.

**Revised 2026-07-08, see 3.11.** That reconstructed-frame result does not
settle the live run. Graded night-clustered rather than flat, and re-checked
against an independent archived line source, the live UNDER run is too
extreme to comfortably attribute to a favorable variance draw (roughly a
1-in-750 event even clustered by night) and it survived re-grading against
archived market lines intact. A favorable variance draw is no longer the
leading candidate; the leading hypothesis is now live-pipeline selection
behavior that the historical reconstruction does not capture (3.11) -- itself
an unproven mechanism, not a demonstrated edge. Planning assumption for next
season is still zero *proven* prior edge, but the honest next step is
measurement (CLV, shadow run), not dismissal. Any real edge now has to come
from new information the market underweights (roadmap items 7-9) or from
confirming the live-pipeline divergence, and in-season CLV tracking plus a
shadow run of the live system (item 6, 4.6) are the arbiters of whether live
betting resumes at meaningful stakes.

### 3.3 Select models honestly (this is why 3.1's results can be trusted)

Change `tune_hyperparameters.py` to rank candidates by **validation ROI only**,
touch the test fold exactly once for the final chosen config, and report both.
Better still, once 3+ seasons of labeled data exist, use walk-forward
validation (train on season 1, validate on 2; train on 1-2, validate on 3;
average) -- `CURRENT_HISTORICAL_DATA.md` section 6 already makes this case.
Also worth adding to the harness: a bootstrap CI on the backtest ROI, since a
~1,000-row test fold carries roughly +/-2pp of standard error on hit rate, and
single-number ROI comparisons inside that noise band are coin flips.

**Implemented 2026-07-07.** The search loop in `tune_hyperparameters.py` now
never touches the test fold (the old version leaked it two ways: candidates
were filtered by *test* bet rate and ranked by combined val+test ROI). New
flow: filter to 15-35% validation bet rate, rank by validation ROI, then one
single test-fold evaluation of the winner -- verified from the run log: 168
validation evaluations, exactly 1 test evaluation. Added `bootstrap_roi_ci()`
(10,000 resamples, percentile method), OVER/UNDER side breakdowns, fold-date
printing, and hard asserts that `feature_cols` is exactly the 114 approved
features with no identifier columns. `ClassifierTrainer.evaluate_profitability`
now also returns per-bet `bet_results` (backward compatible). Walk-forward
remains future work pending a third season of data. Results of the first
honest run are in 3.1.

### 3.4 Give the model the market's own information

The excluded "market-derived features" (`impl_prob_over`, `market_vig`,
`line_vs_recent_avg`, ...) were dropped in January under a "data leakage" label.
They are not leakage: implied probabilities and vig are known *before* the game,
at bet time -- `fetch_and_predict.py` literally has the odds in hand when it
builds features. (The genuinely unavailable-things -- actual saves, outcomes --
are correctly excluded and should stay excluded.)

Adding the no-vig implied probability as a feature is the single most standard
upgrade in sports modeling: it anchors the model to the market and forces its
output to represent *disagreement justified by other features* rather than an
independent opinion that must rediscover everything the market knows. The
predictable risk is that the model collapses toward parroting the market
(output = market prob, no bets ever clear the EV threshold); if that happens,
the honest conclusion is that the non-market features carry less standalone
edge than hoped, and the UNDER filter is doing the real work. Run the
experiment; either result is informative.

**Done 2026-07-08 -- roadmap item 7 executed.**

Design: four feature sets through an identical honest harness (24 validation
evaluations per set -- 6 pre-registered configs x 4 EV thresholds -- selection
on validation only via the 15-35% bet-rate band ranked by val ROI, exactly
one test touch per set, audited from the run log). A (control) = the standard
114 features; B (anchored) = 114 + market features recomputed per-row from
that row's own odds (impl_prob_over/under, market_vig, fair_prob_over -- the
12 contaminated precomputed columns stay dropped) + 8 book one-hots; C =
market features + book one-hots + betting_line only; D = no-model baseline,
probability = fair_prob_over directly. Folds were split by date (train <
2025-10-16, val through 2025-12-03, test from 2025-12-04; 7,844/2,676/2,672
rows), which also fixed the known fold-straddle problem. Every headline
number below was independently verified by from-scratch reproduction.

Test fold, single touch each:

- A: 843 bets, -5.97% ROI, cluster CI [-16.09%, +4.54%], AUC 0.5067 row /
  0.5019 per-night, Brier 0.27668. Sanity-consistent with the v2 clean
  retrain on the row-index split (same signs, similar magnitudes).
- B: 516 bets, -3.10% ROI, cluster CI [-16.04%, +10.19%] -- not
  distinguishable from zero -- AUC 0.5243/0.5214, Brier 0.26368.
- C: degenerate. No config landed in the bet-rate band even after the
  pre-registered fallback widening; its nominal validation winner was a
  single bet. Test: 10 bets, -9.28%, uninformative.
- D: AUC 0.5218/0.5170, Brier 0.24961, zero bets at every threshold. Nuance:
  not strictly "by construction" -- a handful of rows carry negative vig
  where the de-vigged probability exceeds the implied price by up to +0.47%,
  but that never reaches the lowest 5% threshold.

Anchoring diagnostics (validation only): corr(model B, fair_prob_over) =
0.325 -- the model does not collapse to the market; disagreement quantiles p5
-0.244 / p50 -0.026 / p95 +0.192. But B's acted-upon large disagreements hit
only 52.5% (val ROI -2.54%) -- the disagreements did not resolve in the
model's favor. fair_prob_over ranked first by gain; only 2 of the top-20 gain
features were market/book features.

Verdict, stated plainly: market anchoring improves discrimination and
calibration over the unanchored control (test AUC 0.524 vs 0.507, Brier
0.264 vs 0.277, ROI -3.10% vs -5.97%) but produces no edge distinguishable
from zero, and a model with only market information has no standalone signal.
A clean negative on the trading question; the anchoring direction remains
correct for any future model build. Artifacts:
`models/trained/experiment_market_anchor_20260708_184452/`.

### 3.5 Prototype a distributional model (the structural upgrade)

The current architecture answers "P(over this exact line)" per row. The
alternative: model the **saves distribution** directly -- either a two-stage
model (predict shots-against, then save rate) or a single count model (XGBoost
`count:poisson`, or negative binomial for overdispersion) -- and then price
P(saves > line) analytically for *any* line. Three concrete advantages:

1. **It trains on all 10,496 goalie-games across 4 seasons** -- no betting
   lines needed for training, only for the EV layer at the end. That sidesteps
   the entire odds-data-scarcity problem (and the section 2 bug class) at a
   stroke: the label is just the save count.
2. Line sensitivity is structural, not learned -- the multibook workaround
   exists only because the classifier had to be taught that different lines
   have different answers.
3. Calibration is more natural: a well-fit count distribution gives honest
   tail probabilities, and P(over 27.5) vs P(over 24.5) are automatically
   coherent with each other.

This is the biggest-effort item (a week-scale project done properly, with the
same chronological evaluation harness), so treat it as the offseason stretch
goal: build it, backtest it against the fixed-and-retrained classifier, and
keep whichever wins. Do not switch production to it on aesthetics.

**Done 2026-07-08 -- roadmap item 9 executed (prototype).**

Architecture: a shots-against XGBoost (`count:poisson`) feeds a negative-
binomial (NB2) distribution, with dispersion `alpha` fit by method-of-moments
on train residuals (alpha = 0.0193); a separate save-rate XGBoost
(`binary:logistic`, per-shot weighted) supplies q; the saves pmf is the
NB(shots) distribution thinned by Binomial(q), capped at 70 saves, with pmf
normalization asserted in code. P(over line) is priced directly from that pmf
for any posted line. Trained on all goalie-games in
`clean_training_data.parquet` (10,496 rows, 2022-10-07 onward; train/val/test
7,992/730/1,774 rows using the same date boundaries as the classifier
experiments above), i.e. no betting odds are required to train either
submodel. The betting evaluation joins the priced pmf to the multibook odds
frame by `(game_id, goalie_id)`, with 100% join coverage on both val and test.

Intrinsic quality (validation fold): shots-against MAE 5.4864 vs. a naive
rolling-5 baseline at 6.2089 (about 12% better); central 50%/80% predictive-
interval coverage lands at 48.5%/78.9% (close to nominal, i.e. well
calibrated); the PIT histogram is roughly flat with mild excess in the outer
bins.

Key mechanism finding: the save-rate submodel is nearly uninformative game-
to-game -- its correlation with actual per-game save rate is only 0.008, and
its predictions are confined to a 0.875-0.922 range against an actual
per-game save-rate standard deviation of 0.082. In practice the model is
doing "forecast shots against, then apply a near-league-average save rate" --
which lines up with 3.11's variance decomposition (saves vs. shots-against
R^2 = 0.959): almost all the achievable signal in a saves total is a
shots-against forecast, not a save-rate forecast.

Betting result (single test touch, at the only validation-in-band threshold,
0.05 -- thresholds 0.10/0.12/0.15 all fell below the 15-35% bet-rate band on
validation): 888 bets, 53.6% hit rate, +1.06% ROI, row bootstrap CI [-5.29%,
+7.39%], cluster CI [-8.83%, +10.91%] -- the cluster CI spans zero, not
distinguishable from breakeven. OVER: 470 bets, +2.22% ROI; UNDER: 418 bets,
-0.24% ROI. AUC 0.5159 row-level / 0.5202 per-night; Brier 0.25487.

Head-to-head, test fold, all single-touch numbers:

| Model | Test ROI | Bets | Brier |
|---|---|---|---|
| Distributional (this experiment) | +1.06% | 888 | 0.25487 |
| Classifier control A (3.4) | -5.97% | 843 | 0.27668 |
| Market-anchored B (3.4) | -3.10% | 516 | 0.26368 |
| Market baseline D (3.4) | no bets | 0 | 0.24961 |

The distributional prototype is the best-calibrated model the project has
produced and the only one of the four with a positive point-estimate test
ROI -- but the confidence interval spans zero, so this is not a discovered
edge. It also has a capability none of the classifiers have: it prices any
line coherently from one fitted distribution (3.9/3.11 already established
that one full save of line movement shifts win probability 5-6 points, which
matters directly for line shopping).

Honest next steps if this is pursued further: repeat the evaluation on a
later chronological slice before trusting the +1.06% point estimate; because
the leverage is entirely in the shots-against submodel, roadmap item 8's
game-context features (moneyline, totals, opponent pace/attempts) feed
exactly the submodel that carries the signal, strengthening that item's case;
the save-rate side needs genuinely new information (opponent shot quality)
rather than more tuning, since its near-zero game-to-game correlation is a
data problem, not a hyperparameter problem. Artifacts:
`models/trained/experiment_distributional_20260708_190338/`.

### 3.6 Feature additions worth testing (in rough priority order)

- **Game context from the odds market**: the game's total (O/U) and moneyline
  are public before puck drop and encode expected game script -- a heavy
  favorite protects leads and faces fewer late shots; big underdogs get
  shelled. The model currently has zero inputs about expected game state.
- **Opponent schedule/fatigue**: opponent back-to-back flag and days rest
  (tired teams generate fewer shots). The goalie's own rest is a feature;
  the opponent's is not.
- **Special teams rates**: opponent power-play shot volume and own team
  penalty-kill frequency -- PK time inflates shots faced. All derivable from
  boxscores already on disk.
- **Season-environment normalization**: league-wide mean saves fell every year
  for four straight seasons (27.4 in 2022-23, then 26.4, 24.7, 24.2). Features
  expressed relative to the current season's league mean travel across seasons
  better than raw counts, and next season will drift again.
- **Shot quality / xG**: the dead pipeline (`src/features/shot_quality_features.py`)
  already parses the 5,248 play-by-play files on disk for danger-zone metrics.
  Untested, not discredited -- worth one controlled experiment now that the
  evaluation harness is trustworthy.
- Zero-importance features to drop at the same time (they cost nothing but
  noise): `saves_rolling_3`, `goalie_is_back_to_back`, `line_vs_rolling_3`,
  and three short-handed std features that never got a split.

### 3.7 Know what the macro trend did to past findings

The league-wide save decline explains the one previously-confusing result:
in 2024-25 the market's mean line (25.3) lagged reality (24.6 mean saves), so
blind UNDER printed money; by 2025-26 the market adjusted (24.0 line vs 24.1
actual, 51% over rate) and that free edge vanished -- exactly Agent C's
"single-season artifact." The lesson for next season: any component of the
UNDER edge that was just "the market lags a declining-shots environment" decays
as books adjust. The live model's *selection* among unders (69% hit vs a 51.6%
blind-under base rate in the same window) remains an interesting anomaly, but
the clean retrain/calibration work did not demonstrate a repeatable mechanism.
Do not size next season as if +32.6% is a known prior edge.

### 3.8 Early-season handling

Rolling features run on `min_periods=1`: an October 12th prediction can ride on
2-3 games of data. Options, cheapest first: require a minimum of ~5 games
played before a goalie is bettable; seed early-season windows with the
goalie's previous-season averages; or add a `games_played_this_season` feature
and let the model learn its own caution. At minimum, expect October to be the
noisiest month and stake down accordingly.

### 3.9 Codex-authored: where an NHL saves edge can still exist

Authored by Codex, 2026-07-08. The clean retrain and calibration results kill
one thesis: "rolling goalie form plus line-relative features can reliably beat
posted saves lines by itself." They do **not** prove NHL goalie saves are
unbeatable. They say the edge, if it exists, is more likely to live in places
where the book's number is stale, mechanically constrained, or missing a piece
of hockey context the model currently does not see.

The useful mental model: a saves line is mostly a team shot-volume and game
environment price, with a goalie name attached. Books are not asking "how many
saves has this goalie made recently?" in isolation. They are pricing expected
shots against, goalie start probability, opponent pace, favorite/underdog game
script, injuries, rest, and market demand, then rounding to a half-save line
with juice. A model that mostly sees goalie rolling saves is trying to infer
the market's real inputs through shadows.

Places where a real edge is still plausible:

- **Starter/news timing**: goalie confirmations, surprise backups, beat-writer
  hints, morning-skate absences, illness, travel, and back-to-back deployment.
  These are not all equally visible to every book at every minute. A stale line
  after a starter or lineup update is a more believable edge than a generic
  65% model score.
- **Game environment mismatches**: moneyline, total, team total, opponent shot
  rate, score-state expectations, and pace matter more than goalie saves
  rolling averages. Big favorites can face fewer third-period shots when they
  control play; underdogs can face volume but also get pulled into lower-event
  defensive shells. The model currently has no explicit market view of expected
  game state.
- **Opponent and team style**: shot attempts, unblocked attempts, shots on goal
  share, rebound volume, high-danger creation, power-play shot volume, and
  penalty rates. Saves lines are shot-volume props first and save-percentage
  props second.
- **Book timing and copy behavior**: smaller books and fantasy apps may copy a
  market number, move late, or round differently. The current venue analysis
  says Underdog was not broadly hanging inflated totals, but it cannot rule out
  timing-specific staleness because the tracker has no fetch timestamp or
  closing line.
- **Line-shopping and thresholds**: the difference between 24.5 and 25.5 is
  enormous on a saves distribution. Even without a strong model, always taking
  the better side of a one-save spread can be the difference between a
  breakeven process and a losing one.
- **Parlay/ticket economics**: the edge may be in payout structure rather than
  the straight-bet line. That only matters if the legs have a proven edge after
  calibration or CLV. Without that, parlay multipliers amplify variance, not
  skill.

Places where hope is weaker:

- **Generic confidence bands from the current classifier**: calibration showed
  those bands mostly describe overconfidence.
- **More random hyperparameter search on the same 114 features**: the honest
  harness already answered that question well enough.
- **Blind UNDER as a permanent edge**: the market adjusted to the league-wide
  saves decline. Any future UNDER edge must be selected, timed, or priced, not
  assumed.

The practical conclusion: future work should be built around **market error
detection**, not standalone saves prediction. The question is not "what is the
goalie's true saves mean?" The betting question is "when is this particular
book's line or price wrong relative to the information available before puck
drop?"

### 3.10 Codex-authored: next model/data experiments worth doing

Authored by Codex, 2026-07-08. The next experiments should be small,
pre-registered, and designed to answer one business question: can we beat the
closing market or a clean chronological holdout? If an experiment cannot be
judged by CLV, chronological ROI, calibration, and cluster-aware uncertainty,
do not trust it.

Recommended experiment order:

1. **Market-anchored residual model**

   Add the no-vig market probability, book, line, line-open-to-current move,
   and game market context (moneyline, total, team total if available). Train
   the model to predict residual disagreement rather than rediscover the base
   over probability from scratch. This should answer: "does our hockey context
   improve on the market, or does the model collapse to the market?" Either
   answer is useful.

2. **Distributional saves model**

   Build a model for saves as a count distribution, preferably decomposed into
   shots against and save rate:

   - shots against: team/opponent pace, moneyline, total, rest, score-state
     proxies, home/away, special teams
   - save rate: goalie quality, opponent shot quality, team defensive quality
   - pricing layer: convert the distribution into P(over line) for any posted
     line

   This trains on all goalie-games, not only games with archived prop lines,
   and produces coherent probabilities across 22.5, 23.5, 24.5, etc. It should
   be compared head-to-head against the classifier on the same chronological
   windows.

3. **Game-context feature pack**

   Add features that map directly to how saves happen:

   - favorite/underdog status and moneyline-implied win probability
   - game total and team totals
   - opponent shots for, attempts for, unblocked attempts, and high-danger
     chances
   - goalie's team shots against, attempts against, penalty kill volume, and
     penalties taken
   - opponent rest, travel, and back-to-back status
   - confirmed starter status, backup/third-string flag, and prior-night usage

   Prioritize features available before puck drop and reproducible historically.
   Do not add a feature just because it sounds hockey-smart; make it survive
   the honest harness.

4. **Timing/CLV dataset**

   Add fetch timestamps and repeated snapshots. For each line, record:

   - first seen line/odds
   - bet-time line/odds
   - closing line/odds
   - book and venue
   - whether the goalie was confirmed at each snapshot

   This is the fastest path to knowing whether the process is real. P/L can
   lie for months; CLV starts talking in weeks.

5. **Policy model, not only probability model**

   Separate "predict over probability" from "decide whether to bet." The policy
   should know book, line availability, CLV history, edge threshold, max
   exposure, same-game correlation, and ticket construction rules. The current
   system mixes probability estimation and betting policy too tightly.

Evaluation rules for all future experiments:

- Split by full date or game before multibook row expansion so the same
  goalie-night cannot straddle folds.
- Report row-level and `(game_id, goalie_id)` cluster bootstrap CIs.
- Pre-register thresholds on validation; touch test once.
- Compare against market-only baselines and blind direction baselines.
- Track calibration and CLV, not only ROI.
- Treat small positive pockets as hypotheses until they repeat in a later
  chronological slice.

### 3.11 Claude-authored: the live run survived re-testing; the offseason backtests never tested the live system

Authored by Claude, 2026-07-08. Four additional diagnostics were run after
3.9/3.10, reading only `data/betting.db`, the historical multibook odds
archive parquet, and the saved model files (nothing was modified). They
target the question 3.2 and 4.3 left open: is the live Feb-Apr 2026 UNDER run
better explained by a favorable variance draw, or by something the
reconstructed-feature backtests cannot see?

**A. The anomaly, clustered by night, is more extreme than the flat-bet
numbers suggested.** Grading every goalie-night in `betting.db` once (dedup
on the latest row id) gives a blind-UNDER base rate of 52.0% (n=1,004). All
148 graded live UNDER recommendations, already deduplicated to one per
goalie-night and spanning 56 distinct calendar nights, hit 64.9% --
naive z=3.1, night-clustered bootstrap 95% CI [56.6%, 72.6%], with roughly
0.13% of night-level resamples landing at or below the 52.0% base rate. The
65%+ confidence tier (correctly defined as `1 - prob_over >= 0.65` --
`betting.db`'s `confidence_pct` column is not `P(side)`) is tighter still:
n=101 across 38 nights, 72.3% hit, CI [64.5%, 79.6%], with fewer than 1 in
20,000 resamples at or below base. Night-clustering is the conservative test
here (it treats correlated same-night legs as one draw), and the anomaly
survives it.

**E. The picks were re-graded against an independent line source, not just
re-summed from the tracker.** 137 of the 148 live UNDER recs were matched to
the historical multibook odds archive (accent-stripped last name + team +
date +/-1, because `betting.db` stores bare last names like `Bobrovsky` while
the archive stores `F. Lastname`-style short names). The live-fetched line
and the archived all-book median line agree almost exactly: mean gap -0.036,
identical on 92.7% of nights, live line higher (UNDER-favorable) on only
2.2% -- which rules out "the live line was soft relative to the wider
market" as an explanation (the same result holds against the archived
Underdog-only line). Re-grading the same picks against the archived median
line instead of the live-logged line still hits 66.4% (vs. 67.2%
live-graded on those same rows). Restricting further to the 123 of 137
matches where the archived actual-saves value exactly equals the live
actual-saves value (14 excluded as name/date matching noise -- see the
honesty caveats below), the re-graded hit rate is 65.0%, night-clustered 95%
CI [56.4%, 73.5%] -- the entire interval sits above the 52.0% base rate. Soft
or stale live lines and tracker grading artifacts are both ruled out for the
matched set: the selection itself performed against an independent line.

**F. The central new fact: the offseason backtests scored a different
model-input combination than the one that produced the live record.** For
the matched live UNDER recs, the correlation between the live-logged
`prob_over` (what `predictor.py` actually output at bet time) and the old
production model (`tuned_v1_20260201_155204`) re-scored on the reconstructed
clean historical frame is 0.408 (125 pairs), mean absolute difference 0.060
(mean live 0.304 vs. mean reconstructed 0.331). The pairs are range-restricted
to UNDER recs only, which mechanically attenuates the correlation number --
the 0.060 mean absolute difference is the cleaner statement of how far apart
the two probability streams run on the same goalie-nights.

The implication has to be stated carefully: every offseason backtest in this
document -- the clean v2 retrain (3.1), the calibration layer (3.2), the
venue analysis (4.3) -- was run on the *reconstructed* historical feature
frame. They test reconstructed models, not the (old production model + live
feature pipeline) combination that actually generated the live picks. That
combination has never been backtested, and cannot be from data currently on
disk. "The clean retrain failed its single test touch" (3.1) remains true
and stands as a finding about that retrain; it does not, by itself, falsify
the live system, because the live system was never the thing under test.

Two facts rule out the simplest alternative explanations for the gap: v1 and
v2 have identical feature name lists (114 features, verified by set
comparison), so this is not a feature-set difference, and market-anchor
features (implied probability, vig, line movement) were in *neither* model,
so this is not the market-anchoring experiment from 3.4/3.10 showing up
early. The v2 test fold (2025-12-04 to 2026-04-13) fully contains the live
betting window, so the live success and the backtest failure happened on the
same calendar dates -- whatever is driving the divergence is in the pipeline
or the model artifact, not the time period.

**G. The old production model, scored on the clean frame over its own true
out-of-sample window, is directionally consistent with the live UNDER edge --
and not statistically distinguishable from zero.** Scoring
`tuned_v1_20260201_155204` on the clean frame restricted to
`game_date >= 2026-02-02` (the day after it was trained -- genuinely never
seen), using archived lines/odds and its live EV rule (0.12 threshold,
probability-edge semantics): 864 rows scored, 219 bets across 174
goalie-nights, 53.0% hit, +1.36% ROI overall. OVER: n=49, 42.9% hit, -17.53%
ROI. UNDER: n=170 across 138 goalie-nights, 55.9% hit, +6.80% ROI,
night-clustered 95% CI [-10.53%, +23.84%]. One-row-per-goalie-night AUC on
this window is 0.5221 (n=666). Stated plainly: this is directionally
positive on UNDER, in the same window where the clean v2 classifier lost
-5.82% (3.1) -- but the confidence interval spans zero. It is interesting,
not evidence.

**B/C, briefly -- these confirm Codex's 3.9 framing empirically rather than
only conceptually.** On `clean_training_data.parquet` (10,496 goalie-games),
saves correlates with shots-against at 0.979 (R^2=0.959) and with save
percentage at only 0.559 (R^2=0.312); a saves line is a shot-volume prop
first, and the current 114 features are goalie-form-centric rather than
shot-volume-centric. Separately, on 4,580 unique goalie-games with lines, one
full save of line movement shifts win probability by 5.3-6.4 points (e.g.
P(saves>22.5)=61.3% vs. P(saves>23.5)=54.9%) -- more than the entire vig at
typical prices -- and 10.6% of the 3,771 goalie-nights quoted by 2+ books
show a cross-book line spread of at least 1.0 save. Both numbers say the
same thing 3.9 argued qualitatively: line-level and shot-volume information
dominate goalie-form information, and neither the v1 nor v2 model exploits
it directly.

Honesty caveats that apply to all of the above:

- **Project-level survivorship.** This record is being studied because it
  won. That alone should keep the prior on "real, durable edge" modest
  regardless of what any individual test shows.
- **The 65%+ tier was the live parlay rule, not a hypothesis chosen in
  advance of seeing results.** Its prominence in this and earlier sections is
  partly post-hoc; treat the tighter CI as suggestive, not as a
  pre-registered confirmation.
- **14 of 137 archive matches had disagreeing saves values** (name/date
  matching noise, not a systematic bias toward either outcome) -- the
  headline re-grading numbers were recomputed excluding them and the result
  held (65.0%, CI [56.4%, 73.5%]).
- **None of this proves a mechanism.** A correlation of 0.41 between live and
  reconstructed probabilities says the two pipelines disagree, not why, or
  which one (if either) is right. The arbiter is CLV plus a zero-stake
  shadow run of the exact live system next season -- not another backtest on
  reconstructed features.

**What this changes.** Sections 3.2 and 4.3 named a favorable variance draw
as the leading explanation for the live run. That is no longer the
best-supported reading. At the "all UNDER recs" level the anomaly is roughly
a 1-in-750 event even after the conservative night-clustered adjustment, and
the picks were independently re-graded against archived market lines with
the anomaly intact -- ruling out both a soft live line and a base-rate
artifact of the tracker. The leading hypothesis is now that some property of
the live (old model + live feature pipeline) selection behavior is not
captured by the reconstructed-feature backtests (F, correlation 0.41) -- this
is still an unproven mechanism, not a bankable edge, and G shows the plainest
direct test of it (old model, clean frame, true OOS window) lands with a CI
that spans zero. The decisive next-season tests are CLV capture and a
token-stake shadow run of the exact live system: same
`tuned_v1_20260201_155204` model file, same live feature pipeline (not a
reconstruction), every recommendation logged with fetch timestamps and
closing lines (roadmap items 6/4.6).

### 3.12 Codex-authored: first game-context distributional slice improved prediction quality, not betting edge

Authored by Codex, 2026-07-09. This section records the first implementation
slice of roadmap item 8. It deliberately used only data already present in the
repo: `clean_training_data.parquet` plus `data/raw/schedules/*.json`. No
moneyline, game total, team total, starter timestamp, injury, play-by-play, or
xG data was introduced, because those require separate timing-safe ingestion.

The new builder, `scripts/build_game_context_features.py`, creates
`data/processed/game_context_features.parquet` (generated artifact) plus
`game_context_features_metadata.json`. The artifact has 10,496 rows, one per
clean goalie-game, with 32 generated context features:

- team and opponent rest, back-to-back, games-in-last-4-days, and 3-in-4 flags
  from schedule history;
- prior-only season-to-date team shots against, opponent shots for, and goalie
  shots against;
- shifted EMA-5 versions of the same shot-volume contexts;
- relative/z versions of existing rolling shot-volume columns against
  season-to-date baselines.

Leakage guardrails held in the implementation checks: output keys are unique
on `(game_id, goalie_id, team_abbrev, opponent_team, game_date)`, schedule
coverage is 100% for team and opponent, current-game outcome columns and
betting-line columns are not emitted as features, and all season-to-date/EMA
features use prior rows only. Early-season nulls remain intentionally and are
handled by XGBoost as missing values.

The paired experiment, `scripts/experiment_game_context_distributional.py`,
reran the distributional architecture through the same honest date folds and
single-touch betting harness:

| Variant | Role | Test ROI | Bets | Row AUC | Goalie-night AUC | Brier | Cluster ROI CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control` | primary | +1.06% | 888 | 0.5159 | 0.5202 | 0.25487 | [-8.83%, +10.91%] |
| `context_shots` | primary | -1.68% | 758 | 0.5360 | 0.5356 | 0.25223 | [-11.65%, +8.55%] |
| `context_both` | secondary | -1.35% | 747 | 0.5382 | 0.5375 | 0.25204 | [-11.24%, +8.81%] |

The hockey read is encouraging but not tradable: the context features improved
shots-model validation MAE slightly (5.4864 -> 5.4821), reduced test Brier, and
lifted one-per-goalie-night AUC by roughly 1.5-1.7 points. The top context
features are exactly the right kind of signal -- opponent season-to-date shots
for, team season-to-date shots against, goalie season-to-date shots against,
opponent EMA shots for, and opponent 3-in-4. That says the model is learning
shot-volume context rather than random noise.

The betting read is still negative: both context variants selected the 5-point
EV threshold on validation, then lost on the single test touch, mostly because
the added context shifted the bet mix toward UNDERs that did not win enough at
available prices. The OVER subset was positive in both context variants
(`context_shots` +6.66%, `context_both` +9.38%), but those are small,
post-test side slices and must be treated as hypotheses only. This is not a
demonstrated edge.

Independent Codex audit after the run found no blocking leakage or protocol
issue: date folds are disjoint, context joins have 100% coverage, the feature
artifact excludes current-game outcomes, and the experiment keeps validation
selection separate from the single test touch. Two caveats carry forward. First,
the AUC/Brier improvement is a single chronological-split observation, not a
confidence-tested delta, so the precise wording is "improved on this split."
Second, this run itself used the repo's existing probability-edge semantics
(`model_prob - implied_prob`) and ignored push probability in selection even
though the distributional model computes `p_push`. That caveat was investigated
in section 3.13. It did not change the conclusion.

What this changes: item 8 is now partially implemented and no longer purely
speculative. Game-context features do improve the hockey model in the direction
we wanted, but the first current-data slice does not beat the market as a
betting policy. The next item-8 work should add genuinely missing market/game
environment data (moneyline, game total, team totals, and timing-safe line
movement); another small tweak to the same current-data context pack is
unlikely to turn -1.68% into a durable edge.

### 3.13 Codex-authored: push-aware true-EV policy audit did not rescue the distributional model

Authored by Codex, 2026-07-09. This was the next sub-agent wave after 3.12.
The question was deliberately policy-only: keep the same saved game-context
distributional models, same date folds, same validation-only selection, and
same single test fold, but replace the repo's historical "EV" convention
(`model_prob - implied_prob`) with true expected profit per $1 stake.

Correct policy math:

- `OVER EV = P(over) * profit_if_over_wins - P(under)`.
- `UNDER EV = P(under) * profit_if_under_wins - P(over)`.
- `P(push)` contributes 0 profit and should be counted as a zero-profit bet
  if an integer line ever pushes.
- For probability-edge guardrails on true-EV rules, compare conditional
  non-push model probability to no-vig market probability:
  `P(side) / (P(over) + P(under)) - fair_market_probability(side)`.

New files:

- `src/experiments/policies.py`
- `scripts/experiment_push_aware_true_ev_policy.py`

The policy grid compared 20 rules: four legacy probability-edge thresholds,
four pure true-EV thresholds, nine true-EV plus conditional no-vig edge
guardrails, and three line-shop true-EV variants. The script first replayed the
old probability-edge policy and required it to exactly match the 3.12 artifact
before interpreting the new policies. That replay passed for all three
variants.

Result: the true-EV layer did **not** find a better honest policy. On
validation, pure true-EV policies selected far more rows than the old rule
(roughly 42-70% of all book rows before line-shopping) and were mostly negative
ROI. The line-shop variants were also negative on validation. The validation
selector therefore chose the old 5-point probability-edge rule for every
variant, and the single test results exactly matched section 3.12:

| Variant | Selected policy | Test ROI | Bets | Push rate | Cluster ROI CI |
|---|---:|---:|---:|---:|---:|
| `control` | `old_prob_edge_0.05` | +1.06% | 888 | 0.0% | [-8.83%, +10.91%] |
| `context_shots` | `old_prob_edge_0.05` | -1.68% | 758 | 0.0% | [-11.65%, +8.55%] |
| `context_both` | `old_prob_edge_0.05` | -1.35% | 747 | 0.0% | [-11.24%, +8.81%] |

Important caveat: this audit reuses an already-inspected chronological test
fold and was motivated by the 3.12 result. A positive ROI here would have been
post-hoc policy sensitivity, not independent edge proof. Since the result was
negative, the useful conclusion is narrower but still important: the missing
edge is not hiding in the obvious push-aware true-EV conversion on top of the
current distributional probabilities. The next real path is new timing-safe
information, especially game-market/pace context and in-season CLV evidence,
not more threshold shopping.

### 3.14 Claude-authored/Codex-authored: pace/xG ingestion and distributional experiment design

Authored by Claude, 2026-07-09; Component 2 and Component 3 completion authored
by Codex, 2026-07-09. Status: Components 1-3 are implemented and verified;
Component 4 is not built. This section doubled as the pre-registration for the
experiment; the Component 3 interpretation rules below were written before any
pace/xG model result existed.

**Motivation and hypothesis.** The distributional shots-against submodel
currently forecasts shot volume from shots-on-goal rolling averages, which are
noisy. Shot attempts (Corsi/Fenwick), score-adjusted pace, and xG are better
predictors of future shot volume. The specific market hypothesis being tested:
saves prop lines are plausibly set from simpler inputs (goalie rolling saves
plus game total) than a full pace model, so pace/xG features may carry
information the prop market specifically does not price -- even though they
could never beat the game-total market itself. Prior probability is low given
six straight negative experiments; the bar below reflects that.

**Data sources (verified 2026-07-09 by live fetches, three research passes).**

- Primary, team level: MoneyPuck game-by-game file
  `https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv`
  (126 MB, 2008-present, one row per team per situation per game; situations
  `all/5on5/5on4/4on5/other`). Verified columns include
  `shotAttemptsFor/Against` (Corsi), `unblockedShotAttemptsFor/Against`
  (Fenwick), `xGoalsFor/Against`, `scoreAdjustedShotsAttemptsFor`,
  `scoreVenueAdjustedxGoalsFor`, and danger-tier xG splits. `gameId` is full
  NHL format (e.g. `2024021038`) and joins directly to the repo's `game_id`.
- Primary, goalie level: MoneyPuck per-season goalie game logs
  `https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/goalies/{YEAR}.zip`
  (per-situation shots faced, `xOnGoal`/`ongoal`, xG faced, rebounds, freezes;
  `gameId` also NHL format).
- Cross-check: NHL stats API
  `https://api.nhle.com/stats/rest/en/team/realtime?isAggregate=false&isGame=true&limit=-1&cayenneExp=gameTypeId=2 and seasonId={S}`
  -- one row per team-game with `shots`, `missedShots`, `shotAttemptsBlocked`,
  `totalShotAttempts`; verified that `totalShotAttempts` equals Corsi For
  exactly and reconciles with boxscore SOG. No auth; one call retrieves a full
  season. Note the field trap: `blockedShots` is the team's defensive blocks,
  `shotAttemptsBlocked` is its own blocked attempts.
- Rejected for now: Natural Stat Trick. All parameterized requests sit behind
  an active Cloudflare challenge; the sanctioned path needs a free account plus
  manually approved access key (~180 req/hr). User decision 2026-07-09: skip.
  Its unique value (score-and-venue-adjusted 5v5 rates, HD/MD/LD splits beyond
  MoneyPuck's) can be revisited if this experiment shows promise.
- ToS: MoneyPuck explicitly permits pulling the listed download files with
  attribution; add a credit line to README when built. Known joins gotcha:
  Coyotes are `ARI` through 2023-24 and `UTA` from 2024-25.
- The 126 MB CSV must never be committed (GitHub hard-blocks files over
  100 MB): cache under `data/raw/moneypuck/` with a `.gitignore` entry.

**Component 1 -- ingestion, `scripts/fetch_pace_data.py`.** Download
`all_teams.csv` plus per-season goalie zips into `data/raw/moneypuck/`
(gitignored). Cross-validate against the NHL realtime report joined on
`(gameId, team)`: require
`|MoneyPuck shotAttemptsFor(all) - NHL totalShotAttempts| <= 2` on at least
99% of joined rows, print the diff distribution, hard-fail otherwise. Emit
normalized `team_games.parquet` and `goalie_games.parquet` with a
`playoffGame` flag preserved.

*Amendment 2026-07-09, made during Component 1 implementation, before any
experiment ran:* (1) Season scope extended from 2023-2025 to **2021-2025**.
The original spec assumed the training frame started at 2024-25; in fact
`clean_training_data.parquet` starts at 2022-23, and a 2023+ fetch would have
left a third of the distributional train fold with all-null pace features.
2021-22 is fetched only to supply prior-season baselines for 2022-23 rows.
(2) The NHL realtime report's `totalShotAttempts`/`shotAttemptsBlocked`
fields turned out to be 100% null for seasonId=20212022 (populated from
2022-23 on), so for seasons where Corsi is unavailable from the NHL side the
cross-check falls back to Fenwick:
`|MoneyPuck unblockedShotAttemptsFor - NHL (shots + missedShots)| <= 2`, same
bars. Both amendments are data-scope/validation-mechanics changes; no feature
definitions, folds, variants, or interpretation rules changed.

**Component 2 -- feature builder, `scripts/build_pace_features.py`.** Same
contract as `build_game_context_features.py`: one row per clean goalie-game
keyed on `(game_id, goalie_id, team_abbrev, opponent_team, game_date)`;
output `data/processed/pace_features.parquet` plus a visible metadata JSON;
strictly prior-only (shifted) aggregates computed from regular-season games
only; early-season nulls left missing for XGBoost; assert unique keys and
report join coverage (expect ~100%). Hard requirement from the 3.11
reconstruction-infidelity lesson: the feature computation lives in an
importable module (`src/features/pace_features.py`) so a future live path
runs the identical code -- no separate live reimplementation, ever.

Pre-registered feature families (~35-45 features; exact list recorded in the
artifact metadata):

1. Opponent offense pace: rolling-5/10 and EMA-5 of opponent
   `shotAttemptsFor`, `unblockedShotAttemptsFor`, `xGoalsFor` (situations
   `all` and `5on5`), score-adjusted attempts; season-to-date means;
   prior-season per-game baseline.
2. Team shot suppression: the same transforms of team `shotAttemptsAgainst`,
   `unblockedShotAttemptsAgainst`, `xGoalsAgainst`.
3. Combined pace: sums/means of opponent-offense and team-defense EMAs (the
   expected attempt environment at the goalie's net).
4. Special-teams volume: opponent 5on4 attempts/xG per game, team 4on5
   attempts-against/xG-against per game (season-to-date and EMA-5); team 4on5
   ice time per game as a penalty-taking proxy.
5. Goalie workload quality (goalie file): rolling xG-per-shot-faced,
   high-danger share of shots faced, rebound rate. Used only by the
   save-rate-submodel variant.
6. League-relative z-scores of families 1-3 against the season-to-date league
   distribution (same pattern as the game-context builder).

*Codex-authored Component 2 completion, 2026-07-09:* implemented
`src/features/pace_features.py` and `scripts/build_pace_features.py`. The CLI
builds `data/processed/pace_features.parquet` plus
`data/processed/pace_features_metadata.json`; the feature computation lives in
the importable module so any future live path can reuse the exact same
definitions.

Final artifact:

- Rows: 10,496, exactly matching `clean_training_data.parquet`.
- Generated features: 45 across the six pre-registered families.
- Key: unique on `(game_id, goalie_id, team_abbrev, opponent_team, game_date)`.
- Join coverage: team side 100.00%, opponent side 100.00%, goalie side 99.99%.
  The single goalie miss is the known upstream MoneyPuck gap for Spencer
  Knight, `game_id=2025020964`, CHI @ WPG on 2026-03-03.
- Early-season null behavior is intentional and visible in metadata. October
  monthly mean null rates are about 12% because shifted same-season rolling/EMA
  features have no prior games yet; prior-season baseline columns are populated
  for the 2022-23 opening rows from the 2021-22 fetch.

Independent verification by Codex:

- Recomputed representative features directly from raw MoneyPuck rows for
  `game_id=2022020286`, goalie `8474889` (SEA vs LAK, 2022-11-19):
  opponent rolling-5 Corsi For matched exactly (58.2), team season-to-date
  Corsi Against matched exactly (53.0), and goalie rolling-10 xG-per-shot
  matched to floating precision.
- Mutated that current game's team and goalie MoneyPuck rows in temporary
  parquet copies by adding huge values to the current-game attempts/xG/shots
  columns; the row's generated features were unchanged. This is the leakage
  check that matters most for Component 2: game N does not feed game N.
- Compile and builder reruns passed with the bundled Python runtime.

This component is plumbing, not edge evidence. It only establishes that the
pace/xG feature artifact exists and appears timing-safe enough to feed the
pre-registered Component 3 experiment.

**Component 3 -- experiment, `scripts/experiment_pace_distributional.py`.**
Reuses `src/experiments/harness.py` and `src/experiments/distributional_saves.py`
unchanged; same date folds (train < 2025-10-16, val 2025-10-16 to 2025-12-03,
test >= 2025-12-04); validation-only selection over the same policy grid as
3.12; exactly one test touch per variant; goalie-night cluster bootstrap CIs.
Pre-registered variants:

- `control`: exact re-run of the 3.12 control. Must reproduce test Brier
  0.25487 / +1.06% ROI / 888 bets before any new variant is interpreted
  (harness integrity check).
- `pace_shots` (primary): families 1-4 and 6 added to the shots submodel only.
- `pace_context_shots` (primary): pace families plus the 3.12 game-context
  features in the shots submodel -- the best-available combined model.
- `pace_both` (secondary): `pace_context_shots` plus family 5 in the save-rate
  submodel.

Pre-registered interpretation rules:

- An edge claim requires the test cluster CI to exclude zero, and even then it
  is a hypothesis for next-season confirmation, not proof: this test fold has
  been inspected by 3.5, 3.12, and 3.13 and is worn as an independent arbiter.
- Brier/night-AUC improvements vs `control` are single-split observations.
- Named benchmark: the de-vigged market's Brier on this test fold is 0.24961
  (3.4 baseline D). No repo model has ever beaten it. If any variant does,
  that is reported prominently regardless of ROI, because it would be the
  first evidence the model knows something the market consensus does not.
- Negative-result disposition, decided now: if pace features improve
  prediction quality but produce no bettable edge (the pattern of every
  experiment so far), the out-modeling route is treated as close to
  conclusively dead -- the saves market prices at full-analytics sharpness --
  and remaining effort moves to the timing/price/CLV families per 3.13.

*Codex-authored Component 3 completion, 2026-07-09:* implemented and ran
`scripts/experiment_pace_distributional.py`. The final canonical artifact is
`models/trained/experiment_pace_distributional_20260709_100802/`; an earlier
same-result background artifact exists from the sub-agent wave and is not the
one cited here.

Integrity/verification:

- The control reproduction gate passed exactly: test Brier 0.25487, +1.06%
  ROI, 888 bets, selected threshold 0.05.
- The script used the pre-registered folds, validation-only submodel selection,
  validation-only EV-threshold selection, and one test touch per variant.
- Pace feature coverage was 100.00% by artifact key; source flags remained
  team 100.00%, opponent 100.00%, goalie 99.99% because of the known Spencer
  Knight MoneyPuck gap.
- Codex independently reloaded the saved models and regraded the test fold with
  separate summary code. Brier, ROI, bet counts, side counts, and the de-vigged
  market Brier reproduced.

Single-touch test results:

| Variant | Selected threshold | Test Brier | Test ROI | Bets | Night AUC | Cluster ROI CI |
|---|---:|---:|---:|---:|---:|---:|
| `control` | 0.05 | 0.25487 | +1.06% | 888 | 0.5202 | [-8.83%, +10.91%] |
| `pace_shots` | 0.05 | 0.24904 | +9.02% | 616 | 0.5408 | [-3.12%, +21.15%] |
| `pace_context_shots` | 0.10 | 0.25116 | +3.77% | 357 | 0.5458 | [-11.64%, +18.63%] |
| `pace_both` | 0.10 | 0.25136 | +0.89% | 352 | 0.5448 | [-14.49%, +16.18%] |

The important result is `pace_shots`: it is the first repo model to beat the
de-vigged market Brier benchmark on this shared test fold (0.24904 vs 0.24961,
a 0.00057 improvement) and it produced a positive selected-policy ROI. The
honest interpretation is still not "proven edge." The cluster CI crosses zero,
the Dec 2025-Apr 2026 test fold is worn from prior experiments, and the
positive policy result is mostly OVER-heavy (508 OVER bets, 108 UNDER bets).
This is the strongest offline hypothesis so far and justifies next-season
shadow/CLV confirmation; it does not justify scaling stakes by itself.

*Claude-authored independent verification, 2026-07-09 (after the Codex run):*
reloaded the saved `pace_shots` models from the canonical artifact and rebuilt
the test-fold probabilities with an independent pmf implementation
(scipy nbinom/binom convolution instead of the harness's gammaln matrix) and
independent de-vig, grading, and bootstrap code. All headline numbers
reproduced exactly: test Brier 0.24904, market Brier 0.24961, 616 bets,
+9.02% ROI, 508 OVER / 108 UNDER, cluster ROI CI matching to bootstrap noise.
Two statistics were then added that the original run did not compute, and they
change the headline's strength:

- **Paired Brier delta vs the market, test fold:** -0.00057 with a
  goalie-night cluster bootstrap 95% CI of **[-0.00485, +0.00395]**
  (n=2,672 rows, 1,307 clusters). 40.5% of bootstrap resamples had the model
  at or behind the market.
- **Same statistic on the validation fold:** the model was *behind* the market
  there: model Brier 0.25327 vs market 0.25076, paired delta +0.00251, cluster
  CI [-0.00422, +0.00904] (n=2,676 rows, 659 clusters).

So the accurate claim is **market parity, not market superiority**: across the
only two out-of-sample folds available, the point estimates go in opposite
directions and both CIs span zero. The genuinely new and robust finding is the
within-model improvement -- pace features improved the distributional model's
Brier on both folds (val 0.25508 -> 0.25327, test 0.25487 -> 0.24904 vs
control), making it the first repo model to reach market-level calibration.
The betting-policy ROI did not replicate across folds (+0.98% val,
+9.02% test at the same threshold), so the +9.02% should be treated as
consistent with both edge and noise. Confirmation has to come from in-season
CLV/shadow evidence, not from further work on this test fold.

**Component 4 -- live-season path (documented now, built only if justified).**
MoneyPuck updates nightly around 03:40 ET (verified from its update-timestamp
file). The morning fetch workflow would re-download `all_teams.csv` once per
day (within stated ToS), run the shared builder module, and hand features to
`fetch_and_predict.py`. Fallback if MoneyPuck is stale or down: the NHL
realtime report supplies attempt counts with xG features null for the day, or
the previous-day cache is reused with a staleness flag. Post-game latency
(whether last night's games are reliably present by the morning fetch) is an
opening-week verification task.

**Effort estimate.** Ingestion plus cross-check about half a day; builder plus
leakage checks half a day; experiment reusing the harness half a day; a
verification pass on top. Roughly 1.5-2 days total.

### 3.15 Claude-authored: The Odds API historical acquisition plan (cache-first, written 2026-07-09, EXECUTED same day)

Authored by Claude, 2026-07-09, after the user purchased Odds API credits.
`src/betting/odds_archive.py` and
`scripts/fetch_historical_odds_snapshots.py` implement this contract (built
by a Sonnet sub-agent, line-by-line reviewed, then hardened with request
pacing, 401/403 fast-abort, and a consecutive-failure abort). The API key
lives in `.env` as `API_KEY` (`.env` is gitignored; the key must never be
printed to logs or committed). This section is the contract for every
historical Odds API fetch: what we buy, why, in what order, and the caching
rules that guarantee we never pay for the same data twice.

**EXECUTION ACTUALS (2026-07-09, user authorized probe + blanket Phases 1-3).**
All phases run to convergence (dry-run shows 0 planned / 0 estimated except
the two dead events below). Full per-run log in
`data/raw/betting_lines/cache/fetch_log.jsonl`. Starting balance 100,000.

| Phase | Runs | Fetched | Est. credits | Notes |
|---|---|---|---|---|
| Probe | 2 | 12 | 102 | 2023-24 coverage confirmed; 10 cr/odds call and 1 cr/events call confirmed via `x-requests-remaining` deltas |
| phase1-bettime | 2 | 881 | 7,973 | 1 permanent 404 (see dead events) |
| phase1-closing | 1 | 538 | 5,380 | same dead event's closing call |
| phase2 | 2 | 2,808 | 26,370 | 1 dead event (2 calls, bettime+closing) |
| phase3 | 1 | 580 | 11,600 | zero failures |
| **Total** | 8 | 4,819 | **51,425 est / 47,735 actual** | **remaining: 52,265** |

Actual spend ran ~3,690 credits under the call-count estimate: the API does
not charge the full 10 credits for event-odds responses with zero bookmakers
(330 such files in 2023-24 alone). Cache after execution: 7,071 files
(from 2,254). Backup at `s:\Documents\odds-api-backup\betting_lines`
re-synced after each phase and after completion (7,074 files incl. log).

Dead events (permanent 404s, do NOT retry -- each retry costs 10 credits):
- `40c02a07bc44ff52fe319ce9b7bf1594` LAK@CBJ commence 2026-01-27T00:00:00Z:
  present in the 2026-01-25 events list, absent from 2026-01-26 -- game
  removed/postponed; both phase1 anchors 404.
- `c9b4e903f451775cc210c14503de80cd` CHI@BUF commence 2024-01-18T00:30:00Z:
  present in events lists but no odds record exists (postponed game); both
  phase2 anchors 404. These two are why converged dry-runs still show 1
  planned call (phase1-bettime, phase1-closing) or 2 (phase2).

Data quality spot-checks (2026-07-09): 2023-24 = 1,312 events, every event
has both bettime+closing snapshots; books williamhill_us/draftkings/fanduel/
bovada (+betmgm from ~Feb 2024, barstool trace); 99.9% of goalie-book quotes
have both Over and Under; 14.2% of bettime and 11.0% of closing snapshots
have no bookmakers (prop not yet posted -- real market behavior, higher than
2025-26's ~3%). Phase1 2025-26 window: 5 books (dk/betmgm/fanduel/bovada/
betonlineag), 99.8% both-sides. Phase3 bulk files: h2h+totals, wide book
coverage incl. betrivers/unibet_us, proper snapshot envelopes. Region
decision: stayed `us`-only (probe showed the us books are the ones with
saves props; us2 would roughly double saves-pass cost -- not purchased).

**Why these purchases (context from 3.14).** The pace_shots result is market
parity with an unproven +9.02% fold ROI, and the Dec 2025-Apr 2026 test fold
is worn. The three highest-value things money can buy now are: (a) a
higher-power evaluation of the *frozen* pace_shots policy via closing-line
value (CLV), (b) fresh out-of-sample season folds (2023-24, and 2024-25 which
we hold odds for but have never used as an evaluation fold), and (c)
timing-safe game-market features (totals/moneylines), the remaining
pre-registered item-8 upside.

**Inventory: what we already hold (audited 2026-07-09 -- never re-buy).**

- `data/raw/betting_lines/cache/`: 1,976 `odds_*.json` files, exactly one
  snapshot per event, market `player_total_saves`, `regions=us`, six books
  (BetMGM, BetOnline, Bovada, DraftKings, Caesars, Fanatics). Snapshot timing
  measured on a 494-file sample: median snapshot-minus-commence = 0 minutes,
  range [-30, +3] -- these are **closing lines**. Coverage: the full 2024-25
  regular season (1,311 events) plus 2025-10-07 through **2026-01-03** (665
  events). Each file preserves the full API envelope
  (`timestamp`/`previous_timestamp`/`next_timestamp`/`data`). Plus 275
  `events_*.json` daily event-list responses for those same dates.
- `data/betting.db`: **bet-time** lines 2026-01-04 through 2026-04-13 (1,789
  rows; Underdog/BetOnline/PrizePicks/BetMGM), with real outcomes.
- `multibook_classification_training_data.parquet` is derived from both, which
  means the shared test fold is a mix: Dec 2025-Jan 3 rows are closing
  lines, Jan 4-Apr rows are bet-time lines. (Relevant when interpreting the
  pace_shots +9.02%: part of it was earned against the close, part against
  bet-time prices.)
- We hold **nothing** for 2023-24, no bet-time snapshots before 2026-01-04, no
  closing snapshots after 2026-01-03, and no game-level markets (h2h/totals)
  at all.
- Risk flag: the cache is NOT git-tracked (`data/raw/` is gitignored). 1,976
  paid responses exist as a single copy on one machine. See open decisions.

**API facts (vendor docs, checked 2026-07-09).** Historical player props
exist from 2023-05-03 (covers all of 2023-24); snapshots at 5-minute
intervals. Historical event-odds cost 10 credits per region per market per
event; historical bulk odds (featured markets, all events in one call) cost
10 per region per market; historical events lists cost ~1 (probe confirms).

**Caching contract -- rules every fetch script must follow.**

1. **Append-only raw archive.** Extend the existing convention in
   `data/raw/betting_lines/cache/`: one file per API response, body stored
   verbatim including the `timestamp`/`previous_timestamp`/`next_timestamp`
   envelope. Filenames: `events_date={ISO}.json`,
   `odds_{eventId}_date={ISO}_markets={mk}_regions={rg}.json`, and (new, for
   featured markets) `bulk_date={ISO}_markets={mk}_regions={rg}.json`.
   Existing files are never modified or deleted; a failed/partial response
   never overwrites an existing file.
2. **Cache-before-call.** A shared module (`src/betting/odds_archive.py`)
   builds a manifest by scanning the cache (filenames + envelopes are the
   manifest; no second bookkeeping file to drift). A planned request is
   SKIPPED when an existing file with the same scope (event or bulk sport,
   markets, regions) has envelope `timestamp <= requested_ts <
   next_timestamp` -- that is exactly the condition under which the API would
   return a snapshot we already hold. No network call happens without a
   manifest miss.
3. **Dry-run by default.** Every fetch script prints the planned calls,
   skipped-as-cached count, and estimated credit cost, then exits; spending
   requires `--execute`. A `--max-credits` hard cap is required on execute;
   the script reads `x-requests-remaining` after every response and aborts
   below a floor; one retry with backoff on 429/5xx.
4. **Per-run report.** planned / skipped-cached / fetched / credits-spent
   (from response headers), appended to a fetch log in the cache directory.

**Purchase phases (gated; do not start a phase before the prior one is
reviewed).** Credit figures assume `regions=us`; the probe revisits that.

- **Phase 0 -- probe (~200-500 credits).** A few events-list calls plus ~6
  event-odds calls sampling 2023-11, 2024-02, and 2025-12 at 22:30Z and at
  commence. Confirms: saves props actually present in the archive for
  2023-24, both sides quoted, which books, whether `us2` adds books worth
  having (it roughly doubles saves-pass costs), and the true per-call credit
  charge from response headers. If 2023-24 saves coverage is poor, stop and
  re-plan before Phase 2.
- **Phase 1 -- test-fold CLV pack (~15.5k credits).** (a) A bet-time pass at
  **22:30Z** (matches the closing-fetch cron time already chosen for the live
  workflow) for every test-fold game 2025-12-04 to 2026-04-16 (~900 events);
  (b) a closing pass at commence for 2026-01-04 to 2026-04-13 (~630 events),
  completing the closing-line record for the whole fold; (c) events lists for
  dates not already covered (~100 credits).
- **Phase 2 -- 2023-24 season pack (~26.5k credits).** Events lists (~186)
  plus a 22:30Z pass and a commence pass for ~1,312 regular-season games.
- **Phase 3 -- featured markets (~11.2k credits).** Bulk `h2h,totals`
  snapshots at 22:30Z daily for 2023-24, 2024-25, and 2025-26 (~558 game
  dates x 2 markets x 10). Closing totals are deliberately not purchased
  initially.
- Total ~54k credits us-only (+~35k more only if the probe shows `us2` books
  matter for saves lines).

**What each purchase feeds (derived artifacts, pre-registered uses).**

- `scripts/build_odds_snapshots.py` -> `data/processed/saves_lines_snapshots.parquet`:
  one row per (event, goalie, book, side, snapshot pass), **per-book, never
  averaged** (the odds-averaging bug, `HISTORICAL_DATA_ANALYSIS.md` section 1;
  the old `scripts/extract_historical_odds.py` averaging pipeline must not be
  reused). Goalie name-to-id matching reuses `build_multibook_training_data.py`
  conventions.
- **CLV audit of the frozen pace_shots policy (interpretation rules written
  now, before the data exists).** Re-price the frozen policy (models from
  `experiment_pace_distributional_20260709_100802`, threshold fixed at 0.05,
  no reselection of any kind) on the 22:30Z prices; for each bet, CLV =
  de-vigged consensus closing probability of the chosen side minus its
  de-vigged 22:30Z probability, plus per-book price CLV. Report mean CLV with
  a goalie-night cluster bootstrap CI. Rules: CI above zero = strong
  confirmation, justifies opening-season shadow + token live stakes; CI
  spanning zero = unresolved, in-season shadow run remains the arbiter;
  CI below zero = treat the +9.02% fold ROI as luck. This is a re-pricing of
  an already-selected policy, not a new selection -- nothing may be tuned
  against it.
- **Rolling-origin confirmation on fresh folds.** Build a 2023-24
  multibook-style frame; run the frozen pace_shots *recipe* (same feature
  families, same config grid, validation-only selection inside each origin)
  as: train <= 2022-23 -> test 2023-24; train <= 2023-24 -> test 2024-25.
  One test touch each; metrics are the paired Brier delta vs the de-vigged
  market with cluster CI (the 3.14 verification statistic) and policy ROI
  with cluster CI. Recorded caveat: the first origin trains on one season
  plus 2021-22 priors -- it tests the mechanism, not the production model.
- **`market_game_features.parquet`.** Timing-safe game-market features from
  the 22:30Z bulk snapshots (game total line and prices, moneyline implied
  probabilities, derived market pace measures) for a future gated experiment.
  Per the 3.14 verification note, this may only meet the worn 2025-26 test
  fold as part of a single pre-season final-exam touch.

**Derived-artifact results (2026-07-09, evening; agent-built, coordinator-verified).**

- `saves_lines_snapshots.parquet` BUILT and verified: 79,884 rows (28,751
  bettime / 51,133 closing), 99.94% both-sides, 99.11% goalie match, 12
  duplicate rows traced to Bovada's own raw responses (kept, documented).
  Supporting fix: `'Utah Mammoth': 'UTA'` added to
  `scripts/build_multibook_training_data.py` TEAM_NAME_TO_ABBREV (the API
  renamed the franchise for 2025-26; without it every Utah game failed to
  match). Timestamps are stored as ISO strings, not datetime64 -- parse
  before comparing.
- **CLV AUDIT RESULT (pre-registered rules, outcome: STRONG CONFIRMATION).**
  `scripts/clv_audit_pace_policy.py` -> `data/processed/clv_audit_bets.parquet`
  (993 bets, 295 goalie-nights, 762 OVER / 231 UNDER). Wiring gate passed:
  the reloaded frozen models reproduced the original 616-bet +9.02% test
  evaluation bit-for-bit before any new numbers were computed. On the
  uniform 22:30Z bettime pass, the frozen policy's bets show mean
  probability CLV **+0.33%** with goalie-night cluster bootstrap 95% CI
  **[+0.25%, +0.42%]** (n=836 with matchable closing consensus, 84.2%
  coverage) and mean price CLV **+0.93%** CI **[+0.64%, +1.24%]** (n=801,
  same-book same-line, 80.7% coverage). Claude independently recomputed
  every per-bet value from the snapshots parquet (max abs diff 0.0) and the
  bootstrap CIs with independent code and seeds (matched). Per the
  pre-registered rule, CI above zero = strong confirmation: justifies
  opening-season shadow + token live stakes. Honest caveats: the magnitude
  is modest (~1/3 of a probability point; ~0.9 decimal-odds ticks of price);
  the window is the same worn Dec 2025-Apr 2026 fold, so this is a NEW
  evidence channel (market movement toward our bets), not a new season.
  Implementation note recorded: the frozen selection code (`calculate_ev`
  via `decide_bet`) compares model probability to the raw vig-inclusive
  single-book implied probability -- the audit replicated that literal
  behavior for selection; de-vigging is used only inside the CLV metrics
  themselves (additive normalization, `tracking_db.devig_prob` convention).
- `market_game_features.parquet` BUILT and verified: 305,940 rows from all
  580 bulk files; 1,316-1,320 games/season with both totals and h2h at the
  latest pregame snapshot; de-vig sums to 1 at float precision; zero
  timing leakage among `is_latest_pregame_snapshot` rows (agent validation
  + Claude's independent 300-event re-derivation). Gotcha handled: ~20% of
  events' `commence_time` drifts across snapshot days as schedules firm up;
  the script canonicalizes each event to its freshest snapshot's schedule
  before computing the pregame flag.
- **ROLLING-ORIGIN RESULT (pre-registered, outcome: CLEAN NEGATIVE).**
  `scripts/experiment_rolling_origin.py` ->
  `models/trained/experiment_rolling_origin_20260709_222639/` (+ new frames
  `multibook_frame_2023_24.parquet` closing, 8,880 rows / 2,298 goalie-nights
  / 87.6% join rate, and `..._bettime.parquet`). Wiring gate passed (frozen
  artifact numbers reproduced bit-for-bit before any new numbers). The
  pace_shots RECIPE (same config grid, validation-only selection, threshold
  0.05, one test touch per origin) at fresh origins:
  - Origin A (train <=2022-23, test 2023-24 closing): paired Brier delta
    **+0.0134** CI [+0.0090, +0.0178] (model significantly WORSE than
    market), policy ROI **-8.30%** CI [-13.9%, -2.6%], 3,895 bets.
    Bettime secondary: delta +0.0119, ROI -6.54% CI [-12.7%, -0.5%].
  - Origin B (train <=2023-24, test 2024-25 closing): delta **+0.0156**
    CI [+0.0099, +0.0213] (significantly worse), ROI -3.00%
    CI [-7.9%, +1.8%], 4,376 bets.
  Claude independently recomputed every statistic from the per-row
  prediction parquets (matched exactly; grading and profit arithmetic
  verified to machine precision; min edge on placed bets exactly at the
  0.05 rule). Pre-registered caveat applies: Origin A trains on ~1,864 rows
  plus 2021-22 priors and Origin B on ~two seasons -- far less data than
  the production model -- so this tests the MECHANISM at low data volume,
  not the production model itself. Judgment calls documented in the run
  metadata: threshold not reselected (no 2022-23 odds exist to sweep
  against), validation windows carved inside the training pool, PMF cap
  70->90 for new origins only.
  **Synthesis with the CLV result:** the two findings are not contradictory
  -- the frozen production-era policy genuinely beat the close on its own
  fold (new evidence channel), while the recipe trained on less data does
  not even reach market parity on fresh seasons. Together they say: the
  production model MAY have a small timing/pricing edge, but the recipe is
  not demonstrably transferable, and the worn-fold selection concern from
  3.14 stands. Consequence for next season: shadow run + token stakes
  (which the CLV rule already capped) -- nothing here justifies scaling
  beyond that, and the in-season shadow run remains the final arbiter.

**Follow-up diagnostics (2026-07-10, Claude, zero credits, read-only
analysis of the existing artifacts). Three measurements that qualify the
results above; none is an edge claim.**

1. **Fresh-origin CLV (Origin A, 2023-24 bettime bets).** The same CLV
   machinery applied to Origin A's 3,328 bettime-pass bets (2,723 OVER /
   605 UNDER, 970 goalie-nights): probability CLV **+0.037%**, cluster
   95% CI [+0.013%, +0.061%] (n=3,272, 98.3% coverage); price CLV +0.070%
   CI [-0.016%, +0.157%]. Positive sign but ~9x smaller than the frozen
   model's +0.33%. By side: UNDER bets +0.084% CI [+0.035%, +0.138%] vs
   OVER +0.026% CI [-0.001%, +0.055%] -- the UNDER selection carries ~3x
   the CLV, echoing the live-record UNDER concentration. Read: the recipe's
   front-running of the close generalizes in sign but not magnitude,
   consistent with CLV scaling with training-data volume (Origin A trained
   on one season). Origin B cannot be CLV-audited: 2024-25 has essentially
   no bettime snapshots (21 events) -- only a closing pass was ever bought.
2. **Unconditional bettime->close drift baseline (honesty check on the
   +0.33%).** Mean drift of the consensus de-vigged OVER probability from
   the bettime to the closing snapshot, all quoted goalie/lines: 2023-24
   **-0.006%** CI [-0.023%, +0.010%] (no drift); 2025-26 audit window
   **+0.128%** CI [+0.095%, +0.161%] (the whole market drifted toward
   OVER). Given the frozen policy's 762/231 OVER-heavy mix, drift alone
   would produce ~+0.07% CLV with zero selection skill. Net selection
   component of the +0.33% is therefore ~+0.26% -- still clearly positive
   (the CI margin is far wider than the drift correction), but the honest
   headline is "+0.26% selection + +0.07% market drift", not +0.33% of
   pure skill. Any future CLV audit must report this baseline alongside
   the bet-level number.
3. **Cross-book dispersion characterization (raw material for a
   stale-book strategy, descriptive only).** Same-line price dispersion is
   tiny: mean absolute deviation of a book's de-vigged prob from the
   same-line consensus is ~0.35-0.55 prob points, and only ~0.4% (bettime)
   to ~1.0-2.4% (closing) of quotes sit >=3 points from the leave-one-out
   consensus -- against a typical ~3.5-point half-vig, same-line price
   shopping across us books is dead as a standalone edge. Line dispersion
   is the real raw material: ~11% of bettime goalie-nights (16-18% at
   closing) have books posting lines >=1 full save apart (a save is worth
   5-6 prob points per 3.9/3.11), i.e. roughly 150-200 candidate nights
   per season. Exploiting it requires translating prices across lines
   (a distribution shape), which is exactly what the distributional model
   prices coherently. Caveat: closing-snapshot outliers (BetOnline is the
   most frequent) may be stale/suspended boards rather than bettable
   prices; bettime snapshots are the trustworthy pool.

**Open decisions for the user (all resolved 2026-07-09).**

1. Back up the paid archive: RESOLVED -- local folder copy at
   `s:\Documents\odds-api-backup\betting_lines`, re-synced via
   `robocopy ... /E /NFL /NDL /NP` after every fetch phase (exit code 1 =
   success-with-copies). Note: robocopy flags are mangled by Git Bash
   (`/E` becomes a path); run it from PowerShell.
2. `us` vs `us,us2`: RESOLVED -- stayed `us`-only (see execution actuals).
3. 22:30Z bet-time anchor: RESOLVED -- used as planned, matinees use
   commence minus 30 min.

---

## 4. Betting strategy for next season

### 4.1 Rule one: UNDER only

Everything -- live money, flat-bet backtests, both chronological halves, both
model generations, every confidence tier -- says the same thing: OVER
recommendations lose (-29.7% ROI on the current model's live sample). Suppress
them. Cheapest implementation: in `predictor.py`'s
`_determine_recommendation()`, never return OVER regardless of EV. Revisit only
if a post-retrain model shows a *live* (not backtest) OVER record above water
on 50+ bets.

**Caveat added 2026-07-07 after the clean retrain (3.1)**: on the clean-data
backtest, UNDER bets also lose at sharp-book odds (-5.82% on the test window,
which overlaps the live record's Feb-Apr 2026 period). "UNDER only" still
stands as a description of the live record, but its causal story -- model
skill vs. Underdog's soft lines vs. a favorable variance draw on n=168 -- is
now an open question. Roadmap items 3-4 (venue discrepancy analysis,
calibration) are designed to answer it before this rule gets automated.

**Update 2026-07-08, see 3.11**: the live record has since been re-graded
night-clustered and against an independent archived line source, and it
survives both -- a favorable variance draw is now the least-supported of the
three candidate explanations above, though none is proven. Automation
decisions should still wait on CLV and the shadow run (4.6), not on this
finding alone.

### 4.2 The UNDER-only Underdog parlay question, answered

This was the open thread in `HISTORICAL_DATA_ANALYSIS.md`. Simulation on the
live record: Underdog-book UNDER recommendations, deduped to one leg per
goalie-night, nightly tickets under Agent E's verified Underdog multipliers
(Power Play 2-leg 3x, 3-leg 6x; Flex 3-leg 3x / 1.5x for 2-of-3).

Leg quality by confidence tier:

| Tier | Legs | Hit rate |
|---|---|---|
| All UNDER recs | 142 | 66.9% |
| 60%+ confidence | 128 | 69.5% |
| 65%+ confidence | 104 | 73.1% |
| 65-70% band | 57 | 80.7% |

Ticket construction results at the 65%+ tier (50 graded nights, Jan-Apr 2026):

| Strategy | Tickets | ROI |
|---|---|---|
| Singles at stored odds (reference) | 104 | +39.4% |
| 2-leg Power Play (top 2 legs) | 27 | +22.2% |
| 3-leg Power Play (top 3 legs) | 15 | +100.0% |
| Adaptive (3-leg if 3+ legs, else 2-leg, else single) | 39 | +59.8% |
| Bucket all legs into 3s | 32 | +78.1% |
| 3-leg Flex | 15 | +70.0% |

The adaptive strategy's bootstrap 95% CI is [+0.4%, +126%] -- positive, but
barely excluding zero, on 39 nights. It is positive in both chronological
halves (+65.8% / +54.1%). Treat the specific ROI numbers as unstable and the
*direction* as well-supported, because the theoretical math carries the real
weight here:

| True per-leg hit | Single @-120 | 2-leg (3x) | 3-leg (6x) | 3-leg Flex |
|---|---|---|---|---|
| 55% | +0.8% | -9.2% | -0.2% | +11.2% |
| 60% | +10.0% | +8.0% | +29.6% | +29.6% |
| 65% | +19.2% | +26.8% | +64.8% | +48.9% |
| 70% | +28.3% | +47.0% | +105.8% | +69.0% |

Even if the observed 73-81% leg hit rates regress hard to 65%, every ticket
size stays solidly positive, and multi-leg tickets *amplify* a real per-leg
edge (this is the mirror image of why parlays are a sucker bet without one).
The earlier "avoid 2-leg tickets" guidance was correct for ~52% blended legs
and is obsolete for a 65%+ UNDER-only leg pool -- at 65% true hit rate, 2-leg
Power Plays clear +26%.

**Recommended nightly rule:**

1. Pool = UNDER recommendations at 65%+ calibrated confidence, Underdog lines,
   one per goalie, never both goalies from the same game on one ticket (their
   outcomes are negatively correlated, which specifically hurts same-direction
   pairs).
2. 3+ qualifying legs: one 3-leg ticket (Power Play for max EV, Flex if you
   want the variance reduction; both are strongly positive at these hit rates).
   5-6 legs: two tickets, split by confidence rank.
3. Exactly 2 legs: a 2-leg Power Play.
4. Exactly 1 leg: straight UNDER at BetOnline/BetMGM at the best available
   price, or skip if the best price is worse than about -135.
5. Nothing qualifying: no action. This happened on roughly half of nights
   (35 of 74); the discipline is part of the edge.

Verify the current multiplier tables in-app before the season -- Underdog has
been moving toward per-leg pricing on some markets, and if goalie saves become
odds-adjusted rather than flat-multiplier, the EV table above must be recomputed
with the actual multipliers shown at ticket time.

**Caveat added 2026-07-07**: the confidence tiers above are the *raw,
uncalibrated* probabilities of the model trained on corrupted data -- the same
probabilities section 1.3 shows are inverted past 0.6 and the clean retrain
(3.1) shows are grossly overconfident. The observed 73-81% leg hit rates are
real, but the tier labels attached to them are not trustworthy probability
statements. After the calibration layer (3.2) exists, re-derive the leg-pool
threshold from calibrated probabilities; until then the nightly rule's "65%+
confidence" gate should be treated as an empirical filter that happened to
work on one sample, not a validated one. **Resolved 2026-07-07: calibration
confirmed the fear -- the calibrated-probability UNDER pool is nearly empty
(25 qualifying bets in 4+ months of test data at even a 2-point edge
threshold; see 3.2). The nightly rule above should not be automated on the
current model.**

### 4.3 Underdog vs. BetOnline (venue allocation)

**Venue discrepancy analysis (roadmap item 3), done 2026-07-07 -- the
soft-line hypothesis is dead in its cleanest form.** On the 248 goalie-nights
(Jan-Mar 2026) where `betting.db` holds both an Underdog line and a sharp-book
line, Underdog's total matched the sharp number exactly on 95.2% of nights
(mean gap -0.04, median 0.00, range [-1.0, +1.0]); the few deviations sat
*below* consensus -- the unhelpful direction for UNDER bettors -- and March
had zero deviations at all. Zero of the model's 137 live Underdog UNDER
recommendations landed on a night with a favorable gap, so an inflated line
cannot explain any part of the live hit rate; legs at zero gap hit 60-70%
(n=20 all-tier / n=10 at the 65%+ tier), statistically indistinguishable from
legs with no sharp comparison (68-73%). The mechanical fade-the-gap strategy
is untestable at this sample size (one UNDER-qualifying night in 3.5 months).

Caveats, honestly stated: coverage is thin and lopsided -- BetOnline was
tracked only in Jan-Feb, BetMGM only in March, never both at once, and April
has no sharp rows whatsoever -- so "sharp consensus" here is always a single
book, and intraday line movement can't be reconstructed (the tracker stores
no fetch timestamp; 60 of 1,479 deduped goalie-night-book groups showed real
line movement across re-fetches). This rules out "Underdog quotes inflated
totals" as a strategy; it cannot rule out subtler venue effects. All headline
numbers were independently re-derived in a second pass from scratch (note for
future queries: the db's `confidence_pct` column is NOT P(side) -- the 65%+
tier must be computed as `1 - prob_over >= 0.65`).

Useful context that fell out of verification: the monthly base rate of UNDER
at quoted lines was 48% / 45% / 56% / 56% for Jan / Feb / Mar / Apr 2026 --
the late-season league-wide saves decline gave every UNDER a ~4-point
tailwind in Mar-Apr (enough to nudge blind 3-leg parlays past their 55.0%
per-leg breakeven, and no more). That explains only a small slice of the live
legs' 68-73% hit rate. What remains on the table for the live profit: model
selection skill that neither completed diagnostic can see, and/or a favorable
variance draw. The calibration layer (3.2) was the next and last cheap
discriminator between those two on the reconstructed frame -- it reported
(2026-07-07) no pre-registered calibrated edge.

**That is no longer the final word.** 3.11 (2026-07-08) re-graded the live
picks against an independent archived line source and re-tested the anomaly
night-clustered rather than flat: it survives both, with a favorable variance
draw now the least-supported of the candidate explanations. The leading
hypothesis is live-pipeline selection behavior that the reconstructed-frame
diagnostics in 3.2 and this section could not see -- see 3.11 for the full
finding and its caveats.

Keep both venues, with defined jobs. Underdog parlays offer the highest EV per dollar
*when 2-3 qualifying legs exist* (at 65% legs: +26.8% for 2-leg, +64.8% for
3-leg, vs +19.2% for a -120 single). BetOnline singles are the outlet for
1-leg nights (~a third of action nights) and the hedge against Underdog
limiting the account -- which it eventually does to consistent winners; expect
shrinking max-entry limits and treat sustained Underdog access as a finite
resource rather than a permanent one. Practical implications: withdraw profits
on a schedule, don't scale stakes faster than needed, and build the BetOnline
single-bet muscle now rather than after a limit hits.

Line shopping is worth real money at the margins: of 488 goalie-nights quoted
at 2+ books, 9% had lines a full point apart. On a 24.5-vs-25.5 spread, always
take the UNDER at the higher number.

### 4.4 Staking

- If betting continues before a new edge is demonstrated, size it as data
  collection, not income. Baseline: **token stakes / <=1% of bankroll per
  ticket**, hard cap ~3% exposed per night, until CLV and next-season results
  show the edge is real. The old 2-3% ticket guidance only makes sense if a
  65% true-leg pool is re-established.
- Expect long losing streaks by construction: a +EV 3-leg ticket at 65% legs
  still loses 72.5% of the time. The bankroll math only works if a 10-ticket
  losing streak (a 4% probability event over any given 10-ticket stretch) is
  boring rather than ruinous.
- Scale stakes with the bankroll (re-anchor the percentage monthly), not with
  recent results.
- The $100 -> $1,800 run may have included some model/window-specific signal,
  but the completed diagnostics did not demonstrate it. Planning assumption
  for next season should be zero proven prior edge, not last season's realized
  multiple.

### 4.5 Tracking upgrades (do these before opening night)

1. **A `tickets` table in `betting.db`**: ticket id, date, venue, entry type
   (power play / flex), stake, multiplier table at purchase, legs (FK to
   `bets` rows), payout. Without it, next season's parlay results are
   unmeasurable again, and every construction-rule question stays
   unanswerable.
2. **Closing line value (CLV)**: record the line/odds at bet time and again at
   puck drop (one extra fetch). Beating the close is the fastest-converging
   evidence that edge is real -- it stabilizes in weeks, not months, and it
   will flag edge decay long before the P/L does.
3. Keep logging *every* line every night (already the habit) -- the 1,755
   graded no-bet lines are what made this entire analysis possible.

### 4.6 Codex-authored: opening-night operating plan under zero proven edge

Authored by Codex, 2026-07-08. The right next-season stance is not "never bet
again" and not "run back the parlay machine." It is a controlled measurement
program. The goal of October and November should be to answer whether the
process beats the close, not to maximize income.

Default operating rules:

1. **No meaningful scaling until CLV is positive.** Bet sizes should stay
   token-small until there is repeated evidence that the process beats closing
   lines. A profitable week without CLV is not enough evidence to scale.
2. **Every ticket must be representable in the database.** If a bet cannot be
   recorded as a ticket with stake, payout structure, legs, bet-time line, and
   close, do not treat its result as evidence.
3. **Separate straight-bet evidence from parlay evidence.** A parlay win can
   hide bad leg selection. Track leg CLV and ticket P/L separately.
4. **Require a reason code for every bet.** Examples: market-anchor model edge,
   stale starter news, line-shop gap, closing-line move still pending, or
   manual hockey-context override. "Model liked it" is not specific enough
   anymore.
5. **One goalie per game per ticket.** Same-game goalie saves can share pace
   and score-state shocks. Keep same-game exposure explicit and limited.
6. **Prefer line advantage over price heroics.** In saves props, a full save of
   line value usually matters more than a few cents of juice. Track both, but
   prioritize the better number.
7. **Review weekly by CLV first, P/L second.** If CLV is negative after a few
   weeks, cut stakes further or pause. If CLV is positive and P/L is bad, keep
   collecting. If both are positive, then consider cautious scaling.

Suggested opening-night thresholds, pending implementation:

- No automated UNDER-only parlay rule from raw model confidence.
- Straight bets only when the process has either a calibrated/model edge plus
  a line-shop advantage, or a documented news/timing reason.
- Parlays only when every leg individually has a reason code and the ticket
  payout table is stored.
- Max exposure per night: low enough that a 10-ticket losing streak changes
  nothing about the project. The point is information, not fast bankroll
  recovery.

What would count as evidence that hope is back:

- Positive CLV on a meaningful sample, ideally by both line movement and
  no-vig probability movement.
- A model or policy that beats a market-only baseline on a later chronological
  slice.
- Edge concentrated in explainable buckets: stale starters, specific books,
  line-shop gaps, rest/context mismatches, or early market openers.
- Repeatability across months, not only a single hot parlay cluster.

What would count as evidence to stop:

- Negative CLV despite positive short-term P/L.
- Positive ROI concentrated in a few tickets with no leg-level CLV.
- Any model improvement that only appears after changing thresholds post-test.
- A strategy whose explanation depends on "the model is confident" without
  calibrated probabilities or market movement support.

---

## 5. Prioritized offseason roadmap

Reordered 2026-07-07 after the clean retrain came back with no backtest edge
(3.1). The old ordering assumed the model had an edge and the remaining work
was policy plumbing; the honest sequence now records the completed diagnostics,
then moves to a decision gate, tracking infrastructure, and new model-edge
experiments.

| # | Project | Effort | Sections |
|---|---|---|---|
| 1 | ~~Fix multibook matching bug, dedupe, drop placeholder-odds rows, regenerate parquets~~ **done 2026-07-07** (see 2.3) | ~a day | 2.3 |
| 2 | ~~Retrain + honest selection (val-only ranking, test touched once, bootstrap CIs)~~ **done 2026-07-07** (see 3.1/3.3 -- result: no backtest edge on clean data) | 1-2 days | 3.1, 3.3 |
| 3 | ~~**Venue discrepancy analysis**: from `betting.db` (Jan-Apr 2026), for every goalie-night quoted at both Underdog and a sharp book, measure the line gap and test whether "UNDER at Underdog when its line sits above sharp consensus" reproduces the live ROI *without any model*.~~ **done 2026-07-07 -- result: Underdog lines are NOT soft.** 95.2% exact match with sharp consensus on 248 checkable nights; zero of the 137 live UNDER legs sat on a favorable gap. Soft lines do not explain the live profit (see 4.3). | half day | 4.3, 3.1 |
| 4 | ~~**Calibration layer** -- the decisive model-edge test: fit on validation, check whether any honest +EV pockets survive, one pre-registered test touch.~~ **done 2026-07-07 -- result: no demonstrated edge.** Validation AUC was only ~0.54 and test AUC fell to 0.513; calibrated probs compress to 0.467-0.620; pre-registered test policies lose or are uninformative; the calibrated UNDER pool barely exists (see 3.2). | half day | 3.2 |
| 5 | **Strategy decision gate -- resolved 2026-07-08 by the user.** Inputs: item 3 (lines not soft), item 4 (no calibrated edge on the reconstructed frame), and 3.11 (the live run survived night-clustered re-testing and independent re-grading against archived lines; the reconstructed-frame backtests never tested the live system). Decision: proceed as a **measurement program**, not a scale-up -- build item 6 immediately, add a shadow run of the exact live system alongside it, keep stakes token-sized until CLV is demonstrably positive, and run items 7 and 9 through the honest harness rather than treat either as a substitute for shadow-run evidence. The UNDER-only + parlay automation (4.1/4.2) should still NOT be built on the current model's raw confidence bands. | half day | 4.1, 4.2, 3.2, 4.6, 3.11 |
| 6 | ~~`tickets` table + CLV capture in the tracker -- unconditional; CLV is the real-time edge detector next season regardless of what items 3-5 conclude. Scope now explicitly includes: line/odds snapshots with fetch timestamps (not just the bet-time line), closing-line capture at puck drop, and a shadow-run log of the exact live system (`tuned_v1_20260201_155204`, live feature pipeline, every recommendation logged whether or not staked) so 3.11's open question can be settled by next season's data instead of another reconstruction.~~ **implemented 2026-07-08; live DB migration applied 2026-07-09.** `line_snapshots`/`tickets`/`ticket_legs` tables plus an idempotent migration (`scripts/add_tracking_tables.py`); snapshot capture wired into `scripts/fetch_and_predict.py`; closing-line + CLV computation (`scripts/compute_closing_clv.py`, wired into the update-results workflow with a graceful no-op if the migration has not been run); phone-first ticket recording (`scripts/record_ticket.py` + `.github/workflows/record_ticket.yml`, `reason_code` required); a CLV report (`scripts/clv_report.py`); and `model_version` tagging on recommendations for shadow-run attribution. Remaining operational note: a pre-puck-drop closing-fetch cron exists commented-out in `fetch_predictions.yml` pending a deliberate usage/cost decision. | ~a day | 4.5, 4.6, 3.11 |
| 7 | ~~Market-anchored residual experiment (implied probability, book, line movement, game total/moneyline) -- the retrain result strengthens the case: the model's huge unanchored disagreements with the market resolve at 49%~~ **done 2026-07-08 -- result: anchoring improves discrimination/calibration, no bettable edge; market-only model has no standalone signal (see 3.4).** | 1-2 days | 3.4, 3.9, 3.10, 3.11 |
| 8 | New hockey-context features (game total/moneyline, opponent rest, special teams, shot attempts/xG, starter/news timing, season normalization). **First current-data slice implemented 2026-07-09 -- result: better prediction, no bettable edge (see 3.12); push-aware true-EV policy audit also negative (see 3.13).** Schedule/rest + prior-only season-to-date shot-volume context improved test AUC/Brier in the distributional model, but the selected betting policies still lost on the single test touch. **Pace/xG Component 3 is now done (see 3.14): `pace_shots` is the first repo model to beat the de-vigged market Brier on this worn test fold (0.24904 vs 0.24961) and selected +9.02% ROI, but the cluster CI still crosses zero. Treat as the strongest offline hypothesis so far, not proven edge.** | 2-3 days | 3.6, 3.9, 3.10, 3.12, 3.13, 3.14 |
| 9 | ~~Distributional saves model prototype, head-to-head vs classifier (trains on all 10,496 goalie-games, no odds required)~~ **done 2026-07-08 -- result: best-calibrated model yet (test Brier 0.25487), +1.06% test ROI with a cluster CI spanning zero -- no demonstrated edge; signal lives in the shots submodel (see 3.5).** | ~a week | 3.5, 3.10, 3.11 |
| 10 | ~~Check The Odds API historical archive pricing for pre-2024 props~~ **resolved 2026-07-09: props history exists back to 2023-05-03; credits purchased; full cache-first acquisition plan (probe, test-fold CLV pack, 2023-24 season pack, featured markets; ~54k credits) pre-registered in 3.15 -- not yet executed.** | ~2 days incl. audits | 3.14, 3.15 |
| 11 | Trivial carryover: `TheOddsAPIFetcher.DEFAULT_BOOKMAKERS = []` fix (`src/betting/odds_fetcher.py:261`) | minutes | -- |

Items 5-6 were the "must happen before opening night" set if any betting
continues: make the strategy decision consciously, then build tickets + CLV so
next season measures the thing actually being bet. They are now done, including
the live DB migration. Items 7-9, the first current-data item-8 slice, the
push-aware true-EV policy audit, and the pace/xG experiment have now been
evaluated through the honest harness. The remaining work is not hyperparameter
search or threshold selection against the same test fold. It is either
next-season confirmation/integration of the `pace_shots` hypothesis or
additional timing-safe information such as game-market ingestion (moneyline,
totals, team totals, movement).

**Update 2026-07-09:** items 6, 7, and 9 are now done (see above), and the
item-6 migration has been applied to the live `data/betting.db`. Item 8 now has
two current-data implementations: game-context features improved prediction but
not betting policy (3.12), while pace/xG produced the first model Brier below
the de-vigged market benchmark on the worn test fold (3.14). The in-season
measurement program (CLV capture, the shadow run) remains the load-bearing next
step; the new question is how to confirm `pace_shots` without mistaking a worn
test-fold win for bankroll-ready proof.

## 6. Codex-authored live implementation log

Authored by Codex, started 2026-07-09. Purpose: concise, observable trail for
the next edge-search phase. This is not evidence of an edge unless the entry
explicitly says the result survived the honest harness and uncertainty checks.

- **2026-07-09 kickoff:** applied `scripts/add_tracking_tables.py` to the live
  `data/betting.db`; verified `line_snapshots`, `tickets`, `ticket_legs`, and
  `bets.model_version` exist. The database is now schema-ready for CLV/ticket
  tracking.
- **2026-07-09 kickoff:** launched parallel Codex sub-agents for (a) data/source
  inventory for hockey-context features and (b) experiment-harness architecture.
  Working hypothesis: if an edge exists, it is more likely in shot-volume/game
  context and timing/line-shopping behavior than in another generic retrain of
  the existing 114-feature classifier.
- **2026-07-09 agent synthesis:** both sub-agents converged on the same next
  slice: a small, pregame-safe game-context feature pack built from existing
  repo data, evaluated first in the distributional shots-against model. Do not
  start with moneyline/totals, confirmed-starter timestamps, or xG unless new
  ingestion is built; current historical odds caches are goalie-saves props
  only, and postgame starter/shot-quality data can leak if used naively.
- **2026-07-09 implementation:** added `scripts/build_game_context_features.py`
  and generated 32 pregame-safe schedule/season-to-date shot-volume context
  features with 100% schedule coverage and unique goalie-game keys.
- **2026-07-09 experiment:** added and ran
  `scripts/experiment_game_context_distributional.py`. Result: context improved
  test row AUC/Brier (`control` 0.5159/0.25487 -> `context_shots`
  0.5360/0.25223; `context_both` 0.5382/0.25204), but both context betting
  policies lost on the single test touch and cluster CIs spanned zero. No
  demonstrated edge.
- **2026-07-09 audit:** independent Codex review found no blocking leakage or
  protocol issue. Carry-forward caveat: the experiment tested the existing
  probability-edge betting rule, not a push-aware true expected-profit policy;
  that policy layer was then tested in 3.13 and did not change the conclusion.
- **2026-07-09 policy wave kickoff:** launched the next Codex sub-agent wave
  for the push-aware true-EV policy layer. Scope is policy math and harness
  evaluation only: compare the old probability-edge rule with true expected
  profit rules that use `P(over)`, `P(under)`, `P(push)`, actual American odds,
  validation-only selection, one test touch, and goalie-night cluster CIs.
- **2026-07-09 policy audit result:** added `src/experiments/policies.py` and
  `scripts/experiment_push_aware_true_ev_policy.py`, then ran the audit against
  the saved game-context distributional artifact. Legacy-policy replay matched
  the source artifact exactly. True-EV and line-shop policies did not win
  validation selection; all three variants selected `old_prob_edge_0.05`, so
  test results matched 3.12. No demonstrated edge, and the obvious policy-math
  caveat is now closed.
- **2026-07-09 (Claude) pace/xG endpoint research + design:** three parallel
  research passes verified the exact data endpoints for shot-attempt/pace/xG
  ingestion. MoneyPuck game-by-game team and goalie files confirmed as primary
  source (NHL-format gameId join, nightly updates, ToS permits listed
  downloads); NHL stats API `team/realtime` report confirmed as a one-call-
  per-season cross-check (totalShotAttempts = Corsi For, verified exactly);
  Natural Stat Trick found Cloudflare-gated behind manual access-key approval
  and skipped per user decision. Full ingestion/builder/experiment design
  pre-registered in section 3.14.
- **2026-07-09 Codex Component 2:** implemented the importable MoneyPuck
  pace/xG feature builder (`src/features/pace_features.py`) and CLI
  (`scripts/build_pace_features.py`). Built `pace_features.parquet`: 10,496
  rows, 45 generated features, unique goalie-game keys, 100% team/opponent
  coverage, 99.99% goalie coverage with only the known Spencer Knight upstream
  gap. Independent recomputation and current-game mutation tests found no
  leakage. This is ready for the Component 3 experiment gate.
- **2026-07-09 Codex Component 3:** added and ran
  `scripts/experiment_pace_distributional.py`. Final artifact:
  `models/trained/experiment_pace_distributional_20260709_100802/`. Control
  gate reproduced 3.12 exactly. `pace_shots` improved test Brier to 0.24904,
  beating the de-vigged market benchmark 0.24961 by 0.00057, and selected
  +9.02% ROI on 616 bets; cluster CI was [-3.12%, +21.15%], so this is a
  promising next-season confirmation hypothesis, not proof of a tradable edge.
- **2026-07-09 (Claude) Component 3 verification + market-parity correction:**
  independently reproduced every pace_shots headline number from the canonical
  artifact with separate pmf/de-vig/grading code, then added the missing
  statistic: the paired model-vs-market Brier delta is -0.00057 with cluster
  CI [-0.00485, +0.00395] on test and +0.00251 (model behind) with CI
  [-0.00422, +0.00904] on validation. Corrected claim recorded in 3.14:
  market parity, not market superiority.
- **2026-07-09 (Claude) Odds API acquisition plan:** audited the existing odds
  archive (`data/raw/betting_lines/cache/` = one at-commence closing snapshot
  per event, full 2024-25 plus 2025-10-07..2026-01-03; `betting.db` = bet-time
  lines 2026-01-04..2026-04-13) and pre-registered the cache-first purchase
  plan in 3.15: probe, test-fold CLV pack, 2023-24 season pack, featured
  markets; ~54k credits; caching contract guarantees no repeat purchases.
  No credits spent yet.
- **2026-07-09 (Claude) acquisition tooling built:** `src/betting/odds_archive.py`
  (manifest scan + skip rule + append-only atomic writes) and
  `scripts/fetch_historical_odds_snapshots.py` (phased dry-run-default CLI
  with credit caps, pacing, and fast-abort hardening), built by a Sonnet
  sub-agent and independently reviewed/dry-run-verified. Archive backed up to
  `s:\Documents\odds-api-backup`. Zero credits spent; probe awaits the
  user's explicit spend authorization.
- **2026-07-09 (Claude) acquisition EXECUTED:** user authorized probe +
  blanket Phases 1-3. All phases run to convergence same day: 4,819
  responses fetched, 47,735 credits actual (51,425 estimated; empty-book
  responses charged less), 52,265 remaining. Cache grew 2,254 -> 7,071
  files; backup re-synced (7,074). Two permanently-404 postponed games
  documented in 3.15 (do not retry). 2023-24 coverage confirmed: 1,312
  events x 2 snapshots, 4-6 us books, 99.9% both-sides quotes, 11-14%
  empty (prop not posted). Full actuals table in 3.15.
- **2026-07-09 (Claude, evening) all four derived artifacts built and
  independently verified** (Sonnet sub-agents built, Claude re-derived every
  headline number from raw inputs with independent code/seeds): (1)
  `saves_lines_snapshots.parquet` 79,884 rows; (2) CLV audit -- frozen
  pace_shots policy shows probability CLV +0.33% CI [+0.25%, +0.42%], price
  CLV +0.93% CI [+0.64%, +1.24%] = pre-registered STRONG CONFIRMATION
  (shadow + token stakes); (3) `market_game_features.parquet` 305,940 rows,
  zero timing leakage; (4) rolling-origin confirmation -- CLEAN NEGATIVE:
  the recipe at fresh origins is significantly worse than market Brier on
  both 2023-24 and 2024-25 and loses money on 2023-24 (-8.3% ROI, CI fully
  negative). Full results + synthesis in 3.15. Net posture for next season
  unchanged by the good CLV news: shadow run + token stakes, in-season
  shadow is the arbiter.
- **2026-07-10 (Claude) three zero-credit follow-up diagnostics** (full
  numbers in 3.15 "Follow-up diagnostics"): (1) Origin A's bettime bets
  also show positive but ~9x smaller CLV (+0.037% CI [+0.013%, +0.061%]),
  UNDER side ~3x OVER -- the CLV effect generalizes in sign, scales with
  training data; (2) drift baseline -- the 2025-26 audit window drifted
  +0.128% toward OVER market-wide, so ~+0.07% of the frozen policy's
  +0.33% CLV is mix-times-drift, net selection ~+0.26% (2023-24 drift is
  zero); (3) cross-book dispersion -- same-line price shopping is dead
  (deviation ~0.4 prob pts vs ~3.5-pt half-vig), but ~11% of bettime
  goalie-nights have books >=1 full save apart in line, the raw material
  for a cross-line/stale-book strategy requiring distribution-shape
  translation. These motivate the next experiment family: model the LINE
  (movement prediction, cross-line outlier pricing) rather than another
  attempt to out-predict outcomes.
- **2026-07-10 (Claude) reviewed and merged the Codex-authored
  `docs/BREAKTHROUGH_MODEL_PLAN.md`**, now the single canonical plan for the
  next research program. Claude verified Codex's section 3.1 market-data
  claims exactly (audit bettime lead time median 1.67h -- the +0.33% CLV is
  a late-window result; movement frequencies match to the decimal); Codex's
  section 2 shots-bias diagnostics remain unverified and are step 0 of the
  merged execution sequence. Additions in the merge: Claude's three
  diagnostics as section 3.4, mandatory drift-baseline and side-split
  reporting for the movement model, Component G (cross-line outlier
  pricing, no purchase needed), and an alternate-lines probe. Budget
  ceiling 41,340 of 52,265 credits, all purchases probe-gated; nothing
  executed yet.
- **2026-07-10 (Claude) operational constraints added to
  `docs/BREAKTHROUGH_MODEL_PLAN.md` as section 1a** after user review: one
  early-evening decision window (the 22:30Z anchor -- the same window the
  CLV evidence comes from), venues limited to Underdog/PrizePicks (line and
  ticket-construction levers only) and BetOnline (straight bets); the six
  sharp books in the data are consensus instruments, not executable venues.
  Component F reframed as anchor validation, Component G's deployed form is
  a venue-relative bet-time filter (`scripts/check_venue_value.py`, to be
  wired into the daily workflow before opening night), Gate C now requires
  value to survive at the user's real window and venues, and the sequence
  gained a deployment step. 2026-27 is explicitly framed as a measurement
  season.
- **2026-07-10/13 (Claude, six parallel Sonnet sub-agents, zero credits)
  executed steps 0-5 of `docs/BREAKTHROUGH_MODEL_PLAN.md` section 10.**
  Two session-limit interruptions occurred mid-run; all agents resumed
  cleanly from saved transcripts with no lost work. Full detail and pass/
  fail bars in `docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 10
  (Results); plan-level verdicts in `docs/BREAKTHROUGH_MODEL_PLAN.md`
  section 7 (Gate A) and section 4.7 (Component G). Summary:
  - **Step 0 (verify Codex's section 2 diagnostics): all six CONFIRMED**
    by reloading the frozen origin artifacts -- shots bias +1.9501/+1.8445,
    saves bias +1.7863/+1.7511, no-pace-control ablation reproduced at
    +0.4420/+0.0308, the 30.31/29.18/27.41 starter-SOG seasonal decline
    while raw attempt volume rose then flattened, and the in-sample
    dispersion-fitting bug confirmed in code with Origin B's frozen
    artifact shown to have silently fallen back to a Poisson (alpha=0)
    distribution when true held-out dispersion implied alpha ~0.027. One
    downgrade: the "control still trailed the market" claim is solid for
    Origin A but statistically marginal for Origin B (CI touches zero).
  - **Step 2 (season-normalized funnel) and step 3 (exposure mixture):
    both FAIL Gate A.** Season-normalization roughly halves Origin A's
    shots bias but barely moves Origin B; the attempt-to-SOG funnel does
    the reverse. Neither beats the no-pace control's Brier on both
    origins, and lower-tail calibration gets worse under every
    pace-informed variant. The exposure classifier (P(TOI<50)) has
    essentially no discrimination (AUC 0.52-0.55, log loss/Brier
    statistically tied with a constant-base-rate baseline), so the
    exposure mixture fails its lower-tail coverage bar against the
    correct baseline (no-pace control with validation-fitted dispersion
    already applied). **The one clear, reproducible win across all three
    independent agents: validation-fitted dispersion** (never train
    residuals) -- alpha values matched closely across step 0, the funnel
    experiment, and the exposure experiment.
  - **Step 4 (market game-state features): real but partial.** Origin A's
    entire training pool sits inside 2022-23, which has zero
    `market_game_features.parquet` coverage, so its result is
    uninformative by construction. Origin B, with 42% train / 100% val
    coverage, shows a real paired Brier improvement (-0.00414, cluster CI
    [-0.00720,-0.00118], market_state moved Origin B's Brier from worse-
    than-market to better-than-market). Fails the pre-registered
    both-origins bar; not currently blocked by Gate A since Component C
    was never one of its four bullets.
  - **Step 5 (Component G development, 2023-24 only): no confirmed edge,
    and the venue-relative deployment concept is untestable this season.**
    The pre-registered threshold lock failed outright -- no gap threshold
    produced the minimum 20 graded bettime bets required, so the fallback
    threshold (0.02, 11 bettime bets) is explicitly not a usable policy
    lock. All ROI/CLV confidence intervals at every threshold cross zero.
    Flagged bets are heavily concentrated in one book (fanduel, 9/11
    bettime), consistent with the "single lagging book" theory but not a
    general cross-book finding this season. The drift baseline cross-
    checked against the earlier diagnostic within 0.01 probability points
    (pipeline correctness confirmed) and the candidate-pool size matched
    the earlier forecast (10.6% bettime / 15.5% closing vs. the ~11%/16-
    18% estimate). **Critical finding: `betonlineag` has zero quotes
    anywhere in the 2023-24 archive** -- coverage only starts in 2024-25 --
    so the one venue the user can actually execute straight bets at cannot
    be evaluated with this season's data at all. Per the pre-registration's
    own single-touch discipline, the 2024-25 closing-pass test touch does
    NOT proceed on this unlocked policy.
  - **Net effect on the plan:** Gate A, as scored on the combined
    architecture, currently fails -- no purchase (the SOG probe/pass, the
    2024-25 opening saves pass) is authorized by this round's evidence.
    Nothing here overturns the frozen production model's verified CLV
    finding (section 3.15) or the live 2025-26 UNDER record (section
    3.11); it narrows which of the *new* research directions still have
    life in them. Validation-fitted dispersion is adoptable now at zero
    further cost. Component G needs either a materially larger flagged-bet
    sample (more seasons/lines) or 2025-26 data (where BetOnline coverage
    should exist) before its central hypothesis is testable at all.
    (Correction 2026-07-13, see the next entry: the validation-fitted-
    dispersion "adoptable now" claim in this paragraph was subsequently
    corrected -- it fails its own registered central-coverage bar on
    Origin A.)

- **2026-07-13 (dual independent audit of the six-experiment round: Codex
  and Claude, given the identical audit prompt, each re-verified the round
  from raw artifacts -- script review for leakage, fold checks, own-code
  recomputation of Brier/ROI/CLV/AUC/bootstrap CIs, and bias via reloading
  the saved model JSONs against a harness-rebuilt frame).** Both audits
  agree: the experiments genuinely ran, folds are correct, no outcome
  leakage, and every major number reproduces (bias to 4 decimals, CIs to
  bootstrap noise; the three agents' control models are prediction-
  identical, confirming cross-agent fold alignment). Both audits also
  converged on the same defects, now corrected in
  `docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md` (amended Experiment 3 table
  row + new section 10.1) and `docs/BREAKTHROUGH_MODEL_PLAN.md` (Gate A
  verdict correction + rewritten section 10 step 6):
  - **Experiment 3's PASS corrected to FAIL**: summed central-coverage
    deviation worsens on Origin A under val-fitted dispersion (3.36 ->
    7.80) while improving on Origin B (8.43 -> 4.22) -- the registered
    both-origins bar fails. The declaring run never computed the
    train-fitted comparison (diagnostic-only alpha, no test predictions);
    the exposure run's metadata contained the disproof. Lower-tail gains
    and the Origin B Poisson-fallback fix remain real; the corrected
    reading is that single-alpha NB2 cannot fit middle and lower tail
    simultaneously.
  - Experiment 2 deviations recorded (starter-share exposure stage instead
    of the registered per-60 x projected-minutes construction; 4
    duplicated z-score features); Experiment 6 reframed as "gate failed,
    mechanism inconclusive" (its pooled mixture had the round's best
    marginal tail calibration; the classifier, not the shape, is what
    failed); Experiment 7 contract gaps recorded (20-bet minimum lived in
    code not in the binding registration; off-modal-line requirement never
    enforced -- 12 of 42 closing flags were at the modal line).
  - **New post-hoc observation (verified by both audits, registered as
    hypothesis only)**: Origin B 2024-25 closing, market-state variant,
    UNDER picks at the fixed 0.05 EV threshold: +11.18% ROI/bet all books
    (cluster CI [+4.21%, +18.03%], n=2031/762 nights), +8.66% at BetOnline
    (CI [+0.53%, +16.57%], n=513). Blind bet-every-UNDER on the same quote
    universe: +1.06% (CI [-2.77%, +4.82%]) -- so the return is quote
    SELECTION, not the 2024-25 season-wide UNDER drift. Post-hoc slice;
    authorizes nothing except the pre-registered replication below.
  - **Next round decided (all zero-credit; written as plan section 10 step
    6a-6e)**: (6a) Origin C replication of the frozen market-state recipe
    -- train through 2024-25, test 2025-26, primary = paired Brier vs.
    control + selection-over-blind-UNDER delta on the BetOnline bettime
    cut (1,212 goalie-nights verified on disk); (6b) fixed-offset
    deterministic funnel (the registered-but-never-tested construction);
    (6c) fixed-weight early-exit mixture scored as a standalone shape
    against the corrected dispersion criteria; (6d) Component G contract
    repair + zero-outcome-touch 2025-26 volume recon only; (6e) purchases
    stay blocked until 6a reads out -- if it replicates, promote to the
    2026-27 shadow program and reconsider the 2024-25 bettime-pass
    purchase; if not, the market-state result was an Origin B artifact.

- **2026-07-13 (Codex-authored: independent Origin C / Experiment 8
  audit and consequence).** The frozen market-state recipe completed its
  pre-registered Origin C run on 2025-26. Codex reviewed the 1,147-line
  implementation, verified the season fences and book-key provenance,
  and independently rebuilt every headline statistic from the saved
  prediction parquet plus the raw closing and bettime quote parquets with
  different bootstrap seeds. The wiring gate reproduced Origin B
  bit-exactly. **P1 passes:** market-state minus no-pace-control closing
  paired Brier `-0.003111`, CI95 `[-0.005039, -0.001192]`, 5,729 rows /
  2,070 goalie-night clusters. **P2 is INSUFFICIENT SAMPLE:** only 85
  BetOnline bettime UNDERs met the frozen 0.05 probability-edge rule,
  below the registered 100-bet floor; selected ROI `-11.40%` versus
  blind-UNDER `-5.24%`, delta `-6.16` points, CI95
  `[-25.36, +13.25]`. The all-books selection secondary was also null.
  The Origin B post-hoc UNDER lead therefore did not replicate; 2025-26
  threshold returns were carried by OVER instead. Market-state remains a
  real incremental feature block -- it beat the no-pace control in a
  second eligible origin and slightly improved shots MAE -- but it tied
  the closing market and was significantly worse than the bettime market.
  Drift-adjusted flagged-bet CLV was statistically positive but small
  (`+0.12` probability points on 1,073 matched bets / 307 clusters all
  books; `+0.19` on 240 BetOnline bets), and does not establish a
  vig-clearing outcome edge. Per the registered mapping:
  **no promotion, no purchase, and no executable edge claim.** Plan step
  6b (deterministic funnel as a fixed offset) is now first in queue; the
  6c fixed-weight early-exit shape is still unrun and should be bundled
  into that evaluation, while 6d remains contract repair and
  outcome-blind volume reconnaissance only.

- **2026-07-13 (Codex-authored: plan step 6b/6c execution started).**
  Began the next zero-credit architecture step: a deterministic,
  prior-only attempt-to-SOG workload projection used as a fixed base,
  with XGBoost restricted to learning the residual, and a fixed-weight
  pooled early-exit mixture evaluated as the distributional shape rather
  than as an individualized pull-risk classifier. Two independent
  read-only agents are tracing the exact reusable funnel, shots-per-60,
  TOI-bin, and corrected coverage machinery before the binding contract
  is appended to
  `docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md`. No model has been run
  and no Origin A/B result has been viewed for this new architecture.
  **Contract now locked in preregistration section 12:** the residual is
  a Poisson log-offset model; the fixed mixture is mean-matched against
  the identical single-NB2 arm; exposure weights/bins are train-only;
  body dispersion is exposure-conditioned on validation and shared by
  both arms; central coverage and aggregate lower-tail error must improve
  on both origins. The implementation/run worker has been launched with
  ownership limited to one new script and one new artifact directory.

- **2026-07-13 (Codex-authored: Experiment 9 verified result).** The first
  valid registered run completed once with both
  no-pace wiring gates exact and no alternate candidate rerun. The
  verified Gate-A verdict is FAIL: shots bias missed the `0.5` bar on
  both origins (`+0.78`, `+1.99`), and closing Brier worsened versus
  control on both (`+0.00216`, `+0.01526`; the latter CI excludes
  zero). The fixed mixture did improve aggregate lower-tail error and
  negative log score on both origins, but central coverage improved only
  on B and the B bet rate inflated to `59.3%`. Codex plus two independent
  audit agents verified the code path, folds, leakage fences, every
  `base_margin` call, model reloads, persisted PMFs/quotes, P1-P5, and
  fresh-seed cluster CIs. No material defect was found. Two label/guard-
  only fixes were made without rerunning the candidate. The lesson is
  narrow but useful: constant pooled early-exit mass is a better marginal
  distribution shape, yet it smears tail probability across every start
  and cannot rescue an inaccurate mean workload model or posted-line
  Brier. The fixed-offset lead is closed; purchases remain blocked and
  plan 6d's contract repair plus outcome-blind volume reconnaissance is
  next.

- **2026-07-13 (Codex-authored: Component G repair/recon started).** Two
  independent read-only audits traced the failed Experiment 7 lock and
  the 2025-26 snapshot schema before any new candidate count was viewed.
  The repaired contract is now frozen in preregistration section 13. It
  requires the accessible `betonlineag` quote to sit outside every tied
  modal line, moves the 20-bet floor into binding text, uses only the
  frozen NB2 scorer and threshold grid for counting, and fails closed on
  duplicate/conflicting quotes. The run is restricted to one selectively
  read snapshot parquet and cannot load outcomes, grade bets, compute ROI
  or CLV, select a threshold, or spend credits. Its only decision is
  whether at least 20 unique candidates exist at the most permissive
  `0.02` threshold, making a future separately registered validation
  arithmetically possible.

- **2026-07-13 (Codex-authored: Experiment 10 verified result).** The
  repaired Component G reconnaissance is complete and its registered
  verdict is **TOO SPARSE**. From 1,185 paired BetOnline bettime goalie-
  nights, only 31 quotes were strictly outside the complete modal-line
  set. The frozen cross-line translator flagged just one candidate at the
  most permissive `0.02` probability-gap threshold (`0 OVER / 1 UNDER`)
  and zero at every higher threshold, far below the binding minimum of
  20. No outcome or closing data was loaded, the candidate was not graded,
  no credits were spent, and no threshold was selected. Codex audited the
  script/artifact, an independent implementation reproduced every count
  from the selectively read snapshot parquet, and a separate static audit
  found no material defect. The only minor note is that sparse aggregate
  slices describe the lone anonymous candidate narrowly; no identifier or
  outcome is persisted. **Consequence:** the strictly off-modal BetOnline
  strategy is arithmetically nonviable in this archive and is closed
  without an outcome touch. Purchases remain blocked.

- **2026-07-13 (Claude-authored: next-wave ideation, dual-sourced, and
  the expiring-credit reframe).** After the 6a-6e readout closed the
  prior queue, the user redirected effort at historical data with
  approximately 50,000 Odds API credits expiring 2026-07-31 -- which
  flips the plan's purchase posture from "blocked pending evidence" to
  "spend on the highest-information archives this month." Two parallel
  Sonnet research agents (one mining on-disk data, one external) plus an
  independent Codex ideation round converged on the same headline
  discovery: The Odds API's `us_dfs` region carries `underdog` and
  `prizepicks` as named historical bookmaker keys -- venue-exact history
  the project previously assumed was only collectable prospectively.
  Claude independently verified the key vendor facts: no Pinnacle NHL
  saves in any region (settled), `betonlineag` in plain `us`, the
  10-bookmaker billing rule documented for live but NOT historical
  endpoints (probe-verifiable), `includeMultipliers` real, dense intraday
  passes unaffordable. Amended same day after user review: one Codex
  claim was merely softened and one Claude "correction" was itself wrong.
  (a) The 10-bookmaker billing rule is documented on the standard odds
  endpoint whose parameters the historical event-odds endpoint states it
  inherits -- the probe should still confirm actual billing from usage
  headers, but calling the rule undocumented for historical calls was
  overstated. (b) Claude wrongly declared the play-by-play files
  nonexistent after checking only `data/raw/boxscores/`;
  `data/raw/play_by_play/` holds exactly the 5,248 event-level files
  Codex described (already documented in MODEL_TRAINING_GUIDE.md), so
  wave W5 parses the existing archive rather than fetching anything.
  The same review also flagged: the W2 prior must be reconciled with the
  2026-07-07 venue analysis (95.2% exact agreement on 248 nights vs the
  recon's 90.1% on 294 rows -- different windows/comparators/dedup
  rules); the recon's microstructure statistics are unclustered and
  script-less, so W3/W6 stay exploratory leads pending clustered
  re-verification from persisted scripts; the 2024-25 P2 re-test must
  use the frozen Origin B market-state model, never Origin C (whose
  training pool extends into 2024-25 and would leak); and the
  `line_snapshots` table is already wired into the live fetch
  (`scripts/fetch_and_predict.py`) but the 2026-07-09 migration
  postdates the season's last game -- the real open item is the
  commented-out closing-fetch schedule. The zero-credit recon results
  (steam null after
  dedup, DFS tracker staleness null, total-vs-saves-line null, juice
  skew replicated but sub-vig, BetOnline convergence the strongest
  exploratory lead but still unclustered, unpersisted, and sub-vig)
  are written into HISTORICAL_DATA_ANALYSIS.md section 8. The finalized
  acquisition plan and next experiment wave (W1-W6) are in
  BREAKTHROUGH_MODEL_PLAN.md sections 5.7 and 10. No credits spent yet;
  probes awaited user authorization at the time of that entry (superseded
  immediately below).

- **2026-07-13 (Codex-authored: revised W1 acquisition probe executed and
  independently verified).** The user authorized the bounded probe only.
  `scripts/probe_opening_markets.py` made 24 historical event-odds calls
  (eight per season) for standard/alternate saves and SOG across nine named
  books with `includeMultipliers=true`; all returned HTTP 200. Actual spend
  was 800 credits (170/310/320 by season), leaving 51,465. The two core
  purchase maximums (26,230 SOG + 13,110 saves) would leave up to 12,125 for
  a probe-informed remainder. Billing is settled:
  every call's `x-requests-last` equaled `10 * unique returned markets`, so
  nine named books cost one region-equivalent. `scripts/audit_w1_probe.py`
  independently rebuilt the raw responses and joined all 24 games to the NHL
  archives; a Luna audit separately verified every request signature, scope,
  response event, quota transition, and coverage count. Standard sportsbook
  SOG passes the coverage gate: at least two usable books on 24/24 events,
  player-event paired-line completeness 98.53%/100%/99.69% by season, exact
  season-team resolution 347/347, and median 12-15 listed skaters accounting
  for about 47%-61% of actual combined SOG at the book-season median. DFS
  saves are narrower: no DFS data in 2023-24; PrizePicks saves 7/8 in each of
  2024-25 and 2025-26; Underdog saves 0/8 in both. Alternate saves are absent
  in 2023-24 and available mainly at BetOnline thereafter, but all 455 sampled
  outcomes are over-only. Non-null multipliers exist only for Underdog
  2025-26 SOG, never saves. **Consequence:** the two-season sportsbook SOG
  pass clears its acquisition gate and named-book billing for the 2024-25
  saves pass is confirmed, but neither full purchase was authorized or run.
  Close the 2023-24 DFS allocation; narrow W2 to PrizePicks history plus
  prospective Underdog; use alternate-saves credits only for an explicitly
  one-sided ladder design. Full actuals are in BREAKTHROUGH_MODEL_PLAN.md
  section 5.7 and HISTORICAL_DATA_ANALYSIS.md section 9.

- **2026-07-13 (Claude-authored: core purchases authorized; remainder
  decision revised).** The user authorized both core purchases: the
  two-season sportsbook SOG pass and the 2024-25 bettime saves pass. Two
  remainder proposals were rejected before execution. Codex's ~11,000-credit
  alternate-saves ladder purchase was scaled back (over-only quotes cannot be
  de-vigged without an unverified one-sided model; a 1-2k pilot must come
  first). Claude's counter-proposal to buy a 2024-25 closing saves pass was
  itself wrong and was withdrawn after direct verification against
  `saves_lines_snapshots.parquet`: 2024-25 already holds 14,954 closing rows
  across 1,288 events; only the bettime side is missing (258 rows, 21
  events). CLV grading and the W6 second-season pairing therefore unlock as
  soon as the bettime pass lands, at zero extra cost. Execution design: a new
  dedicated script `scripts/purchase_core_bettime_passes.py` following the
  probe's append-only signature-cached record pattern (records under
  `data/raw/betting_lines/passes/core_bettime_202607/`), 2024-25 fetched as
  one combined call per event (`player_total_saves,player_shots_on_goal`, 20
  credits max/event), 2023-24 as SOG-only (10 max/event), nine named books at
  the bettime anchor, credit floor 11,500 protecting the remainder. The
  remainder allocation itself is deferred until purchase actuals land.

- **2026-07-14 (Claude-authored: core purchases executed and independently
  audited).** Both passes completed: 2,626 append-only records under
  `data/raw/betting_lines/passes/core_bettime_202607/`, total spend 38,570
  credits, balance 51,465 -> 12,895, reconciled exactly against response
  headers by a persisted audit (`scripts/audit_core_bettime_passes.py` +
  `audit_summary.json`). One mid-run fix: the first combined-2024-25 run
  aborted at call 649 on an HTTP 404 (the wildfire-postponed CGY@LAK
  2025-01-08, whose event id was reissued); the script was amended to treat
  404/EVENT_NOT_FOUND as a cached, free, per-event answer and continue,
  while all other non-200s still abort. Both 404 games (the other:
  CHI@BUF 2024-01-18, already documented as a dead event) are covered under
  replacement ids, so no game is missing. Headline coverage and the binding
  ingestion caveats (FanDuel duplicate rule, zero Fanatics, commence-drift
  re-anchoring) are recorded in HISTORICAL_DATA_ANALYSIS.md section 9.4 and
  CURRENT_HISTORICAL_DATA.md section 4.2. Next actions: preregister W1/W2
  (and the frozen-Origin-B P2 re-test) before any analysis touches the new
  data; decide the 12,895-credit remainder, starting with the 1-2k
  alternate-ladder pilot.

- **2026-07-14 (Codex-authored: core-pass ingestion independently
  verified).** The frozen-Origin-B P2 re-test, W1 cross-market coherence
  model, and W2 DFS census are now bindingly preregistered as Experiments
  11-13 in `PREREGISTRATION_NO_CREDIT_ABLATIONS.md` sections 14-16.
  `scripts/build_core_bettime_pass_snapshots.py` then produced a new,
  non-mutating processed artifact with 413,758 rows / 23 columns: 2023-24
  SOG 214,252 rows / 1,310 events, 2024-25 SOG 182,686 / 1,301, and 2024-25
  saves 16,820 / 1,244. The registered 10-minute true-puck-drop floor
  excluded exactly three events (BUF@NJD -24.8 minutes, BOS@PHI 5.0,
  PHI@OTT 9.2), not all 80 events flagged by the audit's different
  cached-commence drift diagnostic. After those exclusions, 5,282 exact
  duplicate outcome copies were dropped. The audit's old 5,296 total used
  a price-omitting key: 5,293 were truly identical before drift exclusion,
  while three were conflicting-price groups. All six rows in those groups
  were excluded fail-closed, not tie-broken. Independent reconstruction
  from all 2,626 raw records matched every final outcome key; no Fanatics,
  null line/price, invalid side, or sub-10-minute row survived. Eleven new
  2024-25 events overlap the old 21-event bettime fragment and must be
  deduplicated downstream. No registered experiment has touched the new
  price-level data. At that checkpoint the immediate analysis step was
  Experiment 11's frozen wiring gate and one-touch P2 re-test; the next log
  entry records its completion. The credit remainder is still 12,895.

- **2026-07-14 (Codex-authored: Experiment 11 frozen-Origin-B P2 PASS).**
  The mandatory closing wiring gate reproduced exactly before the new pass
  was opened. The new-pass-only BetOnline primary had 1,719 paired quotes and
  473 fixed-threshold UNDER selections: model ROI `+12.29%`, blind-UNDER ROI
  `+2.63%`, delta `+9.66` points, goalie-night cluster CI95
  `[+2.49, +16.72]`. Train-fitted dispersion agreed (`+9.47`, CI95
  `[+2.35, +16.52]`), as did the all-books secondary. Independent
  reconstruction from persisted row-level universes reproduced the result
  exactly. BetOnline venue CLV did not confirm the outcome result: full-policy
  CLV net of drift was `+0.0167` probability points with CI95
  `[-0.0627, +0.0979]`; all-books CLV was positive. Per the locked consequence
  mapping, promote only to a frozen 2026-27 shadow candidate. This is the
  best honest executable selection evidence so far, but it is still from an
  already-viewed season and is not proof of durable edge. Full result:
  preregistration section 14.11; completed artifacts:
  `models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/`.

- **2026-07-14 (Codex-authored: Experiment 13 W2 DFS census NULL).** The
  fixed 2024-25 development -> 2025-26 confirmation census is complete and
  independently reproduced. PrizePicks differed from sportsbook consensus
  on 443 of 1,868 comparable 2024-25 goalie-nights, but the 420 gradeable
  non-push deviation candidates went 212-208: 50.48% hit rate and `+0.95%`
  per bet under an even-money scoring convention, cluster CI95
  `[-8.57%, +10.48%]`. The 2025-26 archive offered only six deviations;
  five wins produced CI95 `[0.00%, +100.00%]`, which fails the registered
  strict lower-bound bar and is not meaningful confirmation. This is not
  PrizePicks ROI because historical multipliers are unavailable. The prior
  rates now reconcile to Underdog `236/248 = 95.16%` and PrizePicks `51/57
  = 89.47%` under one definition; the old 90.1% and 50/64 samples cannot be
  exactly reconstructed. Close DFS staleness for this cycle. Final artifact:
  `models/trained/experiment_13_w2_dfs_venue_history_20260714_100855/`;
  full contract/result: preregistration section 16.9.

- **2026-07-14 (Codex-authored: Experiment 12 W1 recovery consumed; observed
  calculation encouraging).** The 2023-24 development recipe froze `0.03`
  thresholds before touching 2024-25. The original touch failed before any
  performance statistic; the registered 12R recovery completed calculations
  but then crashed while writing metadata, so the recovery is consumed and
  the official verdict is `NO VERDICT -- INFRASTRUCTURE FAILURE`. Independent
  reconstruction exactly reproduced the persisted result and all 20,000
  bootstrap draws. OVER lost `-8.34%` but beat blind OVER by `+7.17` points,
  CI95 `[+2.73, +11.64]`; UNDER returned `+11.12%` and beat blind UNDER by
  `+9.28` points, CI95 `[+2.49, +16.02]`. Global Brier/log-loss were worse
  than the market, while selected probability CLV was positive but under one
  tenth of a percentage point. Preserve the exact W1 recipe for a 2026-27
  shadow run. Do not claim an official PASS, do not rerun 2024-25, and do not
  infer that both sides were profitable. Full record: preregistration section
  15.11 and historical analysis section 9.8.

- **2026-07-14 (Experiment 14 W6 BetOnline convergence CLOSED).** The
  registered clustered re-derivation (preregistration section 17)
  re-tested the exploratory scratchpad lead (r=-0.147, nominal
  p=2.0e-10, n=1,851 unclustered correlated rows, 2025-26 only) under
  goalie-night-unit, UNDER-side-collapse, >=2-book-consensus
  definitions on both available seasons. Phase A (2025-26, discovery
  season, in-sample): r=-0.05019, CI95 [-0.10550, +0.00543], n=931
  goalie-nights -- roughly one third the original magnitude and
  including zero. Phase B (2024-25, already-viewed development data):
  r=-0.05829, CI95 [-0.12429, +0.00488], n=1,380 goalie-nights, also
  including zero. Sign was negative in both seasons but neither CI
  cleared the registered below-zero bar. The 17.5 EV-stacked filter
  test on the 473 frozen Experiment 11 model-arm bets was run anyway
  as EXPLORATORY-ONLY per protocol: agree-arm (n=260) ROI delta
  -9.1613 points vs. the full-population reference, CI95 [-20.384,
  +1.761]; non-agree-arm (n=111) delta +9.1871 points, CI95 [-7.688,
  +25.661] -- the opposite ordering from the registered sign
  rationale, both CIs crossing zero, neither reported as a finding.
  All 11 structural reconciliation checks against the preregistration's
  data-inventory counts passed exactly. Per the fixed consequence
  mapping, the lead is CLOSED this cycle with no shadow-candidate
  registration, matching the steam-recon and DFS-census precedents.
  Full result: preregistration section 17.9; historical analysis
  section 9.9. Artifact:
  `models/trained/experiment_14_w6_betonline_convergence_20260714_142506/`.

## 7. Appendix: what was checked and found sound

For balance, the things audited during this deep dive that do **not** need
fixing:

- **Leakage discipline in features**: the `.shift(1)` pattern is applied
  consistently in the live pipeline; the inference-time date filter in
  `feature_calculator.py` structurally excludes the current game. No new
  leakage found.
- **Feature importance is sensible**: the top features by total gain are
  exactly what a domain expert would want -- `line_vs_opp_implied_saves`
  (4.7%), `goals_against_rolling_10` (4.4%), `expected_workload_diff` (3.6%),
  `line_z_score_10`, `betting_line`. The Feb 1 line-sensitivity fix genuinely
  took: 108 of 114 features get splits, and line-relative features carry real
  weight.
- **The chronological split policy** is right for this problem and should not
  be relaxed (see `CURRENT_HISTORICAL_DATA.md` section 6).
- **The regularization-heavy hyperparameter philosophy** (reg beats depth) was
  established by a fair experiment and matches the sample size; no reason to
  revisit until the training data roughly doubles via the section 2 fix.
- **`predicted_saves` in the tracker is cosmetic** (line + 5x probability
  offset, `predictor.py:147`) -- fine for display, but nobody should ever
  analyze it as a real regression output.
- **Blind UNDER is not the edge**: the graded-lines base rate is 51.6% under,
  which loses to vig. The live model's UNDER *selection* (69%) was real in
  the database, and it has since survived night-clustered re-testing and
  independent re-grading against archived market lines (3.11) -- too extreme
  to comfortably call a variance draw, though the clean retrain/calibration
  work still did not demonstrate the mechanism on reconstructed features.
  Treat it as an unresolved live anomaly whose leading explanation is now
  live-pipeline selection behavior that neither backtest could see (3.11),
  pending CLV or a shadow run.

Two things surfaced by the 2026-07-08 diagnostics (3.11) are not "found
sound" -- they are gotchas that will silently mislead any future analysis
that forgets them:

- **Reconstruction infidelity**: a backtest run on reconstructed historical
  features does not speak for the live pipeline. Correlation between the
  live-logged `prob_over` and the same old model re-scored on the
  reconstructed frame is only 0.408 (mean absolute difference 0.060) for the
  same goalie-nights (3.11). Any future "does the live system have an edge"
  question must be answered with shadow-run/CLV data collected from the
  actual live pipeline, not by reconstructing historical features and
  backtesting a model on them.
- **Cross-source goalie/line matching needs care**: `betting.db` stores bare
  goalie last names (e.g. `Bobrovsky`) while the historical multibook odds
  archive parquet stores `F. Lastname`-style short names. Joining the two
  requires accent-stripped last-token matching plus team and a +/-1-day date
  tolerance (mirroring the UTC/local fix in 2.3), and even after matching,
  `betting.db`'s `confidence_pct` column is **not** `P(side)` -- for UNDER
  tiers use `1 - prob_over`.
