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
saves model evaluated through the honest harness. As of 2026-07-08, the
platform pieces (tickets, CLV, snapshots, shadow-run logging) are implemented,
and the market-anchoring and distributional-model experiments (roadmap items 7
and 9) both came back clean negatives on a bettable edge -- though each
produced a real methodological gain (better calibration, coherent line pricing
across any line) worth carrying forward.

## Table of contents

1. [Honest state of the system](#1-honest-state-of-the-system)
2. [The critical new finding: corrupted training labels](#2-the-critical-new-finding-corrupted-training-labels)
3. [Model training improvements, ranked](#3-model-training-improvements-ranked)
4. [Betting strategy for next season](#4-betting-strategy-for-next-season)
5. [Prioritized offseason roadmap](#5-prioritized-offseason-roadmap)
6. [Appendix: what was checked and found sound](#6-appendix-what-was-checked-and-found-sound)

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
| 6 | ~~`tickets` table + CLV capture in the tracker -- unconditional; CLV is the real-time edge detector next season regardless of what items 3-5 conclude. Scope now explicitly includes: line/odds snapshots with fetch timestamps (not just the bet-time line), closing-line capture at puck drop, and a shadow-run log of the exact live system (`tuned_v1_20260201_155204`, live feature pipeline, every recommendation logged whether or not staked) so 3.11's open question can be settled by next season's data instead of another reconstruction.~~ **implemented 2026-07-08.** `line_snapshots`/`tickets`/`ticket_legs` tables plus an idempotent migration (`scripts/add_tracking_tables.py`); snapshot capture wired into `scripts/fetch_and_predict.py`; closing-line + CLV computation (`scripts/compute_closing_clv.py`, wired into the update-results workflow with a graceful no-op if the migration has not been run); phone-first ticket recording (`scripts/record_ticket.py` + `.github/workflows/record_ticket.yml`, `reason_code` required); a CLV report (`scripts/clv_report.py`); and `model_version` tagging on recommendations for shadow-run attribution. Two open operational notes: the migration has **not yet been applied** to the live `data/betting.db` (one command, a user decision, since the db is git-tracked), and a pre-puck-drop closing-fetch cron exists commented-out in `fetch_predictions.yml` pending a deliberate usage/cost decision. | ~a day | 4.5, 4.6, 3.11 |
| 7 | ~~Market-anchored residual experiment (implied probability, book, line movement, game total/moneyline) -- the retrain result strengthens the case: the model's huge unanchored disagreements with the market resolve at 49%~~ **done 2026-07-08 -- result: anchoring improves discrimination/calibration, no bettable edge; market-only model has no standalone signal (see 3.4).** | 1-2 days | 3.4, 3.9, 3.10, 3.11 |
| 8 | New hockey-context features (game total/moneyline, opponent rest, special teams, shot attempts/xG, starter/news timing, season normalization). **Case strengthened 2026-07-08**: the distributional experiment (3.5) showed the shots-against submodel is where the saves signal actually lives, and these features feed directly into a shots-against forecast. | 2-3 days | 3.6, 3.9, 3.10 |
| 9 | ~~Distributional saves model prototype, head-to-head vs classifier (trains on all 10,496 goalie-games, no odds required)~~ **done 2026-07-08 -- result: best-calibrated model yet (test Brier 0.25487), +1.06% test ROI with a cluster CI spanning zero -- no demonstrated edge; signal lives in the shots submodel (see 3.5).** | ~a week | 3.5, 3.10, 3.11 |
| 10 | Check The Odds API historical archive pricing for pre-2024 props | an hour | -- |
| 11 | Trivial carryover: `TheOddsAPIFetcher.DEFAULT_BOOKMAKERS = []` fix (`src/betting/odds_fetcher.py:261`) | minutes | -- |

Items 5-6 are the "must happen before opening night" set if any betting
continues: make the strategy decision consciously, then build tickets + CLV so
next season measures the thing actually being bet. Items 7-9 are where
model-side gains live, if any exist -- and none of them should be evaluated
except through the item-2 honest harness, preferably with full-date/game-level
fold boundaries so multibook rows from the same goalie-night cannot straddle
folds. If only one thing gets done next, it is item 6: the tracker must be
able to represent tickets and CLV before more real money produces another
ambiguous record. Do not respond to the item-2 result by running more
hyperparameter searches against the test fold -- that road leads straight back
to the retired +23.31%.

**Update 2026-07-08:** items 6, 7, and 9 are now done (see above); the
migration in item 6 still needs to be applied to the live database before any
of this generates real CLV/shadow-run data, and what remains on the model
side is item 8, whose case 3.5 just strengthened -- the in-season measurement
program (CLV capture, the shadow run) is now the load-bearing next step, not
another offline backtest.

## 6. Appendix: what was checked and found sound

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
