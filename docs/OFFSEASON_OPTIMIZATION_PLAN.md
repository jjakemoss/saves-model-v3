# Offseason Optimization Plan (2026)

Written 2026-07-07. This is the output of an offseason deep dive across the whole
system -- the live betting record in `data/betting.db`, the production model's
internals and calibration, the training data itself, and the Underdog parlay
question left open in [HISTORICAL_DATA_ANALYSIS.md](HISTORICAL_DATA_ANALYSIS.md).
Every number below was computed fresh from the repo's data during this analysis,
not carried over from earlier docs. Where a finding is statistically thin, it says
so.

The one-paragraph version: **the model has a real, live-verified edge on UNDER
recommendations (+32.6% ROI on 168 out-of-sample bets from the current model),
its OVER recommendations are a persistent money-loser that should be suppressed,
and the single highest-impact offseason project is fixing a newly root-caused
training-data corruption bug -- roughly 44% of recent training rows attach one
goalie's betting line to the other goalie's stats and outcome.** On the betting
side, UNDER-only Underdog parlays built from 65%+ confidence legs backtest
strongly positive under real Underdog multipliers, and an adaptive nightly
construction rule (3-leg when the night gives you 3+ qualifying legs, smaller
otherwise) is the recommended default for next season.

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

### 3.3 Select models honestly (this is why 3.1's results can be trusted)

Change `tune_hyperparameters.py` to rank candidates by **validation ROI only**,
touch the test fold exactly once for the final chosen config, and report both.
Better still, once 3+ seasons of labeled data exist, use walk-forward
validation (train on season 1, validate on 2; train on 1-2, validate on 3;
average) -- `CURRENT_HISTORICAL_DATA.md` section 6 already makes this case.
Also worth adding to the harness: a bootstrap CI on the backtest ROI, since a
~1,000-row test fold carries roughly +/-2pp of standard error on hit rate, and
single-number ROI comparisons inside that noise band are coin flips.

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
as books adjust. The part that survives -- the model's *selection* among unders
(69% hit vs a 51.6% blind-under base rate in the same window) -- is real skill,
but assume the blended edge regresses from the observed +32.6% toward something
like half that, and size accordingly.

### 3.8 Early-season handling

Rolling features run on `min_periods=1`: an October 12th prediction can ride on
2-3 games of data. Options, cheapest first: require a minimum of ~5 games
played before a goalie is bettable; seed early-season windows with the
goalie's previous-season averages; or add a `games_played_this_season` feature
and let the model learn its own caution. At minimum, expect October to be the
noisiest month and stake down accordingly.

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

### 4.3 Underdog vs. BetOnline (venue allocation)

Keep both, with defined jobs. Underdog parlays offer the highest EV per dollar
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

- Baseline: **2-3% of bankroll per ticket** (roughly quarter-Kelly for a 3-leg
  Power Play at 65% true legs), hard cap ~8% of bankroll exposed per night.
- Expect long losing streaks by construction: a +EV 3-leg ticket at 65% legs
  still loses 72.5% of the time. The bankroll math only works if a 10-ticket
  losing streak (a 4% probability event over any given 10-ticket stretch) is
  boring rather than ruinous.
- Scale stakes with the bankroll (re-anchor the percentage monthly), not with
  recent results.
- The $100 -> $1,800 run included real edge *and* a strongly favorable variance
  draw. Planning assumption for next season should be the regressed edge from
  section 3.7, not last season's realized multiple.

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

---

## 5. Prioritized offseason roadmap

| # | Project | Effort | Sections |
|---|---|---|---|
| 1 | ~~Fix multibook matching bug, dedupe, drop placeholder-odds rows, regenerate parquets~~ **done 2026-07-07** (see 2.3) | ~a day | 2.3 |
| 2 | Retrain + honest selection (val-only ranking, test touched once, bootstrap CIs) | 1-2 days | 3.1, 3.3 |
| 3 | Calibration layer on the retrained model | half day | 3.2 |
| 4 | UNDER-only policy + adaptive parlay rule in the daily workflow | half day | 4.1, 4.2 |
| 5 | `tickets` table + CLV capture in the tracker | ~a day | 4.5 |
| 6 | Market-anchor feature experiment (implied prob as feature) | 1-2 days | 3.4 |
| 7 | New context features (game total/moneyline, opponent rest, special teams, season normalization) | 2-3 days | 3.6 |
| 8 | Distributional saves model prototype, head-to-head vs classifier | ~a week | 3.5 |
| 9 | Check The Odds API historical archive pricing for pre-2024 props | an hour | -- |
| 10 | Trivial carryover: `TheOddsAPIFetcher.DEFAULT_BOOKMAKERS = []` fix (`src/betting/odds_fetcher.py:261`) | minutes | -- |

Items 1-5 are the "must happen before opening night" set. Items 6-8 are where
the next real accuracy gains live. If only one thing gets done this offseason,
it is item 1 -- every other number in this repo is built on that data.

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
  which loses to vig. The model's UNDER *selection* (69%) is the edge. The
  system is doing something real; the offseason work is about doing it on
  clean data with honest probabilities.
