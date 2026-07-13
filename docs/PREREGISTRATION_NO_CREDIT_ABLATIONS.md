# Pre-Registration: No-Credit Ablations (BREAKTHROUGH_MODEL_PLAN.md section 6.2)

Status: PROPOSED. Written 2026-07-10, before any of the seven experiments
below has been run to completion under this document's rules. This is step 1
of `docs/BREAKTHROUGH_MODEL_PLAN.md` section 10. Step 0 (independently
verifying the section 2 diagnostics -- shots-bias magnitudes, the no-pace
ablation, the in-sample dispersion critique) has NOT been executed as of
this writing; every section-2 number cited below is marked unverified unless
this document itself independently re-derived it during preparation (marked
inline where that happened).

Update (2026-07-10, later the same day, before this document was locked):
step 0 has since completed and CONFIRMED all six section-2 claims by
reloading the frozen origin artifacts -- shots bias +1.9501/+1.8445
(claimed ~1.95/~1.85), saves bias +1.7863/+1.7511, save-rate bias small
(+0.0099/+0.0105), the no-pace control ablation reproduced at +0.4420/
+0.0308 (claimed ~0.44/~0.03), the 30.31/29.18/27.41 starter-SOG decline
confirmed from two independent sources while raw attempt volume rose then
flattened (58.5 -> 61.0 -> 59.8), and the in-sample dispersion fit confirmed
in code (`src/experiments/distributional_saves.py::fit_dispersion`) with the
consequence quantified: held-out residual variance ~2.3x the train-fitted
estimate on Origin A, and Origin B's frozen artifact collapsed to a
Poisson fallback (alpha=0.0) while true out-of-sample variance was ~2x the
Poisson mean (implied alpha ~0.027). One honest downgrade: the "control
still trailed the market" prose sub-claim is solid for Origin A (Brier delta
+0.00708, CI [+0.00320, +0.01103]) but statistically marginal for Origin B
(+0.00533, CI [-0.00009, +0.01079]). Consequence for Experiment 1: its
section 2.4 bar (control |bias| less than half of pace_shots's on both
origins) is already met by step 0's rerun (0.442 vs 1.950; 0.031 vs 1.844);
the concurrently-running funnel experiment's variant (a) serves as an
independent reproduction rather than the first measurement.

## 0. Preamble

### 0.1 What this document binds, and what it does not

The 2023-24 and 2024-25 seasons have already been used as outcome-viewed
test folds in `models/trained/experiment_rolling_origin_20260709_222639/`
(BREAKTHROUGH_MODEL_PLAN.md section 6.1: "The 2023-24 and 2024-25 outcomes
have now been viewed in the rolling-origin experiment. New modeling against
them is development, even when the new input data has not been seen
before."). Consequently this pre-registration cannot claim the classic
double-blind property of "the author has never seen this data's outcome."
Its binding force is not data privacy -- it is procedural discipline:

- **Locks.** Any threshold, calibration method, or shape-translation choice
  that a later single-touch test depends on must be fixed in code/config
  before that test is run, and not revised afterward based on the test's
  own result.
- **Single-touch tests.** Each designated test fold (Origin A 2023-24,
  Origin B 2024-25, and Component G's 2024-25 closing pass) is touched
  exactly once per experiment after its threshold/method is locked. Running
  it, not liking the number, and re-running with a different choice is a
  protocol violation, not a second attempt.
- **Explicit pass/fail definitions**, fixed here, before any of the numbers
  they will be compared against exist for this round.

Where the plan (section 6.1) ranks evidence -- 2023-24 improvement is
hypothesis support only, locked 2024-25 is stronger chronological evidence,
a frozen 2026-27 shadow run is final confirmation -- every experiment below
inherits that hierarchy explicitly in its own section 6.

### 0.2 Concurrent implementation -- an honesty note

Several of the seven experiments below are being implemented concurrently
with this document by parallel agents working in the same tree. This means
the classic pre-registration guarantee ("the standard was written before the
work began") does not hold uniformly across all seven items for this round.
Practical consequence, stated plainly rather than glossed over: for this
round, this document functions as a **registered evaluation standard** --
a fixed yardstick that any concurrently-produced result must be measured
against after the fact -- rather than a strict temporal pre-registration for
every item. What *is* still enforceable, and is the part that actually
protects against self-deception on a real-money system:

- Any threshold, calibration method, or shape-translation choice must still
  be shown to have been fixed independent of the specific test-fold number
  it will be judged against (i.e., derived from training/validation data or
  from the 2023-24 development pass only, never from the 2024-25/test
  result it is meant to gate).
- Any future **confirmatory touch** -- most importantly Component G's
  2024-25 closing-pass test touch (Experiment 7), and anything whose result
  would feed a Gate B or Gate C purchase decision -- must happen strictly
  **after** this document is locked (i.e., after this file is committed and
  not further edited). A concurrently-running agent that has already
  produced a 2023-24 development number before this document existed is not
  disqualified; a concurrently-running agent that touches a designated
  single-touch test fold before this document is locked has invalidated
  that touch for pre-registration purposes and must re-run it, or the
  result must be reported as post-hoc, not confirmatory.

### 0.3 Section 1a operational constraints, restated as a binding deployment filter

Every experiment below is judged first on its own statistical merits
(chronological validity, calibration, cluster-CI honesty), but a
statistically real result does not automatically license anything. Per
plan section 1a, the following filter is binding on top of every gate
below, independent of the statistics:

1. **One decision window.** The user has one reliable early-evening
   decision point per day (matches the 22:30Z / 5:30pm ET / 4:30pm CT
   pipeline anchor). No experiment's pass condition may require intraday
   monitoring or a second daily touch to be actionable; an experiment that
   only works pre-22:30Z is research-only for deployment purposes even if
   its statistics pass.
2. **Three venues.** Underdog Fantasy and PrizePicks (fixed-multiplier
   parlay apps -- line selection and ticket construction only, no price
   lever) plus BetOnline (offshore, straight bets). DraftKings, FanDuel,
   Caesars, BetMGM, Bovada, and Pinnacle-equivalent books in the historical
   data are consensus/research inputs, not executable venues. Any result
   whose edge is concentrated in an inaccessible book (the section 3.4
   BetOnline-closing-outlier caveat is the concrete instance already on
   record) must be reported as real-but-not-deployable, not folded into a
   headline pass.

This filter matters most for Experiment 7 (Component G), whose deployed
form is explicitly the venue-relative filter of section 1a, but it applies
to the whole model architecture: a combined model that only beats the
market with cross-book price-shopping access the user does not have is not
a strategy for this project (plan section 11).

### 0.4 Numbers used below and their verification status

Every historical number quoted in this document was checked against a file
that exists on disk as of 2026-07-10 (paths listed per experiment). Two
classes of number need separate treatment:

- **Independently re-verified while writing this document** (safe to treat
  as ground truth): the `TOI < 50` early-exit count (572/10,496 = 5.45%,
  recomputed directly from `clean_training_data.parquet`'s `toi` column,
  parsed from `"MM:SS"` strings -- matches plan section 3.2 exactly); the
  pace_shots feature-family column counts (`opponent_offense_pace`=12,
  `team_shot_suppression`=12, `combined_pace`=5, `special_teams_volume`=8,
  `league_relative_zscores`=4, summing to exactly 41 -- matches plan
  section 2's "41 MoneyPuck pace features" claim); the rolling-origin fold
  boundaries and paired Brier deltas (pulled directly from
  `models/trained/experiment_rolling_origin_20260709_222639/metadata.json`,
  quoted verbatim below).
- **Still unverified pending step 0** (plan section 2's shots-bias
  magnitudes: predicted shots against ~1.95/1.85 too high on Origin
  A/B, saves ~1.79/1.75 too high, the no-pace control's ~+0.44/+0.03
  reduced bias, and the "dispersion fitted on in-sample training residuals"
  characterization of the ORIGIN-carved models specifically). These are
  Codex-reported read-only diagnostics. Experiment 1 below is partly step 0
  itself: it re-derives the no-pace-control bias number on the fresh
  origins as a first-class deliverable, not just an assumption.

## 1. Shared conventions (referenced by all seven experiments)

**Harness code reused, not reimplemented** (per this repo's convention of
importing rather than duplicating distributional/betting math):
`src/experiments/harness.py` (`split_by_date`, `decide_bet`, `grade_bets`,
`betting_metrics_bundle`, `cluster_bootstrap_roi_ci`,
`evaluate_threshold_sweep`), `src/experiments/distributional_saves.py`
(`SavesDistribution`, `fit_dispersion`, `train_shots_model`,
`train_save_rate_model`, `compute_distribution_predictions`,
`intrinsic_quality_metrics`, `join_and_price`), `scripts/
experiment_pace_distributional.py` (`VARIANTS`, `feature_cols_for_variant`,
`load_pace_modeling_frame`), `scripts/experiment_rolling_origin.py`
(`carve_origin_split`, `season_date_range`, `paired_brier_delta`), `scripts/
clv_audit_pace_policy.py` (`cluster_bootstrap_mean_ci`, `clean_bettime_pass`,
`clean_closing_pass`, `attach_game_id`, `pivot_both_sides`).

**Chronological folds -- confirmed against `scripts/experiment_rolling_
origin.py` and `models/trained/experiment_rolling_origin_20260709_222639/
metadata.json`, not assumed from the plan text:**

| | Origin A | Origin B |
|---|---|---|
| Train | 2022-10-07 to 2023-02-24 (1,864 rows) | 2022-10-07 to 2024-02-29 (4,528 rows) |
| Val (last 49 days of training pool) | 2023-02-25 to 2023-04-14 (760 rows) | 2024-03-01 to 2024-04-18 (720 rows) |
| Test | season 2023-24: 2023-10-10 to 2024-04-18 (2,624 clean-data rows) | season 2024-25: 2024-10-04 to 2025-04-17 (2,624 clean-data rows) |
| Test betting frame | `data/processed/multibook_frame_2023_24.parquet` (closing, 8,880 rows / 2,298 goalie-nights) + `..._bettime.parquet` (7,660 rows, secondary) | `multibook_classification_training_data.parquet` filtered to `season==20242025` (7,463 rows, closing only -- no usable bettime pass exists for 2024-25, see 1.2 below) |
| De-vigged market Brier (closing, already measured) | 0.25000 (n=8,880) | 0.24881 (n=7,463) |

Val is carved from inside each origin's training-pool seasons (never inside
the test season) specifically so the test season is touched exactly once.
This is a genuinely different split mechanic from the production/worn fold
(`src/experiments/harness.py`: train `<2025-10-16`, val `2025-10-16..
2025-12-03`, test `>=2025-12-04`) -- that fold is worn (BREAKTHROUGH_MODEL_
PLAN.md section 6.1) and must not be re-used as a primary test fold for any
experiment below; it may only appear as a secondary/diagnostic reference
(e.g., the existing `control` variant's 0.25487 Brier / +1.06% ROI / 888
bets already recorded in `experiment_pace_distributional_20260709_100802/
metadata.json` is a worn-fold number and is cited below only as prior
context, never as the pass/fail bar).

**EV threshold.** Fixed at 0.05 (the frozen production `pace_shots`
threshold) for every origin-carved experiment's ROI reporting, never
reselected via a validation sweep against Origin A or B -- no betting-line
odds exist for any date before 2023-05-03, so Origin A's validation window
(inside `<=2022-23`) has no market data to sweep against, and the same fixed
threshold is applied to Origin B so both origins are evaluated identically.
This differs from the worn-fold harness's `[0.05, 0.10, 0.12, 0.15]`
validation sweep -- do not import that sweep into any Origin A/B
experiment below.

**Selection at inference/pricing time.** `decide_bet`/`calculate_ev` compare
the model's probability to each book's RAW (vig-inclusive) single-side
implied probability, not a de-vigged one -- confirmed from
`scripts/clv_audit_pace_policy.py`'s own docstring and code. De-vigging is
used only for the paired-Brier and CLV *metrics* computed on top of already-
selected bets, via `betting.tracking_db.devig_prob` (additive/proportional
normalization on the American-odds pair). This is a different de-vig
convention from `scripts/build_market_game_features.py`'s per-book
multiplicative normalization of h2h/totals pairs -- the two are not
interchangeable and Experiment 5 must not conflate them.

**PMF cap.** `ORIGIN_CAP = 90` for all origin-carved distributional
predictions (the production `CAP = 70` undercounts PMF mass for at least
one high-shots Origin A test-fold goalie-night and is reserved for exact
frozen-artifact reproduction only).

**Cluster bootstrap.** Goalie-night cluster (`f"{game_id}_{goalie_id}"`),
10,000 resamples, seed 42, 95% CI, via `cluster_bootstrap_roi_ci` (ROI) or
`cluster_bootstrap_mean_ci` (any other paired statistic, e.g. Brier delta or
probability CLV). Book-rows from the same goalie-night are not independent
observations; row-level (non-clustered) CIs are diagnostic only.

**1.1 Data files verified present on disk (2026-07-10), with shape/date
range as loaded:**

| Path | Rows | Date range | Role |
|---|---:|---|---|
| `data/processed/clean_training_data.parquet` | 10,496 (114 cols) | 2022-10-07 to 2026-04-16, 2,624/season | Base goalie-game rolling features + `toi`, target = `saves` |
| `data/processed/game_context_features.parquet` | matches clean rows | same | Schedule/rest context, no market data |
| `data/processed/pace_features.parquet` + `pace_features_metadata.json` | 10,496 | 2022-10-07 to 2026-04-16 | 45 cols, 6 families (see 1.2) |
| `data/processed/multibook_classification_training_data.parquet` | -- | -- | Production multibook training data, all seasons |
| `data/processed/multibook_frame_2023_24.parquet` / `..._bettime.parquet` | 8,880 / 7,660 | 2023-11-02 to 2024-04-18 | Origin A test betting frame (built by `experiment_rolling_origin.py`, closing/bettime) |
| `data/processed/saves_lines_snapshots.parquet` | 79,884 | 2023-11-02 to 2026-04-16 | Per-book goalie saves quotes, bettime+closing passes, feeds Experiment 7 |
| `data/processed/market_game_features.parquet` | 305,940 | 2023-10-10 to 2026-04-19 | Per-book h2h/totals quotes, feeds Experiment 5 |
| `models/trained/experiment_rolling_origin_20260709_222639/` | -- | -- | Origin A/B fold boundaries, control/pace_shots reference numbers |
| `models/trained/experiment_pace_distributional_20260709_100802/` | -- | -- | Frozen production `pace_shots` artifact (worn fold; reload target for step 0, not a test fold for this document) |

**1.2 A genuine data-coverage gap, flagged up front because it affects
Experiments 4 and 7:** `saves_lines_snapshots.parquet`'s per-season/pass
row counts are 2023-24 bettime=15,682/closing=17,959, **2024-25
bettime=258/closing=14,954**, 2025-26 bettime=12,811/closing=18,220. 2024-25
effectively has no bettime pass (258 rows vs. ~15,000 for the other two
seasons -- this matches the plan's own statement in section 3.4 item 1 that
"2024-25 has no bettime pass"). Any experiment whose 2024-25 touch would
need a bettime-to-close comparison cannot get one this round; see
Experiments 4 and 7 for how each handles it.

---

## 2. Experiment 1 -- No-pace distributional control on fresh origins

**2.1 Hypothesis and failure mechanism targeted.** Plan section 2 diagnoses
the `pace_shots` recipe's failure as raw pace/Corsi features being allowed
to set the absolute shots-on-goal level without season-aware conversion,
producing a ~+1.95/+1.85-shot bias on Origin A/B (unverified). The
competing hypothesis this experiment is designed to rule out is that the
bias is a generic property of the modeling recipe (small training pool,
XGBoost count-model overfitting, choice of validation-only selection) and
has nothing specifically to do with the 41 pace columns. If a model built
from the exact same recipe minus all pace/context features shows a
materially smaller or absent bias on the same two fresh origins, that
isolates the pace features as the mechanism, consistent with the plan's
diagnosis and with Gate A's requirement to "fix the demonstrated mechanism
rather than merely move ROI." If the no-pace control shows comparably large
bias, the section 2 diagnosis is wrong and Gate A's premise needs revisiting
before Experiment 2 proceeds.

**2.2 Data inputs and folds.** `data/processed/clean_training_data.parquet`
+ `data/processed/game_context_features.parquet` only -- `pace_features.
parquet` is loaded (the shared frame-loading function requires it) but its
columns are excluded from both the shots and save-rate feature lists, using
`experiment_pace_distributional.py`'s existing `VARIANTS[0]` (`"control"`:
`shots_use_context=False, shots_use_pace=False, rate_use_goalie_workload=
False`). This variant already exists in code; what does not yet exist is
running it through `experiment_rolling_origin.py`'s Origin A/B carving,
which currently hardcodes the `pace_shots` variant. Folds: Origin A and
Origin B exactly as specified in section 1 above. Betting frames:
`multibook_frame_2023_24.parquet` (+ bettime, secondary) for Origin A;
`multibook_classification_training_data.parquet` filtered to
`season==20242025` for Origin B.

**2.3 Metrics.** PRIMARY: (a) signed mean shots bias on the test fold,
`mean(mu_pred - shots_against_actual)` -- this metric does not yet exist in
`src/experiments/distributional_saves.py` (which only reports MAE) and must
be added as a small, non-controversial addition before this experiment runs;
(b) paired Brier delta vs. the de-vigged market with goalie-night cluster
95% CI (`experiment_rolling_origin.paired_brier_delta`, already implemented
and exercised). SECONDARY: policy ROI with cluster CI at the fixed 0.05
threshold; central 50%/80% coverage and PIT histogram from
`intrinsic_quality_metrics` (on val, reused diagnostically on test);
shots-against MAE vs. the naive `shots_against_rolling_5` baseline.

**2.4 Pass/fail.** This experiment does not itself pass or fail against
Gate A -- it establishes the verified baseline that Experiments 2, 3, and 6
must beat. Its own falsifiable criterion, tied to Gate A's first bullet
("removes the persistent positive shots bias"): the no-pace control's
`|signed bias|` on both Origin A and Origin B test folds must be
substantially smaller than `pace_shots`'s own bias on the same origins
(operational bar: control bias magnitude less than half of `pace_shots`'s,
on both origins -- `pace_shots`'s bias is itself unverified pending this
same run, so both numbers come out of this experiment together). If that
bar is not cleared on both origins, record it as a clean negative for the
section 2 diagnosis and stop the Gate A track pending re-diagnosis, rather
than continuing to Experiment 2 on an unconfirmed premise.

**2.5 Forbidden.** Do not tune the control variant's hyperparameter grid
differently from `pace_shots`'s (both use `SHOTS_CONFIGS`/`SAVE_RATE_
CONFIGS` unchanged, selected on val MAE / weighted log-loss only). Do not
report the existing worn-fold control numbers (Brier 0.25487, ROI +1.06%,
888 bets) as if they were this experiment's result -- they are a different
fold and are cited in section 0.4/1 as prior context only. Do not select
the EV threshold from a sweep; it is fixed at 0.05 per section 1.

**2.6 Interpretation hierarchy.** A materially reduced bias on 2023-24
alone is hypothesis support only. Both origins agreeing is the strongest
evidence this no-purchase program can produce for the pace-feature
diagnosis (2024-25 is itself only a "locked" fold in the weak sense that no
threshold is being selected against it here -- it is a fixed recipe applied
identically to both origins, not a tuned one). Final confirmation, per
section 6.1, is the 2026-27 frozen shadow run, not available this round.

---

## 3. Experiment 2 -- Season-normalized pace and explicit attempt-to-SOG funnel

**3.1 Hypothesis and failure mechanism targeted.** Directly targets plan
section 2's stated conclusion: "raw pace features cannot be allowed to
determine the absolute SOG level without season-aware conversion and
calibration," evidenced by starter shots against falling 30.31 (2022-23)
-> 29.18 (2023-24) -> 27.41 (2024-25) while Corsi inputs did not fall in
parallel (unverified pending step 0, but structurally plausible given the
league-wide shot-suppression trend this project has documented elsewhere).
Two candidate fixes, per the plan's required-ablations table (section 4.2):
(a) season-normalized pace -- use the pre-built `league_relative_zscores`
family (4 columns: `opp_off_all_corsi_ema5_prior_league_z`, `team_def_all_
corsi_against_ema5_prior_league_z`, `combined_all_corsi_ema5_prior_league_
z`, `combined_all_xg_ema5_prior_league_z`) in place of, or alongside, the
raw-count pace families; (b) an explicit attempt-to-SOG funnel -- a new
engineered stage converting Corsi/Fenwick attempt volume into an explicit
shots-on-goal-per-60 estimate scaled to projected exposure minutes, which
does not exist anywhere in this codebase yet and must be built as part of
this experiment.

**3.2 Data inputs and folds.** `clean_training_data.parquet`, `game_
context_features.parquet`, `pace_features.parquet` + `pace_features_
metadata.json` (specifically the `family_columns` dict, confirmed to
contain `opponent_offense_pace` (12), `team_shot_suppression` (12),
`combined_pace` (5), `special_teams_volume` (8), `goalie_workload_quality`
(4), `league_relative_zscores` (4) -- 41 columns feed the shots side in the
existing `pace_shots` variant (families 1-4 and 6), matching section 2's
"41 MoneyPuck pace features" claim exactly, independently confirmed while
writing this document). Same Origin A/B folds and betting frames as
Experiment 1.

**3.3 Metrics.** Same PRIMARY/SECONDARY set as Experiment 1 (signed shots
bias + paired Brier delta primary; ROI, coverage/PIT, shots MAE secondary),
computed separately for each of the four required ablation variants: (i)
no-pace control (= Experiment 1's result, reused, not re-run), (ii) raw
`pace_shots` (reproduces the known failure -- already available from
`experiment_rolling_origin_20260709_222639/metadata.json`, reused, not
re-run), (iii) season-normalized pace, (iv) explicit attempt-to-SOG funnel.

**3.4 Pass/fail, tied to Gate A.** Gate A's first two bullets apply
directly: for at least one of variants (iii)/(iv) to justify inclusion in
the combined model, on BOTH Origin A and Origin B: (a) signed shots bias
must fall to within a small band of zero -- operational bar: `|bias| <
0.5` shots, or statistically indistinguishable from Experiment 1's no-pace-
control bias (overlapping cluster CIs), whichever is the more informative
comparison once Experiment 1's numbers exist; (b) the paired Brier delta
vs. market must be lower (more favorable) than Experiment 1's no-pace-
control delta on both origins, i.e. season-normalization must recover more
than it costs relative to having no pace signal at all. Meeting (a) without
(b), or vice versa, is a partial result and should be reported as such, not
rounded up to a pass.

**3.5 Forbidden.** Do not select between variants (iii) and (iv) by peeking
at Origin A or B test-fold ROI/Brier and keeping whichever wins -- both are
evaluated and reported; if only one is to be carried forward into later
experiments (3, 4, 6, 7), that choice must be justified on the PRIMARY
metrics (bias, paired Brier delta) using the SAME reasoning that would be
written down before seeing the numbers (i.e., prefer whichever produces
smaller bias with a tighter CI, not whichever produces better ROI). Do not
build the attempt-to-SOG funnel using any information from a date after the
funnel's own training cutoff (funnel stage-wise rates must themselves be
prior-only, matching the `.shift(1)` convention used everywhere else in
this codebase).

**3.6 Interpretation hierarchy.** As Experiment 1. Note explicitly the
named failure disposition from plan section 9 ("Season correction improves
Brier but not CLV" -- disposition: the model is becoming more honest but
has not found a trading edge, continue shadow-only) as a legitimate,
non-catastrophic outcome for this experiment specifically.

---

## 4. Experiment 3 -- Validation-fitted dispersion

**4.1 Hypothesis and failure mechanism targeted.** Plan section 2: "Negative
-binomial dispersion was fitted on in-sample training residuals. That made
the distributions too narrow, particularly when a flexible shots model
reduced its own training residuals." Confirmed by reading `fit_dispersion`
in `src/experiments/distributional_saves.py` directly: it fits `alpha` from
`(y_train - mu_train)` residuals on `train_idx`, unconditionally, for every
variant in every experiment that has run in this repo so far, including the
Origin A/B `pace_shots` runs. This experiment tests whether fitting `alpha`
from held-out (validation-fold) residuals instead produces materially
better-calibrated tail coverage without collapsing the distribution's
useful width.

**4.2 Data inputs and folds.** Same as Experiment 1/2. This is a change to
`fit_dispersion`'s call site (or a new `fit_dispersion_val` variant using
the same closed-form NB2 moment-matching math), applied on top of whichever
shots/save-rate model Experiments 1-2 have already selected -- it is a
dispersion-estimation change, not a new feature set, and should be layered
on the Gate-A-candidate architecture from Experiment 2, not run as an
isolated fourth model.

**4.3 Metrics.** PRIMARY: central 50%/80% coverage and PIT-histogram
uniformity from `intrinsic_quality_metrics`, computed on the TEST fold (not
just val, where this function is normally used) for both the train-fitted
and val-fitted dispersion, on both origins. SECONDARY: paired Brier delta
(dispersion mostly reshapes the tails, so its effect on the OVER/UNDER-
threshold-crossing Brier may be small); policy bet rate at the fixed 0.05
threshold, as the operational proxy for "extreme edge inflation."

**4.4 Pass/fail, tied to Gate A.** Gate A's fourth bullet: "Uses validation
-fitted dispersion without extreme edge inflation." Operationalized here
(the plan gives no numeric bar, so this is a Claude-authored construction,
flagged as such): (a) summed absolute coverage deviation from nominal,
`|cov50 - 50| + |cov80 - 80|`, must be no worse under val-fitted dispersion
than under train-fitted dispersion, on the TEST fold, on both origins; (b)
"no extreme edge inflation" -- the test-fold bet rate at the fixed 0.05
threshold under val-fitted dispersion must not blow out relative to train-
fitted (operational bar: stays within the historically observed ~15-45%
bet-rate range this repo's experiments have shown at this threshold,
rather than jumping past it, which would indicate the distribution has
become pathologically narrow and is manufacturing apparent edge from
noise rather than genuine information).

**4.5 Forbidden.** The switch from train-fitted to val-fitted dispersion is
a pre-registered architectural decision applied uniformly to both origins
-- it must NOT be selected per-origin by checking which produces the better
test-fold ROI or Brier and keeping that one. If val-fitted dispersion helps
Origin A but hurts Origin B (or vice versa) on the PRIMARY coverage metric,
report both honestly; do not average away or discard the unfavorable
origin.

**4.6 Interpretation hierarchy.** As Experiment 1. This experiment is a
calibration-quality check more than an edge claim -- a coverage improvement
with no ROI change is still a pass on Gate A's own terms (Gate A is
explicit that "it does not need to beat the market yet").

---

## 5. Experiment 4 -- First-30-days shots-level correction, frozen for rest of season

**5.1 Hypothesis and failure mechanism targeted.** Plan section 3.3: a
post-hoc Platt calibration fit on the first 30 days of each test season and
evaluated on the remainder showed both intervals crossing zero but shifted
toward the market (2023-24 remainder: calibrated Brier 0.24859 vs. market
0.24995, delta CI [-0.00449, +0.00072]; 2024-25 remainder: calibrated
0.24621 vs. market 0.24860, delta CI [-0.00482, +0.00107]) -- evidence of
weak ranking signal hidden under a large seasonal level error, not proof of
an edge, and the plan is explicit that "the method was chosen after
inspecting the failed origins." This experiment targets the SAME failure
mechanism as Experiment 2 (seasonal drift in the shots-to-goals relationship,
per the 30.31/29.18/27.41 starter-shots trend) but attacks it with an
in-season burn-in recalibration rather than a structural feature change, and
critically, PRE-REGISTERS the calibration method here so it is not chosen
by looking at which one wins on the remainder fold, closing the exact gap
the plan itself flags in section 3.3.

**5.2 Data inputs and folds -- with a coverage gap that must be resolved
before this experiment runs.** `clean_training_data.parquet`'s season date
ranges are exact: 2023-24 = 2023-10-10 to 2024-04-18; 2024-25 = 2024-10-04
to 2025-04-17. But the betting-line frames do not start on the same day:
`multibook_frame_2023_24.parquet` (built by `experiment_rolling_origin.py`
from `saves_lines_snapshots.parquet`) starts **2023-11-02**, not 2023-10-10
-- a 23-day gap with no matched betting-line rows at the start of the
Origin A test season. `multibook_classification_training_data.parquet`'s
2024-25 rows start 2024-10-04, matching the season open exactly -- no gap
for Origin B. Consequence: if "first 30 days" is defined by calendar days
from the clean-data season start (matching plan section 3.3's original
convention), Origin A's calibration-fitting window (2023-10-10 to
2023-11-08) would have market-joined rows only from 2023-11-02 onward --
effectively a 7-day fit, not 30. This experiment must resolve that before
running, and the resolution must be locked (not chosen by which produces a
better result): the recommended fix, stated here as the pre-registered
choice, is to define "first 30 days" as the first 30 days of BETTING-LINE
coverage for that origin's test season, not the first 30 days of the
season by game date -- i.e., 2023-11-02 to 2023-12-01 for Origin A,
2024-10-04 to 2024-11-02 for Origin B. This keeps the two origins
comparable in fitting-sample size even though it makes their calendar
windows non-parallel; that asymmetry must be reported, not hidden.

**5.3 Metrics.** PRIMARY: paired Brier delta vs. market (cluster CI) on the
remainder-of-season fold (post-burn-in), for both origins, compared against
the SAME model's uncorrected delta on the identical remainder window.
SECONDARY: OVER/UNDER calibration separately on the remainder fold; policy
ROI with cluster CI at the fixed 0.05 threshold.

**5.4 Pass/fail.** The calibration METHOD (recommended: Platt/logistic
recalibration of the model's output probability, matching section 3.3's
precedent, applied identically to both origins) is locked in code before
either origin's remainder fold is touched. Pass, at the "hypothesis
support" tier consistent with section 6.1: on BOTH origins, the frozen
correction's remainder-fold paired Brier delta point estimate must be lower
(more favorable to the model) than the SAME uncorrected model's delta on
the identical remainder window. A full pass requires this to hold with the
cluster CI's upper bound below the uncorrected model's CI upper bound (i.e.
a real, not just point-estimate, improvement); partial credit (point
estimate improves, CI still overlaps the uncorrected result) is reported
honestly as inconclusive, matching section 3.3's own "both intervals cross
zero" framing rather than being rounded up.

**5.5 Forbidden.** Do not choose the calibration functional form (Platt vs.
isotonic vs. a simple additive shift) by fitting several and keeping
whichever improves the remainder fold most -- lock one method before either
origin's remainder fold is scored. Do not silently use the calendar-day
convention for one origin and the coverage-day convention for the other
without reporting the asymmetry described in 5.2. Do not extend the burn-in
window past 30 days of whichever convention is locked, even if a longer
window looks like it would help, without re-registering that change first.

**5.6 Interpretation hierarchy.** As Experiment 1, with the added caveat
that even a full pass on both origins is evidence about a burn-in
RECALIBRATION strategy, not evidence that the underlying shots/rate model
architecture (Experiments 2/3) is sound -- per section 9's named
disposition "Season correction improves Brier but not CLV: the model is
becoming more honest but has not found a trading edge. Continue
shadow-only."

---

## 6. Experiment 5 -- Existing game-total and moneyline features (Component C)

**6.1 Hypothesis and failure mechanism targeted.** Plan section 4.3:
consensus game total, de-vigged home/away win probabilities, and
approximate opponent expected goals (from total + moneyline) should enter
the exposure and workload components, potentially explaining some of the
shots-bias/exposure-variance the pace-only recipe misses (a different game
state -- e.g., an expected blowout vs. a close, high-pace game -- plausibly
shifts both shot volume and goalie workload independent of rolling Corsi
averages). The plan is explicit this is a modest-expectation test: "A simple
exploratory linear test found only marginal shots-MAE improvement" is
already on record informally; this experiment re-runs it properly, inside
the same origin-carved harness as the other six experiments, rather than as
an ad hoc linear probe.

**6.2 Data inputs and folds.** `data/processed/market_game_features.parquet`
(305,940 rows, per-book h2h/totals quotes, 2023-10-10 to 2026-04-19,
already de-vigged per-book via multiplicative normalization, flagged with
`is_latest_pregame_snapshot` for the timing-safe last-pregame-22:30Z view).
This file has NO existing cross-book consensus aggregation -- it only
de-vigs within a single book+market(+point) group. Building a single
per-game consensus total/moneyline (e.g., a simple mean or leave-one-out
mean of de-vigged per-book probabilities, computed only from
`is_latest_pregame_snapshot == True` rows) is new preprocessing this
experiment must add; the choice of aggregation formula is an ordinary
development decision (made on training-visible data, not against a test
outcome) and does not itself need to be locked, but once chosen it applies
identically to both origins. Join key: `(home_abbrev, away_abbrev,
game_date_eastern)` or the resolved `event_id`-to-`game_id` mapping already
used by `saves_lines_snapshots.parquet`'s `attach_game_id` convention.
Same Origin A/B folds as Experiment 1.

**6.3 Metrics.** PRIMARY: shots-against MAE (vs. the Experiment 2 Gate-A
candidate, with and without the market-game features added) and paired
Brier delta vs. market, both with the market-derived columns folded into
the exposure/workload feature set as section 4.3 specifies (not as a
standalone model). SECONDARY: exposure Brier/log loss if the market
features are also fed to Experiment 6's exposure sub-model; policy ROI.

**6.4 Pass/fail.** Not tied to a Gate A bullet directly (Component C is not
named in Gate A's four criteria), but tied to section 6.2's closing
instruction: "Each component must justify its inclusion through
distributional metrics before it can enter the combined model." Pass bar:
a statistically meaningful (cluster-bootstrap CI excluding zero) reduction
in shots MAE or improvement in paired Brier delta on BOTH origins relative
to the then-current best architecture from Experiments 2-3. A result that
replicates the prior informal finding -- marginal, CI spanning zero -- is a
valid negative: it does not block Gate A (Component C was never one of
Gate A's four bullets) and simply means these features are not carried
into the combined model, or are retained only as an input to Experiment 6's
exposure component (per section 4.1's "or a simpler historical game-state
baseline if that derivation is unstable" fallback) rather than the shots
funnel.

**6.5 Forbidden.** Do not treat any per-book market_game_features row as a
target-season-average signal -- always use the `is_latest_pregame_snapshot`
timing-safe view, never a raw mean across all snapshot dates for an event
(that would leak information from snapshots taken after the true decision
point). Do not present a marginal, non-significant shots-MAE change as "an
edge" (plan section 4.3's own words: "These features should enter the
exposure and workload components. They should not be treated as proof of
edge by themselves.").

**6.6 Interpretation hierarchy.** As Experiment 1.

---

## 7. Experiment 6 -- Exposure-state mixture (Component A)

**7.1 Hypothesis and failure mechanism targeted.** Plan section 3.2: 572 of
10,496 starts (5.45%, independently re-verified for this document directly
from `clean_training_data.parquet`'s `toi` column, parsed from `"MM:SS"`
strings) had goalie TOI below 50 minutes -- early replacements/injuries,
un-labelable by cause with current data. Removing those games after the
fact materially improved failed OVER results (2023-24 OVER ROI -10.58% ->
-4.11% excluding TOI<50; 2024-25 -8.40% -> -3.49%), which is not an
actionable filter (TOI<50 is not known in advance) but demonstrates that a
single negative-binomial count process misrepresents the lower tail. A
quick exploratory classifier for `TOI < 50` only reached AUC 0.53-0.56 --
individual early exits are weakly predictable at best, but per section 3.2,
"a calibrated pooled exposure mixture can still improve the distribution"
even without strong per-event discrimination. This experiment targets the
lower-tail miscalibration mechanism directly, and is explicitly NOT
targeting improved individual early-exit prediction (which the plan already
concedes is close to a ceiling).

**7.2 Data inputs and folds.** `clean_training_data.parquet` (`toi` parsed
to minutes; binary label `toi_minutes < 50`), `pace_features.parquet`'s
`goalie_workload_quality` family (4 columns: `goalie_xg_per_shot_roll10`,
`goalie_xg_per_shot_ema5`, `goalie_high_danger_share_ema5`, `goalie_
rebound_rate_ema5`) as candidate exposure-risk features, optionally
Experiment 5's consensus moneyline/total for the OT-probability piece (per
section 4.1: "Overtime probability derived from de-vigged game moneyline
and total inputs, or a simpler historical game-state baseline if that
derivation is unstable"). Only pregame-safe features -- current-game goals,
saves, shots, final starter status, or postgame lineup information are
forbidden inputs, matching `FORBIDDEN_FEATURE_COLS` already enforced
elsewhere in `distributional_saves.py`. Same Origin A/B folds as
Experiment 1.

**7.3 Metrics.** PRIMARY: exposure Brier score and log loss for the
calibrated `P(TOI < 50)` sub-model (test fold, both origins); lower-tail
calibration -- central 50%/80% coverage and PIT histogram from
`intrinsic_quality_metrics`, computed specifically on the TOI<50 subset of
each origin's test fold (roughly 5-6% of each origin's ~2,624-row test
season, on the order of 140-150 goalie-nights per origin -- small enough
that the cluster CI on this slice will be wide, and that must be reported,
not hidden). SECONDARY: exposure-classifier AUC (reported against the
plan's own 0.53-0.56 ceiling, not expected to exceed it materially);
overall (not just lower-tail) paired Brier delta and policy ROI, to confirm
the mixture does not degrade the non-early-exit majority of starts while
fixing the tail.

**7.4 Pass/fail, tied to Gate A.** Gate A's third bullet: "Produces better
lower-tail calibration." Pass bar: on the TOI<50 subset of BOTH origins'
test folds, the exposure-mixture model's summed coverage deviation from
nominal (`|cov50-50| + |cov80-80|`, same construction as Experiment 3) must
be smaller than the single-NB2 baseline's (Experiment 2's Gate-A candidate,
without the mixture) on the same subset. The exposure sub-model's AUC is
NOT part of the pass bar -- per section 3.2/9, individual early-exit skill
is not being claimed here, only pooled calibration improvement ("do not
claim individualized pull/injury skill").

**7.5 Forbidden.** Do not report exposure-classifier AUC improvements over
0.53-0.56 as if they were the point of this experiment -- the plan's own
risk-disposition (section 9, "Exposure risk is not predictable") says to
retain a pooled calibrated mixture only if it improves lower-tail scoring,
explicitly not to claim individualized skill. Do not use any postgame
signal (actual TOI, actual saves, decision) as a feature at prediction
time -- only pregame-known information may enter the exposure state
mixture.

**7.6 Interpretation hierarchy.** As Experiment 1, with the small-subsample
caveat from 7.3 stated in any report of this experiment's result: a
lower-tail coverage improvement on ~140-150 goalie-nights per origin is
directionally informative but should not be treated as tightly estimated.

---

## 8. Experiment 7 -- Cross-line outlier pricing (Component G)

**8.1 Hypothesis and failure mechanism targeted.** This experiment does not
target a shots-bias mechanism -- it targets a market-structure mechanism
described in section 3.4 item 3: same-line price dispersion across `us`
books is too small to exploit (mean absolute deviation from same-line
consensus ~0.35-0.55 probability points against a typical ~3.5-point
half-vig), but LINE dispersion (books posting different lines, e.g. 24.5 vs
25.5) is real and larger: ~11% of bettime goalie-nights and 16-18% of
closing goalie-nights have books a full save or more apart, worth 5-6
probability points per save (roughly 150-200 candidate goalie-nights per
season). The hypothesis is that a book posting an outlier LINE (not price)
relative to the cross-book consensus, translated across lines via a
distribution shape, can be identified and flagged as a mispriced quote --
without requiring any model edge over the market, only over one lagging
book.

**8.2 Data inputs and folds -- confirmed against `data/processed/saves_
lines_snapshots.parquet`'s actual season/pass coverage, not assumed from the
plan.** `data/processed/saves_lines_snapshots.parquet`. Development and
locking: 2023-24 bettime pass (15,682 rows) AND 2023-24 closing pass
(17,959 rows) -- both available, matching plan section 4.7's "develop and
lock on 2023-24 bettime and closing passes." Single test touch: 2024-25
CLOSING pass only (14,954 rows) -- confirmed the intended fold, because
2024-25's bettime pass has essentially no coverage (258 rows total across
the whole season, vs. ~15,000 for the other two seasons; see section 1.2).
The plan's own text anticipates this: "The bettime version of the 2024-25
test becomes possible after Phase 1" (the not-yet-executed 2024-25 opening
saves purchase). This pre-registration's 2024-25 touch is therefore the
closing-pass version only.

**8.3 An interpretive point that must be resolved before implementation,
flagged here rather than assumed:** plan section 4.7 point 4 says to "grade
flagged bets by outcomes AND by CLV against the closing consensus." Because
the 2024-25 touch is itself a closing-pass snapshot with no later reference
point, there is no temporal bettime-to-close CLV available for that touch
-- "CLV against the closing consensus" at test time can only mean the
flagged book's probability gap versus the translated cross-book consensus
AT THAT SAME closing snapshot (effectively the flagging signal itself,
scored for its relationship to the eventual game outcome), not a forward-
looking price-movement measurement. The genuine temporal CLV check (does a
flagged BETTIME outlier move toward the consensus by closing time,
confirming real early mispricing rather than a permanently soft/suspended
book) is only measurable on 2023-24, where both passes exist, and is
reported here as a DEVELOPMENT-stage diagnostic, not part of the locked
single-touch test. Any future confirmatory bettime-based 2024-25 CLV test
must wait for the Phase 1 purchase and is out of scope for this document.

**8.4 Metrics.** PRIMARY: goalie-night cluster-bootstrap 95% CI on ROI for
the locked policy's flagged bets on the 2024-25 closing-pass single-touch
test, restricted to venue-accessible quotes per section 0.3 (BetOnline
coverage as the direct analog, since Underdog/PrizePicks are lines-only
apps without their own historical archive in this dataset -- their
implementable form is the line-vs-consensus favorability check, reported
separately, not folded into the ROI number); OVER/UNDER split reported
separately per sections 3.4/4.6's mandatory requirement. SECONDARY: the
2023-24 bettime-pass development-stage CLV-toward-close diagnostic
described in 8.3, reported net of the 2023-24 unconditional bettime-to-
close drift baseline (already measured: -0.006%, cluster 95% CI [-0.023%,
+0.010%] -- i.e., statistically zero drift for this specific season, so the
subtraction is a documented no-op in practice, not a number that changes
the result, but must still be reported per the mandatory-drift-baseline
rule rather than silently omitted); coverage-at-bettable-books (% of
flagged quotes that are BetOnline or otherwise venue-accessible); book-
concentration diagnostic (% of flagged bets from a single book -- BetOnline
was the most frequent off-consensus book in the underlying dispersion
diagnostic, and per section 3.4/4.7, closing-snapshot BetOnline outliers
may be stale/suspended boards rather than bettable prices, which is exactly
what this diagnostic is designed to catch).

**8.5 Numeric pass/fail (Component G's gate -- constructed here from
sections 4.7/1a/3.4/4.6 by analogy to Gate C, since the plan does not
itself name a numbered gate for Component G; flagged as an interpretive
synthesis, not a verbatim plan quote).** On the 2024-25 closing-pass single
-touch test, using the threshold and shape-translation method locked on
2023-24 (both selected during development and NOT reselected after seeing
the 2024-25 result): PASS requires (a) cluster-bootstrap 95% CI on ROI
entirely above zero for the venue-accessible subset; (b) OVER and UNDER
splits both reported, with the result not carried entirely by one side (if
it is, the honest disposition is "a hypothesis about that side," per
section 9's "One direction appears profitable" risk -- not a general pass);
(c) not concentrated in one book or a handful of goalie-nights (operational
bar: no single book accounts for more than roughly half of flagged bets,
and no single goalie-night accounts for a disproportionate share of total
profit); (d) value must survive restriction to accessible venues -- a
result that only holds on inaccessible consensus books does not pass for
deployment purposes "however real it is" (section 7, Gate C language,
applied here by direct analogy since Component G's deployed form is
explicitly the section-1a venue filter).

**8.6 Forbidden.** The vig-clearing threshold and the shape-translation
method (Component E's mixture once built, or the existing NB2 pmf as a
first pass, per section 4.7) are selected on 2023-24 data ONLY and then
frozen in code before the 2024-25 closing pass is loaded for scoring -- not
selected by trying several thresholds against 2024-25 and keeping the best.
If the NB2-pmf first pass is used for shape translation, its dispersion
`alpha` must come from the SAME validation-fitted convention as Experiment
3 (or, if Experiment 3 has not concluded first, from a value fit on
2023-24 training/validation data only) -- never from the frozen production
artifact directly, since that artifact's training window (through
2025-10-15) postdates and overlaps differently with the 2023-24/2024-25
window this experiment evaluates. Do not treat a closing-snapshot outlier
as bettable without first checking the book-concentration diagnostic in
8.4 -- a flagged quote that is disproportionately one book (BetOnline
closing snapshots, per the section 3.4 caveat) may be a stale/suspended
board, and the honest disposition in that case is to restrict the strategy
to bettime snapshots and venue-accessible books, per section 4.7's own
instruction, not to claim the full flagged-bet ROI as real.

**8.7 Interpretation hierarchy.** 2023-24 development-stage results
(threshold/method selection, the bettime-to-close CLV diagnostic) are
hypothesis support only. The single 2024-25 closing-pass touch, evaluated
under the locked policy, is the strongest chronological evidence this
no-purchase round can produce for Component G -- but per 8.3, it is
evidence about closing-time cross-line mispricing against game OUTCOMES,
not a genuine forward CLV measurement, and that distinction must travel
with the result whenever it is cited. A true bettime-based confirmatory
touch (closer to Gate C's own standard) requires the Phase 1 purchase and
is out of scope here. Final confirmation is the frozen 2026-27 shadow run
with the venue filter (`scripts/check_venue_value.py`, not yet written)
wired into the daily workflow per section 1a/12.

---

## 9. Cross-experiment notes

- **Ordering matters.** Per plan section 6.2's closing instruction ("Do not
  add all components at once. Each component must justify its inclusion
  through distributional metrics before it can enter the combined model")
  and section 10's sequence, Experiments 1-3 (control baseline, funnel,
  dispersion) establish the Gate-A candidate architecture; Experiments 4-6
  (burn-in correction, market features, exposure mixture) are each layered
  on top of and evaluated against that candidate, not against the raw
  `pace_shots` recipe; Experiment 7 (Component G) is architecturally
  independent and can run in parallel with 1-6, but its shape-translation
  step should prefer Experiment 2/3's output once available over the NB2
  first pass, per section 4.7.
- **Gate A itself** is evaluated on the COMBINED architecture (Experiments
  2+3+6, with 4 and 5 included if they individually justified inclusion
  under 5.4/6.4), not on any single experiment in isolation -- the
  per-experiment pass/fail bars above are necessary conditions feeding one
  eventual Gate A verdict, not seven independent gates each unlocking a
  purchase on their own.
- **Nothing in this document authorizes a purchase.** Gate B (data
  coverage) and Gate C (movement-model evidence) remain downstream of the
  probes in plan section 5, which remain downstream of Gate A passing.

---

## 10. Results (recorded 2026-07-13; Experiments 1-3 and 5-7 executed, Experiment 4 deferred)

Executed by six parallel Sonnet sub-agents (Experiment 1's control-bias
number was produced by the step-0 verification agent and reused, not
re-run; Experiments 2-7 each ran independently). Two session-limit
interruptions occurred mid-run; every agent resumed cleanly from its saved
transcript, no work lost. Full numeric detail lives in each experiment's
`models/trained/experiment_*/metadata.json` and in
`docs/OFFSEASON_OPTIMIZATION_PLAN.md` section 6's 2026-07-13 log entry.
This section records only the pass/fail verdict against each experiment's
own registered bar above.

| # | Experiment | Verdict against THIS document's bar | One-line why |
|---|---|---|---|
| 1 | No-pace control | PASS (bar cleared) | Control bias +0.442/+0.031 vs. `pace_shots`'s confirmed +1.950/+1.845 -- well under half on both origins. |
| 2 | Season-normalized pace / attempt-to-SOG funnel | FAIL | Neither variant clears bias-reduction AND Brier-improvement on both origins; lower-tail calibration worsens under both. |
| 3 | Validation-fitted dispersion | FAIL (corrected 2026-07-13, dual audit; originally misrecorded as PASS) | Under this document's own registered bar (4.4a: summed central-coverage deviation no worse on BOTH origins), Origin A WORSENS under val-fitted dispersion (3.36 -> 7.80; train-fitted A was already nearly nominal centrally at 50.6/77.3) while Origin B improves (8.43 -> 4.22). The run that declared the PASS (`experiment_season_funnel`) computed the train-fitted alpha as diagnostic only and never built test predictions under it, so the registered comparison was never measured there; the exposure run's `coverage_results_test` contains it and shows the split verdict. What IS real: lower-tail P(<=k) gaps improve markedly on both origins, Origin A's PIT uniformity improves (summed 10-bin deviation 0.139 -> 0.069), and Origin B's Poisson-fallback bug is fixed. Val-fitted dispersion is a tails-vs-middle tradeoff, not an unconditional win; do not hard-adopt as a standing default -- the next round pre-registers a dispersion treatment (cross-fitted residuals or an explicit heavier-lower-tail shape) before reuse. |
| 4 | First-30-days burn-in correction | NOT RUN this round | Superseded by the Gate A failure (2/3) -- deferred; the coverage-day convention it pre-registered remains valid for a future attempt. |
| 5 | Market game-total/moneyline features (Component C) | FAIL (both-origins bar) | Origin A structurally uninformative (0% train coverage before 2023-10). Origin B shows a real, CI-excluding-zero Brier improvement (-0.00414) -- a genuine partial result, not carried into the combined model this round because Component C was never a Gate A requirement. |
| 6 | Exposure-state mixture (Component A) | FAIL (vs. PRIMARY baseline) | Exposure classifier AUC 0.52-0.55, statistically no better than a constant base rate; mixture loses to the correct primary baseline (no-pace control + val-fitted dispersion) on lower-tail coverage on both origins. Passes only against the literal old broken (train-fitted-dispersion) recipe, which conflates two effects. |
| 7 | Cross-line outlier pricing (Component G) | LOCK FAILED -- insufficient sample | No gap threshold on the pre-declared grid produced the required 20 graded bettime bets; the reported 0.02-threshold numbers are a fallback for visibility, not a validly locked policy. Per section 0.1's single-touch rule, the 2024-25 confirmatory touch does not proceed on this result. |

**Consequence for Gate A** (plan section 7): the combined architecture --
Experiments 2+3+6, with 4/5 folded in only if individually justified --
does not clear Gate A this round. NO experiment passed unconditionally
(Experiment 3's originally-recorded PASS was corrected to FAIL on
2026-07-13 -- see its table row and section 10.1). Experiments 2 and 6
are clean negatives for their specific proposed mechanisms, not evidence
against the underlying section-2 diagnosis (which Experiment 1 and step 0
both confirm). No SOG probe or 2024-25 opening-saves purchase is
authorized by this round's evidence.

**Consequence for Component G**: its core deployment premise (section
1a's venue-relative filter, keyed on `betonlineag`) is untestable on
2023-24 -- that book has zero quotes in the archive for that season.
Development-stage findings worth carrying forward: the candidate-pool
size (10.6% bettime / 15.5% closing goalie-nights with a full-save line
spread) matches the original section 3.4 forecast closely, and the
book-concentration diagnostic worked as designed (flagged bettime bets
were 82% one book, correctly triggering the "hypothesis about that book,
not a general finding" disposition rather than a false pass). A
materially larger sample -- more seasons, or the 2025-26 data where
BetOnline coverage should exist -- is needed before this component's
central hypothesis is testable at all, let alone locked.

One open discrepancy, unresolved and worth a future look: Component G's
locked dispersion alpha (0.100, grid-searched against outcome Brier) does
not match the ~0.030 reference alpha from the model-residual pipeline
(Experiments 2/3/step 0). The agent's working explanation -- a model-free
consensus mean is a noisier location estimate than a fitted conditional
mean, so wider dispersion partially compensates when calibrated against
raw outcomes -- is plausible but unconfirmed, and did not end up mattering
this round since the threshold lock failed regardless of shape choice
(outcome Brier was nearly flat, 0.24924-0.24942, across the whole
alpha/sigma grid).

### 10.1 Post-hoc audit corrections (2026-07-13, two independent audits)

After section 10 was first recorded, the round was audited twice,
independently: by Codex and by Claude (each recomputing headline numbers
from the raw artifacts -- Brier deltas, ROI, CLV, AUC, bias via model
reload, bootstrap CIs). Both audits agreed the experiments genuinely ran,
the chronological folds are correct, no outcome leakage was found, and
the major numbers reproduce. Both also found the same set of reporting/
implementation defects, corrected here so this document stays honest:

1. **Experiment 3's verdict was wrong and is corrected to FAIL** (see the
   amended table row). The original PASS was declared against the wrong
   metric (lower-tail gaps, which did improve) rather than the registered
   4.4a bar (summed central-coverage deviation, which worsened on Origin
   A: 3.36 -> 7.80 while B improved 8.43 -> 4.22). Root cause: the
   season-funnel run treated the train-fitted alpha as diagnostic-only
   and never built test predictions under it, so the registered
   comparison was never computed in the run that claimed the pass. The
   substantive lesson stands in amended form: a single NB2 alpha cannot
   fit the middle and the lower tail simultaneously (train-fitted A was
   centrally near-nominal but tail-blind; val-fitted fixed the tail and
   overshot the middle). The honest interpretation is that the residual
   distribution has a heavier lower tail than NB2 -- consistent with
   Experiment 6's finding that a pooled early-exit mixture nearly
   perfects the marginal tail (P(<=10) 0.025 vs. actual 0.029 on A).
2. **Experiment 2 deviated from its registration in two ways** (verdict
   unaffected -- the variants failed regardless): (a) the funnel's
   exposure stage used a league-prior starter share of team SOG
   (`fnl_starter_share`) instead of the registered "shots-on-goal-per-60
   estimate scaled to projected exposure minutes" (3.1b); the registered
   construction has still never been tested, and Experiment 6's
   shots-per-60 machinery now exists to build it. (b) The
   season-normalized variant fed 4 duplicated features (the pre-built
   `*_prior_league_z` columns alongside their recomputed `sz_*` twins) --
   redundancy, not leakage.
3. **Experiment 6's verdict is better stated as "gate failed, mechanism
   inconclusive."** The registered conditional-coverage bar is poorly
   suited to a ~5%-weight pooled component riding a non-discriminative
   classifier: the classifier (AUC 0.52-0.55, statistically no better
   than a constant) cannot concentrate tail mass per-game, but the
   pooled mixture's MARGINAL tail calibration was the best of any
   variant tested this round, and its central coverage on B (summed dev
   2.89) beat both dispersion treatments. The reusable idea is a fixed
   (constant-weight) early-exit tail component, no classifier required.
4. **Experiment 7's contract had two gaps** (verdict unaffected -- the
   lock failed on sample size regardless): (a) the >=20-graded-bettime-
   bets minimum lived in the script's `min_flagged_bettime_bets_for_lock`
   and this document's RESULTS table, not in binding section 8 -- a
   future re-registration must state it up front; (b) the policy never
   required the flagged book's line to be off-modal, despite cross-LINE
   translation being the stated hypothesis (1 of 11 bettime and 12 of 42
   closing flags were at the modal line -- those are ordinary same-line
   consensus disagreements, not cross-line translations). Both must be
   repaired before any re-lock attempt.
5. **The most promising single post-hoc observation of the round**
   (surfaced by Codex, independently verified by Claude, and NOT a
   registered result -- recorded as a hypothesis for the next round
   only): on Origin B's 2024-25 closing pass, the market-state variant's
   UNDER-side picks at the fixed 0.05 EV threshold returned +11.18%
   ROI/bet (cluster CI [+4.21%, +18.03%], n=2031 bets/762 nights) across
   all books, and +8.66% (CI [+0.53%, +16.57%], n=513) at BetOnline
   specifically -- the one straight-bet venue the user can execute at.
   Critically, a blind bet-every-UNDER baseline on the same quote
   universe returns only +1.06% (CI [-2.77%, +4.82%]) all-books / +1.11%
   (CI [-3.14%, +5.22%]) BetOnline-only, so the return reflects quote
   SELECTION, not the season-wide 2024-25 UNDER drift. This is post-hoc
   (variant, side, and book sliced after seeing results), single-season,
   and at closing timestamps rather than the executable window; it
   authorizes nothing except the pre-registered Origin C replication
   described in the plan's section 10 next-round entry.

---

## 11. Experiment 8 -- Origin C market-state replication

Registered 2026-07-13 by the lead reviewer (Claude), BEFORE any Origin C
model or test prediction existed anywhere in this repo. This is the plan's
section 10 step 6a. Unlike the round-1 experiments (section 0.2's
concurrency caveat), this registration is temporally strict: nothing below
was chosen after seeing any 2025-26 model output.

**11.1 Hypothesis.** Experiment 5's Origin B result -- market game-state
features (Component C) produce a real paired-Brier improvement over the
no-pace control -- replicates on a fold no model in this repo has ever
been trained toward (test = 2025-26). Secondarily, the post-hoc UNDER-
selection observation (section 10.1 item 5) generalizes to the executable
venue and window: the model's UNDER picks at BetOnline bettime quotes
outperform betting every UNDER quote in the same universe.

**11.2 Frozen recipe -- nothing may be reselected.** Exact feature sets
from `experiment_market_state_20260710_213106/metadata.json`: the
104-column no-pace control shots feature list, plus the 7 `mkt_*` market
feature columns, plus the `mkt_matched` indicator; the same hyperparameter
grid and validation-selection protocol as that run (via the same script
lineage); shared save-rate model design unchanged; `ORIGIN_CAP` 90; fixed
EV threshold 0.05; goalie-night cluster bootstrap, 10,000 resamples, seed
42. No new features, no removals, no grid changes, no threshold changes.

**11.3 Folds (rolling-origin convention, carved with
`ero.carve_origin_split`).** Origin C pool = seasons 2022-23 + 2023-24 +
2024-25; validation = final 49 days of the pool date range; train = the
rest. Test = season 20252026 (expected 2,624 goalie-games). 2025-26 must
not appear in train or val in any form. Market-feature join coverage
(verified 2026-07-13 before registration): 0% of 2022-23, ~100% of
2023-24 onward -- roughly 64% of train rows, 100% of val and test.

**11.4 Data inventory (verified on disk 2026-07-13, quote counts recorded
before any model existed).** Features: `clean_training_data.parquet`,
`game_context_features.parquet`, `pace_features.parquet`,
`market_game_features.parquet` (the `is_latest_pregame_snapshot` view,
same construction as Experiment 5). Closing-pass 2025-26 quotes:
`multibook_classification_training_data.parquet`, season 20252026 (5,729
paired quotes, 2025-10-07..2026-04-13; `book_key` counts: betmgm 1,231,
betonlineag 1,034, draftkings 1,063, bovada 1,026, underdog 927, fanatics
274, betonline 174). The `betonline` vs `betonlineag` key split (174 vs
1,034 rows) must be diagnosed and reported BEFORE grading -- if they are
the same book under two labels, say so and merge with justification; if
provenance is unclear, use `betonlineag` only and report the exclusion.
Bettime-pass 2025-26 quotes: `saves_lines_snapshots.parquet`,
`snapshot_pass == "bettime"` (12,811 rows, of which 2,662 betonlineag
across 1,212 goalie-nights); over/under sides must be paired per book/
line/goalie-night (the pairing pattern in
`scripts/experiment_cross_line_pricing.py` is the reference
implementation) and graded against `clean_training_data` saves.

**11.5 Wiring gate (mandatory, before any Origin C test prediction).**
The new script must first re-run Origin B through its own code path and
reproduce `experiment_market_state_20260710_213106`'s recorded Origin B
paired Brier delta vs. control at closing (mean -0.0041404) to within
1e-4, with the same n_bets (7,463) and n_clusters (2,510). If it does not
reproduce, STOP and report -- do not proceed to 2025-26.

**11.6 Viewed-data status of the test fold.** 2025-26 outcomes are
"viewed" (the live production betting record on that season is known and
UNDER-heavy). No model was ever trained or selected on it, but a raw ROI
number on 2025-26 UNDERs is worthless as evidence by construction. The
primary metrics below are chosen to be robust to this: P1 is side-neutral
and market-relative; P2 nets out the season-wide direction by
differencing against blind-UNDER on the identical quote universe.

**11.7 PRIMARY metrics and pass bars (both must pass; plan step 6e's
"replicates" means exactly this).**
- **P1 (accuracy replication):** paired Brier delta (market-state variant
  minus no-pace control), model probability at each posted line, closing
  pass, all books, non-push rows, goalie-night cluster bootstrap. PASS =
  CI95 entirely below zero.
- **P2 (executable selection effect):** on the universe U = paired,
  gradeable `betonlineag` bettime quotes for test-fold games (pushes
  excluded from grading per standard convention): model arm = UNDER bets
  where the market-state model's EV(UNDER) >= 0.05; blind arm = UNDER on
  every quote in U. Per bootstrap resample (resampling goalie-night
  clusters once, both arms computed within the same resample), delta =
  ROI_model - ROI_blind. PASS = CI95 entirely above zero. American-odds
  profit convention: win pays odds/100 (positive) or 100/|odds|
  (negative); loss = -1. Resamples with an empty model arm are counted
  and reported; if they exceed 1% of resamples, P2 is UNSTABLE (no pass).
  If the model arm has fewer than 100 graded bets, P2 is INSUFFICIENT
  SAMPLE (no pass, not a fail -- report and stop there).

**11.8 SECONDARY metrics (report, no gate):** Brier vs. de-vigged market
(closing and bettime); OVER/UNDER ROI splits at the fixed threshold, both
passes, all-books and BetOnline cuts; the all-books bettime
selection-over-blind delta; bettime-to-close CLV of the model's flagged
bets net of the unconditional 2025-26 drift baseline (same matched-quote
convention as the Component G run); shots-model signed bias and MAE on the
test fold for both variants; join coverage by fold.

**11.9 Dispersion policy (per the Experiment 3 correction, section 10.1
item 1).** Headline results use validation-fitted NB2 dispersion --
matching the frozen Origin B recipe being replicated, NOT because it is
an adopted default. Train-fitted dispersion results are produced
side-by-side as a sensitivity check: if either P1 or P2 flips sign under
train-fitted dispersion, the replication is reported as
DISPERSION-FRAGILE (no pass).

**11.10 Forbidden.** Adding/removing/reweighting features; any
hyperparameter, threshold, or calibration reselection; introducing new
post-hoc slices as results (anything exploratory goes in a clearly
labeled exploratory block of metadata and is excluded from the verdict);
touching 2025-26 rows during training/validation; re-running with
variations after seeing P1/P2 (one shot -- if the wiring gate passes and
the run completes, the first P1/P2 numbers are the result); consuming any
Odds API credit or network resource.

**11.11 Consequence mapping (fixed in advance).** PASS (P1 and P2 both
pass, dispersion-stable): the market-anchored model is promoted per plan
step 6e -- 2026-27 shadow/token-stake candidacy, and the 2024-25
bettime-pass purchase becomes worth reconsidering. FAIL of P1: Experiment
5's Origin B result is treated as origin-specific; Component C drops out
of the front of the queue. FAIL of P2 alone (P1 passing): the accuracy
gain is real but the executable-venue selection effect is not
demonstrated at bettime; no purchase, no promotion, revisit with 6b/6c
architecture work. INSUFFICIENT SAMPLE on P2: report, and the closing-
pass all-books selection delta (secondary) informs -- but does not decide
-- whether a bettime re-test on 2026-27 live data is worth the wait.

**11.12 Results -- P1 PASS; P2 INSUFFICIENT SAMPLE (Codex-verified
2026-07-13).** Experiment 8 ran once, without network access or Odds API
credit, through `scripts/experiment_market_state_origin_c.py`. Artifacts
are in
`models/trained/experiment_market_state_origin_c_20260713_140706/`.
The mandatory wiring gate reproduced Origin B exactly: paired-Brier mean
`-0.0041404240194266384`, 7,463 rows, and 2,510 goalie-night clusters.
Origin C then used 7,134 train rows (2022-10-07 through 2025-02-27), 738
validation rows (2025-02-28 through 2025-04-17), and the expected 2,624
test rows from 2025-26 only.

- **P1 PASS:** market-state minus no-pace-control paired Brier was
  `-0.003111`, CI95 `[-0.005039, -0.001192]`, across 5,729 closing
  quotes and 2,070 goalie-night clusters. Train-fitted dispersion gave
  `-0.003171`, so the sign did not depend on the dispersion fit.
- **P2 INSUFFICIENT SAMPLE:** the gradeable BetOnline bettime universe
  contained 1,185 goalie-nights, but only 85 qualified model UNDER bets,
  below the registered 100-bet floor. Model-arm ROI was `-11.40%`
  versus `-5.24%` for blind UNDER, a `-6.16`-point delta with CI95
  `[-25.36, +13.25]`. This is not a registered failure, but the point
  estimate is unfavorable. The all-books secondary was also unsupportive:
  434 selected UNDERs, delta `-0.85` points, CI95
  `[-18.34, +16.54]`.
- **Market-relative accuracy:** the market-state model was statistically
  tied with the de-vigged closing market (Brier delta `+0.00049`, CI95
  `[-0.00271, +0.00375]`) and worse than the bettime market (`+0.00383`,
  CI95 `[+0.000001, +0.00752]`). It improved its own control; it did not
  demonstrate superior probability estimates to the executable market.
- **The Origin B UNDER mechanism did not replicate:** fixed-threshold
  all-books ROI was carried by OVER at both closing (`+8.26%` OVER versus
  `-2.74%` UNDER) and bettime (`+3.24%` versus `-5.75%`). At BetOnline
  bettime, OVER returned `+2.51%` and UNDER `-11.40%`.
- **CLV remained positive but small:** flagged-bet probability CLV net of
  2025-26 unconditional drift was `+0.00121`, CI95
  `[+0.00033, +0.00209]`, on 1,073 matched bets / 307 clusters all
  books and `+0.00194`, CI95 `[+0.00089, +0.00303]`, on 240 matched
  BetOnline bets / clusters. This is 0.12-0.19 probability points, not
  evidence of a vig-clearing outcome edge by itself.
- **Other checks:** market features slightly improved shots MAE
  (`5.408` to `5.360`) while increasing signed bias (`+0.235` to
  `+0.422`). Market-feature join coverage was 63.16% train, 100% val,
  and 99.92% test. The `betonline`/`betonlineag` closing keys were traced
  to the same book through non-overlapping source paths and merged only
  for the relevant secondary cuts; P2 contained `betonlineag` only.
  The frozen function named `calculate_ev` computes probability edge
  (model probability minus implied probability), not monetary expected
  return; the run correctly preserved that registered repository
  convention rather than silently changing the policy.

Codex independently reconstructed P1, P2, side ROI, market-relative
Brier, drift, and CLV directly from the saved prediction parquet and raw
closing/bettime quote parquets, using separate code and different
bootstrap seeds. Point estimates matched exactly; confidence intervals
matched within expected bootstrap noise. The four logged implementation
deviations are representational or code-path differences with no numeric
effect. The registered consequence is therefore binding: **no promotion,
no data purchase, and no claim that the executable UNDER-selection edge
replicated.** Component C has a reproducible accuracy benefit over the
no-pace control, but the current model remains at best market-parity and
does not have a demonstrated bettime betting edge.

---

## 12. Experiment 9 -- Fixed-offset funnel plus fixed-weight exposure mixture

Registered 2026-07-13 by Codex, before any Experiment 9 candidate model
or Origin A/B candidate prediction existed. This is plan steps 6b and 6c
as one zero-credit experiment. Origins A and B are viewed development
seasons, so even a full pass is architecture evidence, not an untouched
betting-edge confirmation.

**12.1 Hypothesis.** The prior attempt-to-SOG funnel failed because its
stages were offered to XGBoost as ordinary features and its exposure
stage used starter share. The deterministic projection itself was nearly
centered. Locking that projection as a workload-rate offset, allowing the
learner to estimate only a multiplicative residual, and representing
early exits with a constant train-fold exposure mixture can remove the
shots-level bias without sacrificing Brier or central calibration.

**12.2 Fixed-offset workload-rate model.** Reuse the strictly prior-only
funnel arithmetic from `scripts/experiment_season_funnel.py`, but remove
`fnl_starter_share`. Define:

```text
A = opp_corsi_ema5 * team_corsi_against_ema5 / prior_league_corsi
F = A * opp_unblocked_frac * team_unblocked_frac
      / prior_league_unblocked_frac
c = prior_league_sog / prior_league_fenwick
r_opp  = (opp_shots_roll10 / opp_fenwick_ema5) / c
r_team = (team_shots_against_roll10 / team_fenwick_against_ema5) / c
lambda0_60 = F * c * sqrt(clip(r_opp, 0.5, 2.0)
                           * clip(r_team, 0.5, 2.0))
```

League rates exclude the entire current game date. Missing offsets use a
fixed `30.0 SOG/60` fallback and their count is reported; no test row may
be dropped. This simple fallback is locked to avoid choosing a prior-
season or rolling fallback after seeing which helps.

The target is `y60 = shots_against / (max(TOI_minutes, 10) / 60)`, with
the training target winsorized at the train-fold 99.5th percentile only.
Use the existing `ds.SHOTS_CONFIGS`, `objective="count:poisson"`, and
select by validation rate60 MAE. The deterministic rate is XGBoost's raw
margin:

```text
z_i = log(max(lambda0_60_i, 1e-3))
lambda_hat_60_i = exp(z_i + f_theta(X_i))
```

`base_margin=z` must be supplied for train, validation, and every
prediction. `X` is exactly the 104 no-pace base+engineered columns. No
`fnl_*`, pace, context, offset, actual TOI, or current-game outcome
column may enter `X`. An additive squared-error residual model and a
free-input funnel variant are forbidden.

**12.3 Exposure and distribution shape.** From each origin's training
fold only, define `pi = mean(TOI < 50)`; build Laplace-1 empirical TOI
weights using 5-minute bins on `[0,50)` and 1-minute bins on `[50,66)`.
No exposure classifier is trained. Let `Tbar` be the mixture-weighted
mean of those bins. The single-body arm uses
`mu_i = lambda_hat_60_i * Tbar / 60`. The fixed-mixture arm is:

```text
P_mix(S=s) =
  pi * sum_b w_early,b  H(s | lambda_hat_60 * t_b/60, alpha, q_i)
  + (1-pi) * sum_b w_normal,b H(s | lambda_hat_60 * t_b/60, alpha, q_i)
```

where `H` is the existing NB2-shots/Binomial-saves compound PMF,
`ORIGIN_CAP=90`, and `q_i` comes from one shared no-pace save-rate
model. The single and mixture arms therefore have the same expected mean;
only distributional shape differs.

The PRIMARY body `alpha` is fit once per origin on validation residuals
using rate predictions composed with actual validation TOI. This uses TOI
only to calibrate the exposure-conditioned body variance; actual test TOI
is forbidden in every prediction. The same alpha is shared by the single
and mixture arms. A train-fitted-alpha version using actual train TOI is
reported as a sensitivity analysis only; no per-origin alpha selection is
allowed.

**12.4 Folds, baseline, and wiring gate.** Reuse
`season_date_range`, `carve_origin_split`, and
`date_range_test_idx` unchanged:

- Origin A: train 2022-10-07..2023-02-24; val
  2023-02-25..2023-04-14; test season 2023-24.
- Origin B: train 2022-10-07..2024-02-29; val
  2024-03-01..2024-04-18; test season 2024-25.

The Gate-A baseline is the exact no-pace control: direct shots-count
model, same 104 columns/grid, shared save-rate recipe, and identical
joined quote rows. Before candidate test predictions, the new script must
reproduce the prior no-pace test shots biases `+0.4419625` (A) and
`+0.0308310` (B) within `1e-4`, plus its closing-pass Brier point
scores `0.2552057011` (A) and `0.2512895068` (B) within `1e-6`. If
the gate fails, stop. Do not run the candidate.

**12.5 PRIMARY pass bars -- all are required on both origins.**

1. **Shots level:** combined candidate
   `abs(mean(mu_pred - shots_against)) < 0.5`.
2. **Probability accuracy:** on identical closing quote rows,
   `Brier(candidate fixed mixture) - Brier(no-pace control) < 0`.
   Report a 10,000-resample goalie-night cluster CI. A point improvement
   with CI crossing zero technically clears the inherited Experiment 2
   bar but must be called statistically weak.
3. **Central coverage:** on all 2,624 test starts,
   `D = abs(cov50-50) + abs(cov80-80)`; require
   `D_mixture <= D_single`. Inclusive discrete-PMF 25/75 and 10/90
   quantiles are fixed.
4. **Lower-tail calibration:** for
   `K={5,10,15,20,25,30}`, define
   `L = sum_k abs(mean(F_i(k)) - mean(1[saves_i<=k]))`; require
   `L_mixture < L_single`. No cutoff may be selected or omitted after
   viewing results.
5. **No extreme edge inflation:** closing-pass bet rate under the
   repository's fixed 0.05 probability-edge policy must remain in the
   historical 15%-45% band.

No averaging across origins is allowed. Failure of any bar on either
origin means the combined 6b/6c Gate-A candidate fails. If shots bias and
Brier pass but the mixture bars fail, report the fixed-offset mean model
as a partial mechanism result, not a Gate-A pass.

**12.6 SECONDARY metrics.** Report test shots MAE; randomized PIT;
full-distribution negative log score; coverage and lower-tail tables;
paired Brier versus the de-vigged market; side calibration; fixed-0.05
bet count/rate/ROI with goalie-night cluster CIs; train-alpha
sensitivity; test `TOI<50` diagnostics clearly labeled postgame-only;
offset raw coverage and fallback counts; persisted offset formula,
winsor cap, `pi`, TOI bins, alpha, feature list, models, predictions,
metadata, and run log.

**12.7 Forbidden and consequence.** No Odds API or network use; no new
features; no hyperparameter, bin, cutoff, alpha, threshold, or fallback
selection against either test fold; no ROI-based variant choice; no
actual test TOI in predictions; no dropped test rows; no second candidate
run after viewing the first completed P1-P5 readout. A full pass
authorizes the low-credit Gate-B data probes and a frozen future-season
shadow candidate, not live-stake promotion or a claim of market edge. A
failure leaves purchases blocked and moves the queue to plan step 6d or
a newly preregistered architecture justified by the failure mechanism.

**12.8 Results -- COMBINED GATE A FAIL (Codex-verified 2026-07-13).**
Experiment 9 ran once through
`scripts/experiment_fixed_offset_mixture.py`; artifacts are in
`models/trained/experiment_fixed_offset_mixture_20260713_144811/`.
Both mandatory no-pace wiring gates reproduced exactly before candidate
test prediction: A bias/Brier `+0.441962493 / 0.255205701`; B
`+0.030831029 / 0.251289507`. The exact 104-column feature identity,
Origin A/B folds, every candidate `base_margin` call, model reloads,
train-only exposure parameters, offset fallbacks, and zero-network policy
all passed audit.

| Registered bar | Origin A | Origin B |
|---|---:|---:|
| P1: `abs(shots bias) < 0.5` | `+0.778396` -- **FAIL** | `+1.991173` -- **FAIL** |
| P2: mixture minus control closing Brier `< 0` | `+0.002161`, CI95 `[-0.002530,+0.007015]` -- **FAIL** | `+0.015262`, CI95 `[+0.008460,+0.022075]` -- **FAIL** |
| P3: mixture central deviation no worse than single | `4.8323 vs 3.5137` -- **FAIL** | `1.5168 vs 5.5716` -- **PASS** |
| P4: mixture aggregate lower-tail error below single | `0.091605 vs 0.123107` -- **PASS** | `0.298533 vs 0.329355` -- **PASS** |
| P5: closing bet rate 15%-45% | `41.914%` -- **PASS** | `59.333%` -- **FAIL** |

The fixed-offset mean model is not a partial mechanism pass: P1 and P2
failed on both origins. Candidate shots MAE also worsened versus control
(A `5.7960 vs 5.7556`; B `5.7068 vs 5.4117`). Closing ROI was
`-2.33%` on 3,722 A bets, CI crossing zero, and `-3.95%` on 4,428 B
bets, CI crossing zero. The candidate was significantly worse than the
de-vigged market on Brier on both origins.

The distribution-shape result is real but insufficient: the fixed
mixture improved aggregate lower-tail error and full-distribution
negative log score on both origins (single to mixture: A
`3.5627 -> 3.4242`; B `3.5124 -> 3.3928`). It improved central
coverage only on B. A constant roughly 5%-6% early-exit component adds
useful marginal tail mass, but cannot identify the actual short starts:
postgame-only `TOI<50` bias remained `+14.66` shots on A and
`+15.30` on B. Smearing that tail mass across every game therefore
improves whole-distribution scoring while degrading the posted-line
probabilities that drive betting. Train-alpha sensitivity did not rescue
any conclusion.

Codex independently rebuilt P1-P5, PMF coverage/tails/log scores, policy
ROI, quote probabilities, fallbacks, and fresh-seed cluster CIs from the
saved parquets. Two additional independent agents separately audited the
code and recomputed the artifacts; neither found a material leakage,
wiring, fold, or metric defect. The audit made two label/guard-only
corrections without a rerun: `statistically_weak` now applies only to a
negative Brier improvement whose CI crosses zero, and future runs verify
PMF expected means directly. Current single-vs-mixture PMF expected-saves
gaps are at most `0.00121` saves from cap truncation and are immaterial.
The one Origin A integer-line push follows the inherited harness's
binary-over convention; changing that single row does not affect a
verdict.

**Binding consequence:** purchases remain blocked; this architecture is
not promoted or rerun. Retain the fixed-weight mixture only as a possible
shape layer if a future, independently justified mean model exists. The
next queued item is plan 6d: repair Component G's contract and run only
the predeclared outcome-blind 2025-26 volume reconnaissance.

---

## 13. Experiment 10 -- Component G executable-volume reconnaissance

Registered 2026-07-13 by Codex before any 2025-26 Component G candidate
count was calculated. This repairs the omissions identified in the dual
audit of Experiment 7 and performs only an outcome-blind arithmetic
screen. It does not grade a bet, select or validate a policy, or authorize
an outcome touch.

**13.1 Question and binding population.** Does the available 2025-26
bettime archive contain at least 20 unique, executable BetOnline goalie-
nights that the frozen cross-line scorer flags at its most permissive
predeclared threshold after the target quote is required to be strictly
off-modal? The binding target is `book == "betonlineag"` in
`data/processed/saves_lines_snapshots.parquet`; the `betonline` alias from
the separate live-tracker fallback is not merged. Use season dates
2025-08-01 through 2026-07-31, `snapshot_pass == "bettime"`, and at most
one selected quote per `(event_id, goalie_id)`.

**13.2 Outcome firewall and input identity.** The reconnaissance may open
exactly one row-bearing input, the snapshot parquet above, pinned at
SHA-256
`E81CA4EA01B1DFB3F69068782B75E18B4D174BEF82DD34D8E341B59CBC94ED56`.
It must read only:

```text
event_id, commence_time, game_date_eastern, requested_ts, resolved_ts,
snapshot_pass, book, goalie_id, side, line, price_decimal
```

The script must not open `clean_training_data.parquet`, a closing frame,
the betting database, or any result table; must not load or derive saves,
shots against, goals against, TOI, result, grade, outcome, profit, ROI, or
CLV; and must contain no outcome join or outcome-calibration call. The
embedded `goalie_id` has historical postgame matching lineage, so this is
accurately described as **no outcome columns loaded**, not as a guarantee
that the source parquet's identity lineage was built without postgame
information.

**13.3 Quote cleanup, pairing, and modal rule.** Within the allowed season
and pass, retain the earliest `requested_ts` per event. Drop only exact
duplicates on `(event_id, requested_ts, book, goalie_id, side, line,
price_decimal)`. Abort on conflicting prices for the same quote key/side
or on more than one paired line for a `(event_id, goalie_id, book)`; do not
silently take the first. Pair OVER and UNDER only at the exact
`(event_id, goalie_id, book, line, commence_time, game_date_eastern)` key
and require both prices.

Compute line frequencies from all paired books before selecting
BetOnline. The modal set is **every** line tied for the maximum distinct-
book count. A target is strictly off-modal only if its line is outside
that entire set; the old directionally asymmetric `mode().min()` tie rule
is forbidden. No separate full-save line-spread minimum is imposed,
because Experiment 7 never bound one into `flag_bets`; line spread must be
reported as a diagnostic.

**13.4 Frozen reconnaissance scorer.** Freeze the Experiment 7 NB2
translator exactly for volume measurement: `alpha=0.100000`, support cap
70, paired-price additive de-vigging, quote-to-NB2-mean inversion, and a
leave-one-book-out consensus formed by averaging the other books' implied
means. Require at least one non-BetOnline peer. Translate that consensus
mean to BetOnline's posted line and define:

```text
gap_side = translated fair probability - raw vig-inclusive implied probability
```

At threshold `t`, select OVER only when `gap_over >= t` and
`gap_over > gap_under`; otherwise select UNDER when `gap_under >= t`.
This preserves Experiment 7's scorer and American-rounding convention.
It is a frozen counter, not a rehabilitated model or valid betting policy.

Report every threshold in the old predeclared grid:
`0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20`.
No threshold may be selected from these counts, and `0.02` remains an
invalid fallback rather than a lock.

**13.5 Primary arithmetic gate and reporting.** The primary count is the
number of unique strictly off-modal `betonlineag` goalie-nights at
`t=0.02`. Because the thresholds are nested:

- `<20` candidates means every future threshold necessarily fails the
  minimum and Component G is arithmetically too sparse at the user's
  accessible venue. Deprioritize it; no outcome touch follows.
- `>=20` candidates means only that a future lock may be arithmetically
  possible. It does not pass Component G, prove edge, select a threshold,
  or authorize an outcome touch without a separate preregistration.

For each threshold report total unique candidates and OVER/UNDER counts.
Also report the full BetOnline paired-quote denominator, strictly off-
modal denominator, month counts, peer-book counts, line-spread summaries,
missing-goalie counts, exact duplicates removed, and all fail-closed QA
assertions. Persist aggregate counts and metadata only; do not persist
identifiable candidate rows.

**13.6 Repaired future lock rule -- not executed here.** Any separately
authorized outcome-bearing development run must apply quote cleanup,
strict off-modal status, BetOnline accessibility, side selection, and
outcome matching before testing eligibility. A threshold requires both
`>=20` unique graded BetOnline bettime goalie-nights and `>=20` of those
with non-null exact-line closing consensus before its clustered CLV can
be evaluated. The smallest eligible predeclared threshold whose goalie-
night-clustered 95% CI for CLV net of unconditional drift is entirely
above zero may be locked; any fallback remains `NO LOCK`. An untouched
chronological confirmation must then show a BetOnline-only clustered ROI
CI entirely above zero, report sides separately, and pass concentration
checks. Inaccessible-book results cannot carry the decision.

**13.7 Results -- TOO SPARSE (Codex-verified 2026-07-13).** The registered
reconnaissance ran once through
`scripts/experiment_cross_line_volume_recon.py`; aggregate-only artifacts
are in
`models/trained/experiment_cross_line_volume_recon_20260713_153032/`.
The input hash matched the registration, exactly the 11 allowed columns
were selectively loaded, and no outcome, closing, database, grading, ROI,
CLV, or network path was opened. The run used zero Odds API credits.

After the earliest-request and exact-duplicate rules, 5,763 paired book
quotes covered 1,370 goalie-nights. BetOnline supplied 1,185 paired
goalie-night quotes. Only 31 were strictly outside the complete tied-
modal set; all 31 had at least one valid non-BetOnline peer. The frozen
scorer then produced:

| Threshold | Total | OVER | UNDER |
|---:|---:|---:|---:|
| `0.02` | **1** | 0 | 1 |
| `0.03` through `0.20` | **0** at every threshold | 0 | 0 |

The primary arithmetic gate therefore fails decisively: `1 < 20`. The
one candidate occurred in December 2025; it is not graded and carries no
edge interpretation. Every strictly off-modal BetOnline quote represented
exactly a one-save line spread. Twenty NB2 inversions failed across the
full all-book frame, but none reduced the 31-row target off-modal/peer
denominator, and the registered fail-closed handling was preserved.

Codex reviewed the script and artifact directly. A separate agent rebuilt
the result from the raw snapshot parquet without importing the experiment
script and reproduced every denominator, monthly count, threshold/side
count, and QA result exactly. A second static audit found no material
implementation, firewall, scorer, modal-set, persistence, or verdict
defect. It noted only that sparse month/peer/line-spread aggregates can
narrowly describe the lone anonymous candidate; no event, goalie, quote,
or outcome identifier is persisted.

**Binding consequence:** Component G's strictly off-modal BetOnline form
is arithmetically nonviable on this season and is closed without an
outcome touch. No threshold is locked, no grading or confirmatory run is
authorized, and purchases remain blocked. Revisit only if a materially
different executable venue or quote product creates a much larger
pre-registered candidate universe; do not loosen the off-modal rule or
minimum after seeing this count.
