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

---

## 14. Experiment 11 -- Frozen-Origin-B P2 re-test on the 2024-25 bettime pass

Registered 2026-07-14 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, BEFORE any parsing or analysis of the core_bettime_202607
records beyond the persisted audit
(`data/raw/betting_lines/passes/core_bettime_202607/audit_summary.json`,
generated 2026-07-14T05:31:14Z). This is plan section 5.7's stated
rationale for the 2024-25 saves pass, operationalized as a binding test.

**14.1 Hypothesis and honesty note.** Experiment 8's P2 (section 11.7)
tested whether the frozen Origin B `control_plus_market_state` model's
UNDER selection beats a blind-UNDER baseline on the executable
`betonlineag` bettime venue, and returned INSUFFICIENT SAMPLE (85
qualified UNDERs against a 100-bet floor, section 11.12). The purchased
2024-25 bettime pass makes that same test runnable, for the first time at
meaningful scale, on the season Origin B's model was actually built and
tested for. Stated plainly, per this document's own discipline:
Experiment 5 already measured Origin B's 2024-25 CLOSING accuracy result
(paired Brier delta vs. control `-0.0041404`, sections 10.1/11.5), 2024-25
outcomes are already viewed (plan section 6.1), and the live 2025-26
betting record is UNDER-heavy by construction (section 0.1;
`docs/OFFSEASON_OPTIMIZATION_PLAN.md` section 4). None of that is new.
What IS genuinely new is the EXECUTABLE bettime BetOnline selection test
itself, on a season the frozen model never trained or validated on
(14.2's fold boundaries), at roughly double Experiment 8's qualified-bet
sample -- BREAKTHROUGH_MODEL_PLAN.md section 5.7's own estimate, not
independently re-derived here since doing so requires the price-level
join this document is not allowed to perform before registration.

**14.2 Frozen recipe -- nothing may be reselected.** Reuse, unchanged, the
Origin B `control_plus_market_state` artifact from
`models/trained/experiment_market_state_20260710_213106/metadata.json`:
112-column shots feature list (104 no-pace-control columns + 7 `mkt_*` +
`mkt_matched`); shots model config `shallow_highreg`
(`max_depth=2, learning_rate=0.05, min_child_weight=30, subsample=0.8,
colsample_bytree=0.8, n_estimators=400, reg_lambda=5.0`,
`origin_b_control_plus_market_state_shots_model.json`); shared save-rate
model config `base` (`max_depth=3, learning_rate=0.05,
min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
n_estimators=400, reg_lambda=1.0`,
`origin_b_shared_save_rate_model.json`); dispersion
`alpha=0.026775577916660614` (negative-binomial, fit on `val_idx` --
validation residuals, per the Experiment 3 correction, section 10.1 item
1); `ORIGIN_CAP=90`; fixed EV threshold `0.05`. Origin B's fold
boundaries are unchanged and not re-carved: train 2022-10-07 to
2024-02-29 (4,528 rows), val 2024-03-01 to 2024-04-18 (720 rows), test
season 20242025 (2,624 rows). No model is retrained -- this is INFERENCE
ONLY: THREE frozen model JSON files are reloaded bit-identical (XGBoost
JSON round-trips exactly, the same reload convention Experiment 8's
script used for its train-fitted-dispersion sensitivity pass):
`origin_b_control_plus_market_state_shots_model.json`,
`origin_b_no_pace_control_shots_model.json` (the control leg -- required
because the 14.4 wiring gate reproduces a paired delta AGAINST the
no-pace control variant), and `origin_b_shared_save_rate_model.json`
(shared by both variants). The control leg's own frozen recipe is
likewise not re-derived: winner config `depth4_mcw20` (`max_depth=4,
learning_rate=0.05, min_child_weight=20, subsample=0.8,
colsample_bytree=0.8, n_estimators=400, reg_lambda=1.0`), val-fitted
dispersion `alpha=0.02852173299997726` (both verified in `metadata.json`
at `origin_b/variants/no_pace_control`). Each variant is repriced under
its own recorded val-fitted alpha from `metadata.json` (market-state
`0.026775577916660614`, control `0.02852173299997726`) via the shared
harness functions (`join_and_price`, `compute_distribution_predictions`,
section 1). The control model exists solely to compute the 14.4
wiring-gate delta and any vs-control secondary; P2 itself uses only the
market-state variant. No new features, no grid changes, no threshold
changes, no re-fitting of dispersion against the new data.

**14.3 Data inventory (verified 2026-07-14).** Existing archive, directly
re-read from `data/processed/saves_lines_snapshots.parquet` (independent
of `audit_summary.json`): 2024-25 bettime pre-purchase = 258 rows / 21
unique `event_id` (negligible, matches section 1.2's "2024-25 has no
bettime pass"); 2024-25 closing = 14,954 rows / 1,288 unique events, of
which `betonlineag` closing = 3,912 rows / 1,094 unique events; 2023-24
bettime = 15,682 rows / 1,125 unique events (cross-checks the audit's
`n_parquet_2023_24_bettime_saves_events: 1125` exactly -- independent
confirmation). New purchase, from `audit_summary.json` (not independently
re-derived, since that requires opening the raw records): the
`combined-2024-25` pass covers 1,313 events; saves present on 1,244;
`betonlineag` saves on 1,050; `prizepicks` saves on 1,139; `underdog`
saves on 0 (SOG only); 1,233 of the 1,244 saves events intersect the
existing 2024-25 closing archive (`n_2024_25_clv_usable_intersection`).
The 1,050 `betonlineag` events from the new pass are Experiment 11's
primary bettime population; whether any overlap with the pre-existing
21-event archive is not established here (checking would require joining
raw event identifiers not opened for this registration) and must be
resolved by the ingestion script's own dedup logic (14.5 rule 2) before
grading, not assumed a priori.

**14.3a Source-population clarification (Codex, 2026-07-14, before the P2
touch).** The first execution attempt passed the section 14.4 wiring gate,
then stopped before model pricing, outcome grading, bootstrap calculation,
or any secondary after finding that 8 of the 11 overlapping event ids had
different old/new requested anchors. This exposed an ambiguity in the final
sentence above: "resolved" did not mean concatenate the old fragment into
the new population and choose an anchor. The first sentence is controlling:
Experiment 11's bettime population is the newly purchased
`core_bettime_202607` pass only. The pre-existing 21-event fragment is not
appended and contributes zero quotes to U; therefore its 11 overlapping ids
cannot double-count the new pass and require no price or anchor tie-break.
The old parquet remains relevant only as provenance for the historical
coverage statement, not as an Experiment 11 input. This clarification is
source-based, was fixed without viewing any P2 statistic, and does not alter
the registered model, venue, threshold, bootstrap, pass bar, or consequence
mapping. The stopped gate-only artifact is retained as an audit record.

**14.4 Wiring gate (mandatory, before any new bettime quote is loaded).**
Reload all three frozen Origin B model files per 14.2, reprice BOTH
variants (each under its own recorded val-fitted alpha) on the EXISTING
2024-25 CLOSING pass (`multibook_classification_training_data.parquet`,
season `20242025`) through the new script's own code path, and reproduce
`experiment_market_state_20260710_213106`'s recorded model-vs-control
paired Brier delta (`brier_vs_control_closing`: mean
`-0.0041404240194266384`, CI95
`[-0.007196770975912929, -0.0011800274158189096]`) to within `1e-4` on
the mean, with n_bets=7,463 and n_clusters=2,510 exact. If it does not
reproduce, STOP and report -- do not load the new bettime pass.

**14.5 Binding ingestion rules for `core_bettime_202607` (global -- also
binding on Experiments 12 and 13 by reference).** Fixed here, before any
price-level record is opened:
1. **Raw-record parsing and pairing.** The raw pass records are NOT in
   the snapshot-parquet schema: they are `core_event=*.json` envelopes
   whose `raw_body` holds the verbatim API response, and the canonical
   parser `scripts/build_odds_snapshots.py` (the script that built
   `saves_lines_snapshots.parquet`) scans a different directory layout
   and keeps only `player_total_saves`. Parsing into snapshot-schema
   rows MUST reuse that script's conventions -- one row per (event,
   snapshot, book, player, side); goalie identity matched by its
   last-name-plus-opponent convention against
   `clean_training_data.parquet`; `goalie_name_raw`/player name kept
   verbatim; no cross-book or cross-side averaging (the odds-averaging-
   bug rule, `docs/HISTORICAL_DATA_ANALYSIS.md` section 1) -- adapted to
   the pass-record envelope and extended to `player_shots_on_goal`.
   Output goes to NEW parquet artifacts; the existing
   `data/processed/saves_lines_snapshots.parquet` is never mutated by
   these experiments' ingestion. Pairing then uses the same generalized
   pairing function Experiment 8 used for its bettime frame --
   `experiment_rolling_origin.build_season_multibook_frame`, which calls
   `clv_audit_pace_policy.pivot_both_sides` -- the concrete
   implementation of the pattern section 11.4 pointed to in
   `scripts/experiment_cross_line_pricing.py`. Both sides required at
   the exact `(event_id, goalie_id, book, line)` key.
2. **Duplicate outcomes.** Drop byte-identical duplicate outcome records
   before pairing (the FanDuel 2023-24 SOG case: 5,280 duplicate pairs
   per `audit_summary.json`'s
   `exact_duplicate_extra_copies_per_season_book_market`).
3. **Conflicting-price groups.** The 3 known FanDuel 2023-24 SOG
   conflicting-price duplicate groups (`duplicate_groups_with_conflicting_
   price: 3`) are excluded entirely, not tie-broken by taking either
   price -- if the raw schema does not cleanly distinguish a
   standard-market entry from an alternate-market echo of the same line,
   abort and report those groups as excluded rather than guess
   (fail-closed, per Experiment 10's precedent). This is 2023-24 SOG
   only, so it cannot affect Experiment 11's own 2024-25 saves universe;
   it is stated once here for Experiment 12 to reference.
4. **Commence-drift handling.** For every event, recompute
   `effective_gap_minutes = (API-returned commence_time) - (requested
   bettime anchor used for that event's call)`, where the anchor is the
   purchase script's own formula (`min(22:30Z on the game's Eastern
   date, cached_commence_time - 30 minutes)`,
   `scripts/purchase_core_bettime_passes.py::compute_bettime_ts`).
   Exclude any event where `effective_gap_minutes < 10`. This excludes at
   least the one event `audit_summary.json` already flags as
   anchor-at-or-after-commence (2024-10-05 BUF@NJD, requested anchor
   14:40:00Z vs. true commence 14:15:13Z, `effective_gap_minutes`
   approximately -24.8 -- excluded a fortiori). The audit's
   `n_drift_gt_5min=80` / `n_drift_gt_30min=3` figures measure a
   DIFFERENT quantity (`|cached_commence - true_commence|`, not the
   anchor-to-commence gap) and are not a substitute count; the exact
   number of additional events excluded by the `<10`-minute rule is not
   yet known and must be computed and reported by the ingestion script
   before any price-level join, not estimated here.
5. **Fanatics.** Expected absent from both new-pass seasons
   (`book_keys_never_seen: ["fanatics"]`, confirmed in the audit); if the
   ingestion script finds any Fanatics row, treat it as a schema surprise
   and report before proceeding, not silently include it.
6. **Pushes.** Excluded from grading per standard convention (saves ==
   line), matching section 11.7's P2 language exactly.

**14.6 Universe, PRIMARY metric, and pass bar (mirrors 11.7 P2 exactly).**
Universe U = paired, gradeable `betonlineag` bettime quotes for Origin
B's 2024-25 test-fold games, built per 14.5 (pushes excluded from
grading). Model arm = UNDER bets where the frozen model's
`EV(UNDER) >= 0.05` (`calculate_ev`'s literal probability-edge
convention -- model probability minus raw vig-inclusive implied
probability, unchanged, per section 11.12's confirmation that this
convention was correctly preserved in Experiment 8). Blind arm = UNDER
on every quote in U. Per goalie-night cluster bootstrap resample (10,000
resamples, seed 42, both arms computed within the same resample draw),
`delta = ROI_model - ROI_blind`. American-odds profit convention: win
pays `odds/100` (positive) or `100/|odds|` (negative); loss = -1.
PASS = CI95 entirely above zero. Resamples with an empty model arm are
counted and reported; `>1%` of resamples empty makes P2 UNSTABLE (no
pass). Fewer than 100 graded model-arm bets makes P2 INSUFFICIENT SAMPLE
(no pass, not a fail -- report and stop there), per section 11.7's
identical construction.

**14.7 SECONDARY metrics (report, no gate).** All-books bettime
selection-over-blind delta (same construction as 14.6, all books instead
of `betonlineag` only); OVER/UNDER ROI splits at the fixed threshold,
both passes (bettime and closing), all-books and BetOnline cuts;
bettime-to-close CLV of the model's flagged bets, net of the
unconditional 2024-25 drift baseline (same matched-quote convention as
Experiment 8/Component G), measurable for the first time this season
since both passes now exist; shots-model signed bias and MAE reused
verbatim from the frozen artifact (`workload_shots_against_test`: mean
bias `+0.5219091642193678`, MAE `5.337973478363781`, n=2,624 -- not
recomputed, since no model is retrained); join coverage by book.

**14.8 Dispersion and sensitivity policy.** Headline results use the
frozen model's validation-fitted dispersion
(`alpha=0.026775577916660614`), matching section 11.9's convention
exactly, not because it is an adopted default. A train-fitted-dispersion
sensitivity pass is produced side-by-side by reloading the same shots
model and refitting `alpha` on `train_idx` via the same shared
`experiments.distributional_saves.fit_dispersion` function, then
repricing test-fold predictions under that alpha (same
reload-and-sanity-check convention as Experiment 8, including the
`<1e-9` reload-parity assertion before either arm is trusted). If P2's
sign flips under train-fitted dispersion, the result is reported as
DISPERSION-FRAGILE (no pass).

**14.9 Forbidden.** No retraining, no feature changes, no hyperparameter
or threshold reselection of any kind -- this is inference against a
frozen artifact, not a new model. No re-running after seeing P2 (one
shot: if the wiring gate passes and the run completes, the first P2
number is the result). No post-hoc slices reported as results. No Odds
API credit or network use. No touching `data/betting.db`.

**14.10 Consequence mapping (fixed in advance, symmetrical to 11.11,
consistent with plan section 6.1).** PASS: the bettime UNDER-selection
mechanism is promoted to 2026-27 shadow-candidacy consideration -- this
is development evidence on a viewed season (per 6.1), not proof of edge,
and must be reported as such even on a pass. FAIL (CI95 not entirely
above zero, with >=100 qualified bets): the Origin B UNDER-selection
effect is treated as NOT demonstrated at the executable venue in either
available season (2025-26 per Experiment 8, 2024-25 per this experiment)
and drops from candidacy; it does not reopen without a new architecture
or a new season of bettime coverage. UNSTABLE: report as a
wiring/sample-structure finding, not a verdict on the mechanism.
INSUFFICIENT SAMPLE: report the all-books secondary and the closing-pass
context, and stop -- this neither promotes nor closes the mechanism.

**14.11 Implemented result -- PASS (Codex-authored and independently
verified, 2026-07-14).** `scripts/experiment_11_frozen_origin_b_p2.py`
first reproduced the frozen closing wiring gate exactly: paired Brier delta
`-0.0041404240194266384`, CI95
`[-0.007196770975912929, -0.0011800274158189096]`, 7,463 rows / 2,510
goalie-night clusters. The first attempt then stopped before model pricing,
grading, bootstrap calculation, or secondaries when the old/new source
ambiguity described in 14.3a surfaced. After 14.3a was recorded, the completed
run used only the new pass and performed the first P2 touch.

PRIMARY BetOnline result: U = 1,719 paired non-push quotes / 1,719
goalie-night clusters; 473 UNDER bets qualified at the frozen 0.05 edge.
Model-arm ROI was `+12.2895%` versus `+2.6329%` from blind UNDER on the same
quote universe, a `+9.6566` percentage-point delta. The registered 10,000-draw
cluster bootstrap (seed 42) gave CI95 `[+2.4891, +16.7200]`, with zero empty
model-arm resamples. This clears the registered PASS bar.

The train-fitted-dispersion sensitivity used alpha `0.023324830664020284`
versus headline alpha `0.026775577916660614`: 477 qualified bets, delta
`+9.4697` points, CI95 `[+2.3506, +16.5206]`. The sign did not flip, so the
result is not dispersion-fragile. Frozen JSON reload parity was exactly zero
for both shots means and save rates.

Registered secondaries agreed on outcomes: all-books selection-over-blind
delta `+10.2195` points, CI95 `[+4.1973, +16.3162]` (1,848 model bets / 6,430
quotes). Bettime all-books policy ROI was `+9.61%`; BetOnline policy ROI was
`+10.77%`, with BetOnline UNDER `+12.29%` and OVER `+7.19%`. CLV is the main
caveat: full-policy CLV net of unconditional drift was positive all-books
(`+0.2718` probability points, CI95 `[+0.2226, +0.3219]`, 2,446 matched), but
BetOnline itself was only `+0.0167` points, CI95 `[-0.0627, +0.0979]` (647),
consistent with zero.

Independent verification reconstructed EV, payouts, model-arm membership,
point ROIs, and the paired cluster bootstrap directly from the persisted
universes and reproduced every primary, all-books, train-dispersion, and CLV
number. Seeds 7 and 20260714 also left the primary lower bound above zero;
this is an unregistered robustness diagnostic, not an additional gate. The
2024-25 closing input contains zero `betonline` alias rows, so the runner's
`{betonlineag, betonline}` closing-secondary mask is numerically identical to
`betonlineag` alone. Reused helper log messages say 2023-24/2025-26 in two
places, but the pinned frames and filters are 2024-25; this is cosmetic.

Artifacts: completed run
`models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/`, including
row-level universes and `output_manifest.json`; retained gate-only stop
`experiment_11_frozen_origin_b_p2_20260714_085614/`. Consequence per 14.10:
promote the mechanism only to 2026-27 shadow-candidacy consideration. This is
encouraging development evidence on an already-viewed season, not proof of a
durable executable edge, particularly because venue-specific CLV did not
confirm it.

---

## 15. Experiment 12 -- W1 cross-market coherence model

Registered 2026-07-14 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, BEFORE any parsing or analysis of the core_bettime_202607
records beyond the persisted audit. This is plan section 10's W1 (NEXT
WAVE block; section 5.7's SOG-purchase rationale).

**15.1 Hypothesis and identity.** The hockey identity
`E[saves] ~= E[opponent team SOG] - E[opponent goals]` is near-exact by
boxscore construction for realized values: `saves = shots_against -
goals_against` holds exactly in 10,425/10,496 = 99.3% of
`clean_training_data.parquet` rows (verified 2026-07-14); 71 rows differ
by exactly 1 (likely shootout-goal bookkeeping, not investigated further
here). The hypothesis is that two INDEPENDENTLY MARKET-DERIVED estimates
of the right-hand side -- opponent team SOG aggregated from listed-skater
SOG props, and opponent goals from moneyline/total -- can be combined
into an implied saves distribution that is sometimes materially
incoherent with the saves market's own quoted line, and that betting only
the incoherent cases beats a blind baseline on the same quote universe.
This is development work, not a re-test of an existing frozen artifact --
no locked recipe exists yet, unlike Experiment 11.

**15.2 What is bound now versus development freedom.** Development
experiments cannot freeze every implementation detail before seeing any
data; the freedom that follows is made explicit rather than assumed.
BOUND NOW: the identity itself (15.1); the data sources (15.3); the
mandatory hazards as constraints, not suggestions (15.4); the global
ingestion rules (15.5, by reference to section 14.5); the freeze protocol
(15.6); and the confirmatory bars for the single 2024-25 touch (15.7),
fixed before any 2024-25 row is loaded. LEFT TO DEVELOPMENT (chosen on
2023-24 data only, and must be written into the frozen artifact per 15.6
before the touch, but the CHOICE itself is not pre-specified here): the
exact coverage-adjustment functional form (e.g. a fixed book-season
multiplier vs. a per-team/per-game regression); the skater-to-team
attribution method for SOG props (which team a listed skater's line
counts toward, and how unresolved or non-dressing skaters are handled --
section 14.5 rule 1 binds only the row-level parsing conventions, not
this attribution step; the W1 probe's roster-archive approach resolved
347/347 sampled names, HISTORICAL_DATA_ANALYSIS.md 9.2, and is a
candidate, not a lock); the incoherence threshold and its units
(probability points vs. saves); the bet rule (fixed-threshold vs.
EV-ranked); the stake convention; which existing distributional-model
components, if any, are reused for the shape-translation step.

**15.2a Execution lock (Codex-authored, 2026-07-14, before W1 development
price/outcome analysis).** The following resolves the remaining implementation
freedom before the development runner is built:

1. Development is chronological: fit/calibrate on `2023-10-10..2024-02-15`
   and select the betting threshold on `2024-02-16..2024-04-18`. Events, not
   book rows, are the split unit. The fitted development models are then
   frozen unchanged; there is no refit on the validation period before the
   2024-25 touch.
2. Development and confirmation are separate runner stages. The development
   stage must use predicate-level 2023-24 reads, assert the maximum input date,
   and persist the complete recipe plus input hashes. The confirmation stage
   must verify that frozen artifact before any predicate-level 2024-25 read,
   write a one-touch marker, and refuse a second completed touch.
3. SOG player-to-team attribution uses the season-team roster index from the
   local play-by-play archive and only the two teams in the event. A player is
   retained when exactly one event team is a season-roster match; unresolved
   or two-team-ambiguous players are excluded and counted. Actual-game roster
   presence may validate identity but may not be used to drop a quoted
   non-dresser.
4. Each sportsbook player's exact paired O/U quote is de-vigged first. Only
   half-point SOG lines enter the model. The fair OVER probability is inverted
   through a Poisson count distribution to an implied player mean, so prop
   medians are not summed as expectations. Player means are summed within
   event-team-book; the cross-book median aggregate and median listed-player
   count form one event-team row. DFS books are excluded.
5. The coverage model is a robust affine development-train regression from
   aggregate player mean plus listed-player count to actual team SOG, one row
   per event-team. No same-game actual coverage fraction is an inference-time
   input. The registered opponent-goals heuristic is reproduced exactly, but
   uses the latest h2h/totals snapshot with `requested_ts <=` the W1 core
   snapshot anchor for that event; a later pregame snapshot is forbidden.
6. Team-SOG projection minus market-implied opponent goals is translated to
   starter saves by a second robust affine regression learned only on the
   development-train goalie outcomes. This is the explicit starter-exposure
   correction: it absorbs average relief/empty-net mismatch without using
   current-game TOI, actual roster participation, or any other postgame input.
   A development-train empirical residual distribution supplies
   `P(over)`, `P(under)`, and `P(push)`; integer saves lines use the conditional
   `P(over)/(P(over)+P(under))` comparison because pushes are void.
7. Eligible saves quotes are exact paired sportsbook O/U records with matched
   SOG projection, timing-safe goals projection, and starter outcome. The unit
   is `(event_id, goalie_id, book, line)`; duplicate signal exposure is handled
   only by the registered goalie-night cluster bootstrap. Flat one-unit stakes
   use the quoted decimal payout. Pushes are excluded from grading.
8. On the development-validation period, each side selects one probability-gap
   threshold from the fixed grid `{0.03, 0.05, 0.07, 0.10, 0.12}`. Selection
   maximizes that side's goalie-night-cluster bootstrap lower CI for
   model-minus-blind ROI delta, subject to at least 100 graded selected bets
   and at most 1% empty-arm resamples; ties choose the larger threshold. If no
   threshold is eligible, freeze `0.05` and label the development side
   insufficient. The full grid is persisted.
9. Overall confirmation semantics: if no side reaches 100 selected bets, the
   experiment is `INSUFFICIENT SAMPLE`. If exactly one side qualifies, that
   side alone determines PASS/FAIL. If both qualify, overall PASS requires
   both to pass; one pass and one fail is `MIXED / NO OVERALL PASS`; both fail
   is FAIL. An `UNSTABLE` qualified side prevents overall PASS.

This lock deliberately favors a small, interpretable market-identity model
over a new high-dimensional feature search. Any alternative architecture is a
new experiment, not an Experiment 12 rerun.

**15.3 Data.** Development (2023-24, viewed per plan section 6.1, so
hypothesis-support-tier evidence only): the new `sog-2023-24` pass (1,312
of 1,313 events have SOG, `audit_summary.json`) plus the existing 2023-24
bettime saves archive (`saves_lines_snapshots.parquet`, 15,682 rows /
1,125 unique events, independently reverified 2026-07-14, section 14.3).
Confirmatory single touch (2024-25): the new `combined-2024-25` pass (SOG
on 1,301 events, saves on 1,244, per `audit_summary.json`) -- one pass
supplies both legs for this season, unlike 2023-24 which needs the SOG
purchase joined to the pre-existing bettime archive. Opponent-goals
estimation reuses the existing consensus-building convention from
`scripts/experiment_market_state_features.py`
(`build_market_state_events`, `attach_market_state_features` --
consensus total = median of the totals point value across books at the
latest pregame snapshot, approximate opponent expected goals =
`consensus_total * opponent_win_prob_devigged`, per that module's own
design notes, already reused by Experiments 5 and 8), joined via
`market_game_features.parquet` (305,940 rows, 2023-10-10 to 2026-04-19,
section 1.1 -- covers both development and confirmatory seasons).
Team-SOG coverage-adjustment CALIBRATION (development only, never
touching 2024-25) uses the existing actual `team_shots` column already
present in `multibook_classification_training_data.parquet` and the
underlying boxscore features of `clean_training_data.parquet`.

**15.4 Known hazards, bound as mandatory model-development constraints
(from plan section 10's NEXT WAVE W1 entry and the W1 probe,
BREAKTHROUGH_MODEL_PLAN.md section 5.7 / HISTORICAL_DATA_ANALYSIS.md
section 9.2, not re-derived here).**
1. SOG props cover only listed skaters -- median 12-15 skaters per event,
   roughly 47%-61% of actual combined team SOG at the book-season median
   (probe-verified, HISTORICAL_DATA_ANALYSIS.md 9.2). The coverage
   adjustment is load-bearing and MUST be estimated on 2023-24
   development data only, never against 2024-25.
2. Prop lines are medians, not means -- do not treat a summed set of
   listed-skater median lines as if it were a summed set of means; any
   aggregation must explicitly account for this or be justified as
   robust to it.
3. Empty-net and backup-relief goals break the identity in the tails --
   `saves = shots_against - goals_against` holds closely for realized
   totals (15.1), but the MARKET-implied opponent-goals estimate (from
   moneyline/total) does not cleanly separate empty-net scoring or a
   relief goalie's shots/goals from the starter's. This is a known
   source of tail error, not a bug to be silently patched away.
4. Book-level SOG coverage breadth is a probe gate, already passed: at
   least two usable books on 8/8 sampled events in every season,
   both-side completeness 98.53%/100%/99.69% across
   2023-24/2024-25/2025-26 (section 9.2) -- broad enough for W1
   development, with the coverage adjustment still the load-bearing open
   question, not a solved one.

**15.5 Global ingestion rules.** Per section 14.5, referenced not
restated: pairing convention, duplicate/conflicting-price handling (the
FanDuel 2023-24 SOG case directly applies here, unlike Experiment 11),
the commence-drift `>=10`-minute rule, the Fanatics-absent expectation,
push exclusion.

**15.6 Freeze protocol.** After development on 2023-24 ONLY, the full
pipeline -- feature construction, coverage adjustment, incoherence
threshold, bet rule, stake convention -- is frozen and written into the
run's persisted metadata (mirroring this document's existing artifact
convention) BEFORE the single 2024-25 touch. The 2024-25 run happens
exactly once. There is no prior frozen recipe to reproduce via a wiring
gate (unlike Experiment 11, which reuses an existing artifact) -- the
freeze event IS this registration's 15.2/15.6 discipline plus the
development-stage artifact it produces; the analogous protection is that
the frozen parameters must be visible in that artifact BEFORE the
2024-25 file is opened, not derived afterward.

**15.7 Confirmatory bars for the 2024-25 touch, preregistered NOW.**
PRIMARY: selection-over-blind delta on the same-side blind baseline
within the identical quote universe -- the Experiment 8 P2 design
(section 11.7) generalized to whichever side the coherence rule selects.
Side handling (a judgment call made here, since the coherence rule can
flag either side depending on incoherence direction, unlike Origin B's
model which only ever selects UNDER): compute the delta SEPARATELY per
side (OVER-flagged vs. UNDER-flagged) against that side's own
blind-every-quote baseline in the same universe; a side's delta is
PRIMARY only if it has >=100 graded model-arm bets, goalie-night cluster
bootstrap (10,000 resamples, seed 42), CI95 entirely above zero to PASS.
If only one side clears the 100-bet floor, that side alone is the
primary result and the other is reported as INSUFFICIENT SAMPLE, not
folded into a pooled number; if neither side clears 100 bets, the
experiment's primary verdict is INSUFFICIENT SAMPLE overall. The same
`>1%` empty-model-arm-resample UNSTABLE rule from section 14.6/11.7
applies per side. SECONDARY: Brier or log-loss of the implied saves
distribution versus the de-vigged saves market (closing and bettime,
both seasons); CLV where pairable against the existing 2024-25 closing
archive (1,233 pairable events per `audit_summary.json`'s
`n_2024_25_clv_usable_intersection`). Per plan section 6.1: passing on
viewed 2024-25 is development evidence for the 2026-27 shadow season,
not proof of edge, regardless of which bars clear.

**15.8 Forbidden.** No 2024-25 looks of any kind during development
(feature construction, threshold selection, coverage-adjustment fitting,
or exploratory plotting). No threshold reselection after the 2024-25
touch. No post-hoc slices (by side, book, or date range) reported as
results once the touch has happened. No credits, no network calls. No
touching `data/betting.db`.

**15.9 Interpretation hierarchy.** As section 0.1: 2023-24 development
results are hypothesis support only. The single locked 2024-25 touch is
stronger chronological evidence but remains development evidence for a
2026-27 shadow candidate (plan section 6.1), not an untouched
betting-edge confirmation -- both 2023-24 and 2024-25 outcomes were
already viewed before this registration (Experiment 5/8's rolling-origin
runs). Final confirmation is the frozen 2026-27 shadow run, out of scope
here.

**15.10 Confirmation-touch failure and recovery registration
(Codex-authored, 2026-07-14, written before any recovery source read).** The
original confirmation touch created its immutable marker at
`2026-07-14T15:56:49Z`, predicate-loaded the five registered 2024-25 sources,
and stopped before saves pricing. Pandas' default CSV float parser changed
the last bits of the frozen residual values, so their canonical semantic hash
did not match. The residual file's byte SHA-256 still matched the development
manifest, and reading it with `float_precision="round_trip"` reproduces the
registered semantic hash exactly.

Per the frozen failure policy, that original touch is consumed and can never
be resumed, deleted, or represented as an Experiment 12 result. It produced
no quote universe, selected bets, grades, bootstrap draws, Brier/log-loss,
CLV, or primary/secondary statistic. The only post-read work completed was
event mapping and roster/SOG construction before the residual guard. The
preserved records are `confirmation_touch.json`,
`confirmation_touch_history.jsonl`, `confirmation_touch_failed.json`, and
`confirmation_20260714_105649/`. **Original-touch verdict: NO VERDICT --
INFRASTRUCTURE FAILURE.**

Because no result or grading information was calculated or observed, one
separately registered recovery touch is authorized as **Experiment 12R**.
This is not a retry under the original marker. Its binding rules are:

1. The development runner, frozen recipe, coefficients, residual file,
   thresholds (`0.03` OVER and `0.03` UNDER), source predicates, primary bars,
   and all interpretation rules remain byte-for-byte or value-for-value
   unchanged. No refit, reselection, slice, or model alteration is allowed.
2. The failed confirmation runner is retained unchanged as evidence. A new
   recovery runner must differ only where needed to (a) load the residual CSV
   with round-trip float parsing, (b) validate both its byte hash and semantic
   hash during a source-free preflight, and (c) use distinct append-only
   `recovery_touch` markers so the consumed original marker is never hidden.
3. Before the recovery marker, preflight must verify every development
   artifact hash, both frozen hashes, the authorization token, the exact
   failed-touch records, the round-trip residual semantic hash, and synthetic
   tests. It must open no 2024-25 source.
4. `recovery_touch.json` is created atomically before the first predicate
   source read. Any failure after that marker consumes Experiment 12R and no
   further historical confirmation is permitted this cycle.
5. The recovery report must disclose both touches. A valid 12R result is
   interpreted under the original section 15.7 bars but carries the disclosed
   operational deviation; it cannot erase the failed first attempt or be
   described as a pristine single execution.

This recovery is permitted solely because the failed run exposed no outcome
grade or model-performance statistic. Had any primary or secondary result
been computed, no recovery touch would be authorized.

**15.11 Implemented result and consumed recovery
(Codex-authored and independently verified, 2026-07-14).** Development
completed on 2023-24 and froze `0.03` for both sides. Neither development
validation interval cleared zero: OVER selected 1,028 quotes and returned a
model-minus-blind delta of `-4.80` points (CI95 `[-14.49, +4.82]`); UNDER
selected 1,293 with delta `+2.94` points (CI95 `[-5.02, +11.14]`). The frozen
recipe and its 17 artifact hashes passed independent verification.

The original confirmation touch then failed exactly as recorded in 15.10,
before any price, grade, or performance statistic was produced. Experiment
12R passed its source-free preflight, created its recovery marker, and
completed the frozen calculations. It then failed while writing final
metadata because the runner requested `preflight["manifest"]` while the
preflight object exposed that record as `development_manifest`. Under 15.10,
this post-marker failure consumes 12R. There is no completion marker, final
metadata, output manifest, or official result JSON, and no further historical
recovery is permitted this cycle.

The persisted unofficial calculation was nevertheless reconstructed
independently, including all 20,000 bootstrap draws, with no numerical
discrepancy. Its 6,429 unique sportsbook quotes covered 2,142 goalie-nights,
with zero duplicate units, pushes, timing violations, or grading/payout
errors:

| Side | Selected quotes | Selected ROI | Blind same-side ROI | Delta | Goalie-night CI95 |
|---|---:|---:|---:|---:|---:|
| OVER | 2,852 | -8.34% | -15.51% | +7.17 points | [+2.73, +11.64] |
| UNDER | 1,597 | +11.12% | +1.84% | +9.28 points | [+2.49, +16.02] |

Both sides therefore met the registered *selection-over-blind* numerical
bars in the observed calculation, but only UNDER was profitable in absolute
terms. Game-level and date-level sensitivity bootstraps retained positive
lower bounds. The model was worse than the de-vigged market over all quotes:
bettime Brier delta `+0.00306`; closing Brier delta `+0.00393`, CI95
`[+0.00057, +0.00737]`; closing log-loss delta `+0.00819`, CI95
`[+0.00123, +0.01531]`. Selected probability CLV was positive but tiny:
approximately `+0.048` percentage points OVER and `+0.093` UNDER; OVER's
decimal-price CLV interval crossed zero.

**Official procedural verdict: Experiment 12R -- NO VERDICT,
INFRASTRUCTURE FAILURE; recovery touch consumed.** Evidentiary
interpretation: the frozen calculation reproducibly meets both registered
selection-over-blind numerical bars and makes this recipe a worthwhile
2026-27 shadow candidate. It is not an official registered PASS, does not
show two profitable sides, does not overcome worse overall calibration, and
does not establish a demonstrated betting edge. The authoritative directory
is `models/trained/experiment_12_w1_cross_market_20260714_104047/`; both
failed-touch records and the `recovery_20260714_111134/` artifacts must be
retained unchanged.

---

## 16. Experiment 13 -- W2 DFS venue-history census

Registered 2026-07-14 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, BEFORE any parsing or analysis of the core_bettime_202607
records beyond the persisted audit. This is plan section 10's W2 (NEXT
WAVE block). Unlike Experiments 11 and 12, this is explicitly a CENSUS,
not an edge search: its bars are about definitional discipline, and any
claim beyond "census finding" requires the same statistical standard
CLAUDE.md sets for this whole project.

**16.1 Question and scope.** Do PrizePicks (and, as a prospective note
only, Underdog) saves lines deviate materially from same-timestamp
sportsbook consensus, and if so, how large and which direction is the
deviating minority, and does it outcome-grade favorably? Scope:
PrizePicks saves for 2024-25 (the new pass) and 2025-26 (existing
archive, verified below); Underdog gets a one-paragraph
prospective-collection note only, per 16.3 -- it has no historical saves
in any season, probe-verified (HISTORICAL_DATA_ANALYSIS.md section 9.3:
"Underdog returned no saves market in either season" of the W1 probe;
confirmed again in the new purchase, `underdog` saves = 0 events in
`audit_summary.json`).

**16.2 Reconciling the 95.2%/90.1% priors -- registered procedure, not a
result.** Two existing analyses report different DFS-vs-consensus
agreement rates and disagree on window, comparator, units, and dedup:
`OFFSEASON_OPTIMIZATION_PLAN.md` section 4.3 (2026-07-07): 95.2% exact
agreement, 248 goalie-nights, Jan-Mar 2026 only, comparator = a SINGLE
"sharp" book (BetOnline Jan-Feb, BetMGM March, never both at once, no
sharp rows at all in April -- by that section's own admission, "sharp
consensus" here is always one book), source = `data/betting.db`.
`HISTORICAL_DATA_ANALYSIS.md` section 8 (2026-07-13): 90.1% exact
agreement, 265/294 ROWS (not goalie-nights), comparator = "sportsbook
consensus" (unspecified aggregation), window unspecified beyond the
recon date; PrizePicks separately at 78.1% (50/64) with the caveat that
"its stored prices are hardcoded placeholders, so only hit-rate, never
ROI, is legitimate there." Both were unpersisted, exploratory scripts
(section 8's own statistical-standard caveat) and, per verification done
for this registration (16.3), both necessarily drew their DFS-side data
from `data/betting.db`, since no processed parquet contains any
PrizePicks row and only a season-20252026 Underdog slice exists in
`multibook_classification_training_data.parquet`.

Before Experiment 13 grades anything, it must register ONE agreement
definition and recompute BOTH prior questions under it as its FIRST
deliverable, so the 95.2%/90.1% gap is explained rather than adopted.
The registered definition:
- **Units of analysis:** one row per goalie-night-book quote (`game_id,
  goalie_id, DFS_book`, one row per calendar day) -- not per re-fetch,
  not per book-pair comparison. This matches the goalie-night cluster
  convention used throughout this document family, rather than either
  prior's row-level or night-level-with-single-comparator convention.
- **Same-timestamp comparator set:** the median line across all
  sportsbooks present in `data/betting.db` (or `saves_lines_snapshots.
  parquet` where a season has processed coverage) for that goalie-night,
  captured within the SAME calendar-day fetch as the DFS quote --
  explicitly NOT a single "sharp" book (correcting section 4.3's
  single-book comparator) and explicitly NOT an unspecified "consensus"
  (correcting section 8's underspecified one). A true same-SECOND
  comparison is not reconstructable for pre-migration data
  (`betting.db`'s `line_snapshots` table has 0 rows before the
  2026-07-09 migration, per HISTORICAL_DATA_ANALYSIS.md section 8's
  operational note) -- this is a real limitation of the underlying data,
  not a modeling choice, and must be stated as such rather than papered
  over.
- **Dedup rule:** if multiple fetches of the same goalie-night-book
  occurred on the same calendar day, use the LAST fetch (closest to game
  time), matching the production pipeline's own daily-overwrite
  behavior.
- **Agreement tolerance:** exact match (line difference = 0.0 saves) as
  the primary definition (matching both priors' apparent convention);
  report a secondary +/-0.5-save tolerance band as a robustness check,
  since half-point lines are common and both priors' own summary
  statistics implicitly carried a wider tolerance's worth of rounding
  (e.g. section 4.3's "range [-1.0, +1.0]" for the deviations that DID
  occur).

Recomputing both prior windows (the Jan-Mar 2026 sharp-comparator sample;
the 2026-07-13 recon's sample) under this single definition, side by side
with the two original numbers, is the FIRST deliverable of Experiment
13's execution -- not performed in this registration, which by
construction has not opened `data/betting.db` or any price-level record.

**16.3 Data, verified 2026-07-14.** New purchase: PrizePicks saves on
1,139 of 1,244 saves-covered events in the `combined-2024-25` pass
(`audit_summary.json`, `key_questions["2024-25_events_with_prizepicks_
saves"]`). Existing 2025-26 archive: independently checked every
processed parquet in `data/processed/` for any PrizePicks row --
`saves_lines_snapshots.parquet`, `multibook_classification_training_
data.parquet`, `classification_training_data.parquet`, `market_game_
features.parquet`, `clv_audit_bets.parquet`, `multibook_frame_2023_24.
parquet`, `multibook_frame_2023_24_bettime.parquet` -- NONE contain a
PrizePicks row, in any season. The only DFS-book presence anywhere in
the processed parquets is `book_key == "underdog"` in
`multibook_classification_training_data.parquet`, season `20252026`,
927 rows, sourced via that build script's `parse_betting_db()` path from
`data/betting.db` (not independently reverified against `betting.db`
itself here, per this task's constraint). This means the 2025-26
PrizePicks comparator that both prior analyses (16.2) and this census
need lives ONLY in `data/betting.db` -- `docs/CURRENT_HISTORICAL_
DATA.md` records 128 PrizePicks rows there as of 2026-07-02 (the season
was still in progress; the current count is unverified here). Reading
`data/betting.db` read-only IS therefore necessary for Experiment 13's
execution -- this is standard practice elsewhere in this repo (section
4.3 itself is a `betting.db` analysis) and is not restricted by this
registration's own "do not touch `data/betting.db`" constraint, which
binds only the writing of this document, not the future authorized
script. That script must snapshot-pin whatever subset of `betting.db` it
reads (row count, max `game_date`, a checksum of the extracted rows)
BEFORE any outcome grading, mirroring Experiment 10's SHA-256
input-pinning discipline, so the census's own input cannot silently
drift between the reconciliation step and the grading step.

A second binding caveat, carried from section 8: `betting.db`'s stored
PrizePicks/Underdog PRICES are hardcoded placeholders, not real market
prices -- only hit-rate and line-comparison analysis is legitimate on
that slice, never ROI. The new purchase's PrizePicks prices ARE real
API-sourced prices, but the W1 probe found no PrizePicks payout
multiplier in any season (the new pass's saves outcomes likewise show
`n_non_null_multiplier: 0`; the W1 probe's 222 non-null multipliers were
exclusively Underdog 2025-26 SOG, HISTORICAL_DATA_ANALYSIS.md 9.3) -- so
even the new pass cannot support a true DFS-parlay ROI calculation; any
ROI-style number reported must use the standard straight-bet
approximation explicitly flagged as such, not real PrizePicks payout
economics.

**16.4 Global ingestion rules.** Per section 14.5, referenced not
restated, for the new-pass records: pairing, duplicate/conflicting-price
handling, the commence-drift `>=10`-minute rule, the Fanatics-absent
expectation, push exclusion.

**16.5 Census deliverables.** Under the section 16.2 registered
definition: deviation rate by season (2024-25, 2025-26) and venue
(PrizePicks only for grading; Underdog descriptive-only per 16.6); size
and direction of deviations (signed, in saves); outcome grading of the
deviating minority with goalie-night cluster bootstrap CIs (10,000
resamples, seed 42); same-timestamp sportsbook-consensus comparison (the
16.2 definition, applied uniformly). Report the majority (agreeing)
population's base rate alongside the minority's, since a census needs
both to be legible.

**16.6 Any-edge-language bar.** This experiment produces census findings
by default. Any claim beyond "census finding" -- i.e. any language
suggesting a bettable deviation -- requires BOTH: (a) goalie-night
cluster-bootstrap CI95 clearing zero on the deviating minority's outcome
grade, AND (b) a chronological split holding (2024-25 develops, 2025-26
confirms, or the reverse if 2025-26's larger sample makes it the
development season -- either order must be stated up front in the
execution script, not chosen after seeing which way it points). Per
CLAUDE.md's standing instruction, a marginal p-value or a small,
non-clustered sample must be reported as statistically weak, not rounded
up. Even a result clearing both bars is development evidence for a
2026-27 shadow candidate (plan section 6.1), never immediate proof of
edge. Underdog is excluded from any edge-language claim entirely --
descriptive note only (927 rows, `book_key=="underdog"`, season
`20252026`, `multibook_classification_training_data.parquet`, no CI, no
ROI, prospective-collection note per plan section 5.7/10 W2).

**16.7 Forbidden.** No threshold or window reselection after seeing
deviation rates. No post-hoc slicing presented as a result once the 16.2
reconciliation and 16.5 census have run. No PrizePicks/Underdog ROI
claim using betting.db's placeholder prices. No credits, no network
calls. No writes to `data/betting.db` (read-only access only, per 16.3).

**16.8 Consequence mapping.** This is a census; "pass/fail" does not
apply the way it does to Experiments 11/12. If the deviation rate is
small and/or the deviating minority does not clear 16.6's bar: close the
census as a null finding, matching Component G's precedent (section 13's
"closed without an outcome touch" disposition) -- DFS venue staleness is
not pursued further this cycle. If 16.6's bar clears on a chronological
split: the deviation-selection mechanism becomes a development candidate
for a filter stacked on model EV (matching the W6 disposition already on
record for the BetOnline convergence lead, plan section 10 NEXT WAVE),
not a standalone strategy, and is queued for its own preregistration
before any confirmatory touch. Either outcome, the 95.2%/90.1%
reconciliation itself (16.2) is retained as a standing correction to
both prior documents' informal numbers, independent of what the rest of
the census finds.

**16.9 Implemented result (Codex-authored and independently verified,
2026-07-14).** `scripts/experiment_13_w2_dfs_venue_history.py` completed the
fixed `2024-25 development -> 2025-26 confirmation` census. For the legacy
tracker slice, the registered last observation was reconstructed as the
maximum SQLite `id` per goalie-night-book. That matches the append-only
production write order, but it is not a wall-clock timestamp or a true
same-second venue comparison; an identical or reverted quote that was not
inserted cannot be recovered.

The prior reconciliation explains why the informal percentages should not
be mixed. Under the registered all-sportsbook-median definition, Underdog's
eligible Jan-March and full persisted samples are both `236/248 = 95.16%`
exact agreement. The old `90.1%` window and aggregation were not persisted
and cannot be reproduced. The full eligible PrizePicks tracker sample is
`51/57 = 89.47%`, not the old unreconstructable `50/64 = 78.1%` slice.

PrizePicks had 1,868 comparable goalie-nights in 2024-25: 1,425 exact
agreements (`76.28%`) and 443 deviations (249 DFS-above-consensus UNDER
candidates; 194 DFS-below-consensus OVER candidates). Of 420 gradeable
non-push deviations, 212 won: `50.48%` hit rate and `+0.95%` even-money
profit per bet, with goalie-night cluster CI95 `[-8.57%, +10.48%]`. This is
an explicitly labeled even-money outcome grade, not PrizePicks ROI. The
2025-26 confirmation slice had only 57 comparable rows and six deviations;
five won, but its six-cluster CI95 was `[0.00%, +100.00%]`, so it did not
clear the strict `lower > 0` bar and is far too small to rescue the null
development result. Underdog remained descriptive-only (`236/248` exact).

**Verdict: CENSUS NULL FOR EDGE LANGUAGE.** The fixed chronological bar did
not clear, so DFS venue staleness is closed for this cycle under 16.8. This
does not say every future PrizePicks disagreement is bad; it says this
historical rule did not demonstrate repeatable selection value. The final
artifact is
`models/trained/experiment_13_w2_dfs_venue_history_20260714_100855/`.
Two construction stops (`100009`, `100036`) occurred before outcome access;
`100623` stopped while writing post-grade reporting; and `100702` is retained
as explicitly invalidated after a denominator audit. Independent verification
matched all 1,875 development and 1,036 legacy normalized rows to source,
found zero consensus or grading discrepancies, reproduced both registered
bootstrap intervals, and confirmed every input hash. One cosmetic audit note
is retained in final metadata: the pre-outcome normalized JSONL still carries
a stale 2024-25 `grade_status` construction label. The graded CSV and summary
are authoritative; the label has no numerical or verdict effect.

---

## 17. Experiment 14 -- W6 BetOnline bettime-to-close convergence

Registered 2026-07-14 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, before any convergence statistic was computed on any season.
This is plan section 10's W6 ("BetOnline convergence-filter policy,"
deferred until the 2024-25 pass landed) and
`HISTORICAL_DATA_ANALYSIS.md` section 8's "BetOnline price convergence"
exploratory lead, now operationalized as a binding registration per that
section's own statistical-standard caveat.

**17.1 Hypothesis and honesty note.** The hypothesis: when BetOnline's
(`book_key`/`book` `betonlineag`) bettime implied probability deviates
from the other books' bettime consensus at the same line, BetOnline's own
price reverts toward that consensus by closing. This must be read against
three honesty constraints, stated plainly rather than rounded up:

1. The discovery statistic (r=-0.147, nominal p=2.0e-10, n=1,851
   correlated quote rows) was computed by an unpersisted scratchpad
   script that no longer exists and was never clustered by goalie-night
   (HISTORICAL_DATA_ANALYSIS.md section 8's own statistical-standard
   caveat, added 2026-07-13). Even this registration's Phase A (17.3) is
   therefore a RE-DERIVATION from raw snapshot rows, not a confirmation
   of the original number -- there is no prior artifact to reproduce a
   wiring gate against, unlike Experiment 11.
2. 2025-26 is the ONLY season with BetOnline bettime coverage in the
   existing archive and is also the discovery season -- the season the
   original scratchpad recon was run on. Any Phase A result is therefore
   IN-SAMPLE with respect to the original lead, not an out-of-sample
   check, even after clustering fixes the inference.
3. 2024-25 outcomes and prices have already been opened repeatedly by
   this document family -- Experiment 11 (section 14) priced and graded
   the frozen Origin B model against the same `combined-2024-25`
   BetOnline bettime pass this experiment's Phase B and 17.5 reuse, and
   Experiment 12 (section 15) additionally viewed 2024-25 outcomes.
   Phase B is therefore DEVELOPMENT EVIDENCE on an already-viewed season,
   exactly as sections 14.10/15.9/16.8 already state for their own
   2024-25 touches, NOT confirmation of edge on an untouched season.

There is no untouched season available for this lead anywhere in this
project's current data. Per plan section 6.1's interpretation hierarchy,
the strongest possible outcome of Experiment 14 is promotion of the
convergence filter to 2026-27 shadow candidacy, stacked on model EV
-- never a standalone strategy and never "edge" language, regardless of
how cleanly every bar below clears.

**17.2 Registered definitions.** Exact and fail-closed, fixed before any
deviation or reversion value is computed on any row:

- **De-vig method.** Proportional (multiplicative) normalization of the
  two-sided quote at the SAME book, SAME line, SAME snapshot pass, SAME
  goalie-night -- never mixing books or sides (the odds-averaging-bug
  rule, `docs/HISTORICAL_DATA_ANALYSIS.md` section 1, restated as
  binding here). From that book's own paired decimal prices:
  `raw_p_over = 1 / price_decimal_over`, `raw_p_under = 1 /
  price_decimal_under`, `overround = raw_p_over + raw_p_under`,
  `p_over_devigged = raw_p_over / overround`, `p_under_devigged = 1 -
  p_over_devigged`. A book contributes a de-vigged probability only when
  BOTH sides of that exact line are present for that book at that
  goalie-night (the existing pairing convention, section 14.5 rule 1) --
  a single-sided quote is not de-vigged and does not contribute.
- **Other-books rule (DFS exclusion, not a fixed book list).** "OTHER
  books" excludes `betonlineag` itself and excludes DFS venues
  (`prizepicks`, `underdog`) by name, not any other book_key --
  whichever non-DFS sportsbook keys are actually present on a given row
  are automatically eligible. DFS venues are excluded because they do
  not run genuine two-sided vig-priced markets: `underdog` carries zero
  saves-market rows in any pass (HISTORICAL_DATA_ANALYSIS.md section
  9.3; reconfirmed in 17.8), and the new pass's own `prizepicks` saves
  quotes were independently found (17.8) to carry simultaneous
  alternate lines on the same side for the same player (e.g. Filip
  Gustavsson `Over 24.5` and `Over 28.5` on the same event) -- a
  different quoting structure than a single fixed-line two-sided market,
  and not a valid consensus input regardless of the separate
  hardcoded-placeholder-price caveat that already applies to
  `betting.db`'s PrizePicks rows (section 16.3).
- **Consensus.** For a given `betonlineag` bettime quote (event,
  goalie-night, side, line), `consensus_p_side = MEDIAN` of the
  de-vigged probability for that side across all OTHER qualifying books
  quoting the EXACT SAME line at bettime for the same goalie-night.
  Require at least 2 other qualifying books at that exact line, else the
  quote is EXCLUDED from the deviation universe; the excluded count is
  recorded and reported (17.3/17.4/17.5), never silently dropped.
- **Deviation metric.** `deviation_under(goalie-night) =
  betonline_p_under_bettime_devigged - consensus_p_under_bettime`
  (UNDER side only, per the side-collapse rule below). "BetOnline is
  stale-high on the Under side" is DEFINED as `deviation_under >
  0` (BetOnline's own bettime de-vigged UNDER probability exceeds the
  same-line other-book median). Exact ties (`deviation_under == 0`) are
  registered as NOT stale-high. This sign is fixed now and is not
  revisited after seeing any Phase A, Phase B, or 17.5 number.
- **Reversion metric.** `reversion_under(goalie-night) =
  betonline_p_under_closing_devigged - betonline_p_under_bettime_devigged`,
  computed ONLY within the PRIMARY universe: pairs where BetOnline's
  saves line is IDENTICAL at bettime and closing for that goalie-night.
  Line-changed pairs are EXCLUDED from every quantitative
  deviation/reversion computation in this experiment and are counted and
  reported as a coverage statistic only (17.8). No secondary
  all-pairs/line-changed analysis will be run in Experiment 14 -- bound
  now. Rationale: translating a probability-space reversion across a
  change in the underlying point line requires a shape/translation
  model (the kind of machinery Experiment 12 built for a different
  purpose); building one here would itself be exactly the kind of
  post-registration feature/model construction 17.6 forbids.
- **Units, clustering, and side convention.** Over and Under of the same
  book/line/goalie-night are mirror images after de-vigging
  (`p_over_devigged = 1 - p_under_devigged` at both BetOnline and
  consensus), so carrying both sides as independent observations
  double-counts every goalie-night. Registered convention: keep exactly
  ONE observation per goalie-night -- the UNDER side -- for every
  deviation/reversion statistic in Phase A, Phase B, and 17.5. Rationale:
  this matches Origin B's UNDER-only construction (section 14.2),
  Experiment 11's UNDER-only P2 universe (section 14.6), and the live
  betting record's structural UNDER lean (section 0.1) -- an established
  project convention, not an ad hoc choice for this experiment. A
  goalie-night whose UNDER-side quote fails the >=2-other-books gate or
  the line-identical requirement is excluded outright and is never
  backfilled by its OVER-side mirror. On top of this one-row-per-
  goalie-night collapse, inference is a goalie-night cluster bootstrap,
  10,000 resamples, seed 42, resampling goalie-night clusters with
  replacement -- matching sections 14.6/15.7/16.5's convention exactly.
- **Dedup (multiple snapshots per pass).** Verified against the actual
  parquet row structure (17.8), not assumed:
  - `data/processed/saves_lines_snapshots.parquet` (Phase A's sole
    source, and Phase B's closing-side source): within a single
    `snapshot_pass` label, duplicate rows sharing `(event_id,
    goalie_name_raw, book, side)` do occur -- most densely in 2025-26
    `bettime` (914 of 11,897 natural-key groups, a 7.68% group rate;
    188 of 2,474 `betonlineag`-only groups, a 7.60% group rate) and
    negligibly in 2025-26 `closing` (4 of 18,216 groups, 0.02%; zero
    among `betonlineag`-only closing rows) and in both 2024-25 passes
    (zero duplicate groups in either). Registered rule: within each
    `snapshot_pass` label, for rows sharing the natural key, keep only
    the row with the MAXIMUM `resolved_ts` (the latest actually-observed
    API response timestamp among fetches bucketed into that pass); ties
    broken by maximum `requested_ts`, and any remaining tie broken
    deterministically by original row order, with the tie count logged.
    Rationale: `resolved_ts` is the actually-observed wall-clock capture
    time, versus `requested_ts`'s nominal target -- consistent with this
    document family's existing preference for observed timestamps over
    nominal ones (section 14.5 rule 4's use of API-returned
    `commence_time` over the requested anchor is the precedent). This
    rule applies uniformly to both `bettime` and `closing` buckets.
  - `data/processed/core_bettime_202607_snapshots.parquet` (Phase B and
    17.5's bettime source): the `betonlineag` saves population itself
    has ZERO within-pass duplicate `(event_id, player_name_raw,
    book_key, side)` groups (3,498 rows / 3,498 groups) and exactly one
    `requested_ts`/`fetched_at` per event -- no dedup is needed for the
    `betonlineag` side of this parquet. The all-books saves population
    has 23 duplicate-key groups, entirely attributable to `prizepicks`
    alternate-line offerings (already excluded by the DFS rule above),
    not duplicate fetches of the same market. If the runner nonetheless
    encounters a within-pass duplicate on a qualifying non-DFS book, the
    same max-`resolved_ts` rule above applies by reference.

**17.3 Phase A -- clustered re-derivation on 2025-26.** Source:
`data/processed/saves_lines_snapshots.parquet`, season 2025-26 (games
with `game_date_eastern` in the 2025-26 hockey season), `snapshot_pass`
in `{bettime, closing}`, `book == betonlineag` for the deviation/
reversion target plus qualifying other books for consensus, all per
17.2's definitions and dedup rule. Build `deviation_under` and
`reversion_under` per goalie-night, restricted to the PRIMARY (line-
identical bettime-to-close) universe, gated by the >=2-other-books-at-
the-same-line requirement, UNDER side only.

Registered statistic (PRIMARY): cluster-bootstrap CI95 (10,000
resamples, seed 42, resampling goalie-night clusters) of the Pearson
correlation `r` between `deviation_under` and `reversion_under` across
goalie-nights. Pearson `r` is PRIMARY because it matches the original
discovery statistic's own units; an OLS slope
(`reversion_under ~ deviation_under`) is computed and reported as a
SECONDARY diagnostic only and does not gate the result.

PASS bar: CI95 entirely BELOW zero (replicating the original lead's
negative-correlation direction, now with clustered inference). If the
CI includes zero, Phase A is CLOSED and Phase B / 17.5 still run per
17.7 but are labeled EXPLORATORY-ONLY in the final report, not
confirmatory of anything. Degenerate-resample rule (the Pearson-`r`
analogue of section 14.6's empty-model-arm rule): any bootstrap draw
in which `r` is undefined (zero variance in the resampled
`deviation_under` or `reversion_under` vector) is counted as a
degenerate resample; if more than 1% of the 10,000 draws are
degenerate, report the Phase A result as UNSTABLE alongside the CI,
not as a clean pass or fail. Also report, unconditionally: n
goalie-nights in the PRIMARY universe; n excluded by the
>=2-other-books gate; n excluded as line-changed; and the degenerate-
resample count.

**17.4 Phase B -- second-season replication on 2024-25.** Bettime side:
`data/processed/core_bettime_202607_snapshots.parquet`,
`pass_name == "combined-2024-25"`, `market_key == "player_total_saves"`,
`book_key == betonlineag` for the deviation target plus qualifying other
non-DFS books at the same line, per 17.2. Closing side: the EXISTING
archive, `data/processed/saves_lines_snapshots.parquet`, season 2024-25,
`snapshot_pass == "closing"`, `book == betonlineag` (3,912 rows / 1,094
events, 17.8) -- the new purchased pass carries bettime rows only
(verified in 17.8; this is consistent with the task design, not a
contradiction requiring a stop) and cannot supply a closing side itself.

**11-event-overlap dedup (bound now, identical to Experiment 11's
14.3a resolution).** The pre-existing 21-event 2024-25 `bettime`
fragment already inside `saves_lines_snapshots.parquet` (season 2024-25,
`snapshot_pass == "bettime"`, 258 rows) contributes ZERO rows to Phase
B's bettime population. Only the newly purchased `combined-2024-25` pass
is used for the bettime side. "New pass wins" is operationalized as
TOTAL EXCLUSION of the old fragment, not a row-level merge or an
anchor-level tie-break -- this is the exact clarification 14.3a made
after Experiment 11's first attempt stopped mid-run over 8 of the 11
overlapping event ids having different old/new requested anchors;
registering it identically here means that ambiguity cannot recur.

Join key between the new-pass bettime rows and the existing-archive
closing rows for the same goalie-night: `event_id` plus resolved goalie
identity (`goalie_id` where both sides' pipelines resolved it; otherwise
`goalie_name_matched`/`goalie_name_raw` fallback), matching Experiment
11's own wiring-gate join convention (section 14.4) -- not re-derived
here.

Registered statistic, bar, and degenerate-resample rule: identical to
17.3 (Pearson `r`, 10,000-resample seed-42 goalie-night cluster
bootstrap, CI95 entirely below zero to PASS, OLS slope secondary, >1%
degenerate-resample draws makes it UNSTABLE). Also report,
unconditionally: n goalie-nights with both a new-pass bettime and an
existing-archive closing `betonlineag` row; n in the PRIMARY (line-
identical) universe; n excluded by the >=2-other-books gate; n excluded
as line-changed.

**17.5 EV-stacked filter test (2024-25 only, persisted frozen artifacts,
no model rerun).** Reuse, unchanged and read-only,
`models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/
p2_primary_betonlineag_universe.parquet` (verified in 17.8: 1,719 rows /
135 columns, one row per goalie-night quote, `cluster_id` already
1-to-1 with rows; 473 rows have `is_model_arm == True` at the frozen
`ev_under >= 0.05` threshold; `ev_under`, `is_model_arm`,
`profit_if_under`, `cluster_id`, `event_id`, `goalie_id`, `game_id`,
`book_key`, `betting_line`, `odds_over_decimal`, `odds_under_decimal`
all present). No repricing, no retraining, no re-selection of the 473.

Population: the 473 `is_model_arm == True` rows only. For each, compute
`deviation_under` per 17.2 (deviation ONLY -- 17.5 needs no reversion
and no closing-side line-identity check, since it tests agreement with
a bet placed AT bettime, not a bettime-to-close comparison), using the
SAME `combined-2024-25` pass as Phase B, joined to the frozen universe
by `(event_id, goalie_id)`, with `book_key == betonlineag` verified and
the frozen `betting_line` checked against the deviation metric's own
BetOnline bettime line. Any row where the two lines do not match exactly
is excluded and counted, not silently trusted. Any of the 473 lacking a
computable `deviation_under` (fewer than 2 other qualifying books at
BetOnline's exact bettime line) is excluded from the filter test and
counted, not imputed.

**Sign convention (registered now, bound regardless of which way the
split later favors either arm).** "Agree-arm" = model-arm UNDER bets
where `deviation_under(goalie-night) > 0`, i.e. BetOnline is stale-high
on the Under side per 17.2. "Non-agree arm" = model-arm UNDER bets where
`deviation_under <= 0` (computable, not stale-high; includes exact
zero). Rows lacking a computable `deviation_under` are excluded from
BOTH arms, not folded into non-agree. Rationale for why stale-high
is registered as "agreeing" with a model-arm UNDER bet, stated for
auditability, not as an empirical claim: a model-arm UNDER bet already
means the frozen model finds UNDER underpriced against BetOnline's raw
bettime price. If BetOnline is ALSO already stale-high on Under (its own
price is currently LESS favorable to an UNDER bettor than the sharp
consensus would justify, not more), the model's edge estimate survives
despite, not because of, a currently generous BetOnline number -- a
stronger case for the edge being real rather than an artifact of
BetOnline's own temporary underpricing. This reasoning is not verified
by any of the data examined for this registration; it motivates the
choice but does not substitute for the result.

Metrics: PRIMARY = ROI delta, agree-arm ROI minus the FULL 473-bet
model-arm ROI, using the persisted `profit_if_under` values verbatim (no
repricing), cluster-bootstrapped (10,000 resamples, seed 42) over the
agree-arm's own `cluster_id` values. SECONDARY = the identical
construction for the non-agree arm vs. the full model arm; plus each
arm's CLV where matchable, reusing
`models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/
flagged_bets_with_clv.parquet` (verified in 17.8: 2,539 rows / 140
columns, includes `bet_side`, `clv_prob`, `clv_prob_net_of_drift`,
`bettime_devig_prob_chosen_side`, `closing_consensus_prob_chosen_side`)
joined to each arm's bets by `(event_id, goalie_id, book_key)` -- this
file has more rows than `p2_primary` (2,539 vs. 1,719), so the runner
must join-filter it down to each arm's exact keys, never assume row
alignment. No fresh CLV computation.

Honesty bar (100-bet minimum per arm, registered per the task's own
instruction): any arm with fewer than 100 graded bets is
INSUFFICIENT SAMPLE for that arm specifically, not a fail; if both arms
fall below 100, the whole 17.5 test is INSUFFICIENT SAMPLE. Stated
plainly: with only 473 total model-arm bets split two ways, and further
trimmed by the >=2-other-books coverage gate, INSUFFICIENT SAMPLE in one
or both arms is a realistic outcome, not a remote one, and must be
reported as such rather than the split being redrawn to avoid it. Even a
clean pass (both bars clear, agree-arm delta CI95 entirely above zero)
is development evidence only, on an already-viewed 2024-25 season (per
17.1) -- it cannot by itself promote anything if Phase A fails (17.7).

**17.6 Forbidden.**

1. No retraining, no repricing of the frozen Origin B model or its
   inputs, no feature or threshold search beyond the exact definitions
   registered in 17.2.
2. No touching `data/betting.db` -- reads are forbidden too, unlike
   section 16.3's carve-out for Experiment 13; this experiment does not
   need that table and may not open it.
3. No Odds API credit spending, no network calls.
4. No modification of any pre-existing file in `models/trained/`,
   including the Experiment 11 artifact directory reused here --
   read-only reuse only.
5. No writes to any existing parquet (`saves_lines_snapshots.parquet`,
   `core_bettime_202607_snapshots.parquet`, or any file already under
   `data/processed/` or `models/trained/`). Any new output goes to a new
   artifact directory only, following this document family's existing
   convention (e.g. `models/trained/experiment_14_.../`).
6. No post-hoc slicing (by book, date range, or threshold) reported as
   a result once Phase A, Phase B, or 17.5 have been computed.
7. No changing the 17.2 side convention or the 17.5 stale-high sign
   convention after seeing any deviation, reversion, correlation, or ROI
   number -- both are fixed by this registration.
8. One registered execution. If the runner crashes mid-run, it may be
   fixed and rerun ONLY if NO phase's registered statistic (Phase A's
   `r`/CI, Phase B's `r`/CI, or 17.5's ROI-delta CIs) was yet computed
   and printed or logged; otherwise the computed phases' numbers stand
   as-is and must be reported. Explicit clarification, stated so it is
   not misapplied later: 2024-25 is already-viewed development data
   (17.1; plan section 6.1; Experiments 11 and 12's own disclosures),
   NOT a virgin confirmatory touch -- so section 15.10's touch-
   consumption / one-shot-marker recovery machinery (built to protect a
   genuinely untouched season) does NOT apply to Phase B or 17.5. A
   Phase B or 17.5 crash before its statistic is computed may simply be
   fixed and rerun under this same registration, with no 12R-style
   recovery sub-registration required, precisely because there is no
   virgin touch to protect here.

**17.7 Consequence mapping (fixed in advance).** Phase A AND Phase B
BOTH PASS (CI95 entirely below zero on both) -> the convergence filter
is registered as a 2026-27 shadow-candidate filter stacked on model EV,
joining the Experiment 11 and Experiment 12 shadow candidates already on
record (sections 14.11, 15.11) -- it is NOT promoted to live betting and
is NOT edge language, regardless of how the 17.5 EV-stack result lands.
EITHER phase FAILS (CI95 includes zero) -> the lead is CLOSED this
cycle, matching the steam-recon (section 8's first bullet) and
DFS-census (section 16.8) precedents; per the task's own instruction,
17.5 is still computed and reported in this case but is labeled
EXPLORATORY-ONLY and cannot reopen or promote the lead on its own.
17.5 INSUFFICIENT SAMPLE (either or both arms) does not by itself close
or promote the lead -- it is reported as a scale finding. UNSTABLE
(either phase's degenerate-resample rate exceeds 1%) is reported as a
methods/sample-structure finding, not a verdict, mirroring section
14.10's UNSTABLE handling.

**17.8 Data inventory (verified 2026-07-14).** Read-only Python against
the two parquets and the Experiment 11 artifact directory; no
deviation, reversion, correlation, ROI, or outcome-linked quantity was
computed.

`data/processed/saves_lines_snapshots.parquet`: 79,884 rows / 15
columns; `snapshot_pass` values `{bettime: 28,751; closing: 51,133}`;
`book` values (8, no DFS venue present anywhere in this parquet):
`barstool, betmgm, betonlineag, bovada, draftkings, fanatics, fanduel,
williamhill_us`; `side` values are exactly `{Over, Under}`. Rows by
`snapshot_pass` x season: `bettime` 2023-24 15,682 / 2024-25 258 /
2025-26 12,811; `closing` 2023-24 17,959 / 2024-25 14,954 / 2025-26
18,220. `betonlineag`-only rows by the same cut: `bettime` 2024-25 76 /
20 events, 2025-26 2,662 / 725 events; `closing` 2024-25 3,912 / 1,094
events, 2025-26 3,926 / 1,046 events -- this independently reconfirms
the task's stated 2024-25 closing figures (3,912 rows / 1,094 events)
exactly. 2025-26 bettime books present with row counts: `betmgm` 2,964,
`draftkings` 2,902, `betonlineag` 2,662, `bovada` 2,377, `fanduel`
1,878, `fanatics` 28 (`barstool`/`williamhill_us` do not appear in this
specific cut). Some events carry two distinct `requested_ts` values
within the same `bettime` label (max 2, mean 1.087 per event across 781
2025-26 bettime events) -- the source of the dedup finding in 17.2,
inspected on one example event (`01060baa72bf643120a46cda7c3e04c1`,
`requested_ts` `2026-04-02T22:30:00Z` and `2026-04-03T00:00:00Z`) where
prices were mostly stable across the two fetches with a few single-cent
drifts. Phase A join-coverage check (structural line-match count only,
not a deviation/reversion value): after the 17.2 dedup rule, 2025-26
`betonlineag` bettime rows 2,662 -> 2,474 deduped; closing rows 3,926 ->
3,926 (no change). Joined `(event, goalie, side)` pairs with both a
deduped bettime and closing `betonlineag` row: 2,063, of which 1,927
(93.4%) are line-identical (the PRIMARY-eligible pairs) and 136 (6.6%)
are line-changed; 1,032 unique goalie-nights have both a bettime and a
closing `betonlineag` quote on at least one side.

`data/processed/core_bettime_202607_snapshots.parquet`: 413,758 rows /
23 columns; `pass_name` values `{combined-2024-25, sog-2023-24}`;
`season` values `{2024-25, 2023-24}`; `snapshot_pass` is `bettime` ONLY
(zero closing rows anywhere in this parquet -- confirmed, consistent
with the task's design that Phase B's closing side must come from the
existing archive, not a contradiction); `market_key` values
`{player_shots_on_goal, player_total_saves}`. `player_total_saves` rows:
16,820 / 1,244 events, all `season == "2024-25"` -- matches the task's
and `CURRENT_HISTORICAL_DATA.md` section 4.2's stated 1,244 saves
events exactly. `betonlineag` saves rows: 3,498 / 1,050 events --
matches the task's stated 1,050 exactly. Saves `book_key` values:
`betmgm, bovada, betonlineag, prizepicks, williamhill_us, draftkings`
(no `fanatics`, consistent with the expected-absent rule in section
14.5 rule 5; no `underdog`, consistent with its zero-saves finding; no
`barstool` in this specific saves cut). `betonlineag` saves rows carry
zero within-pass duplicate-key groups and exactly one `requested_ts`/
`fetched_at` per event (1,050 events, both stats identically 1.0 mean /
1.0 max) -- no dedup needed for the `betonlineag` population itself. The
23 duplicate-key groups found in the all-books saves population are
entirely `book_key == prizepicks` (e.g. Filip Gustavsson `Over 24.5` and
`Over 28.5` on the same event) -- alternate-line offerings, not
duplicate fetches, and moot under 17.2's DFS exclusion.

Experiment 11 artifact directory
`models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/`
exists and contains `bettime_frame_allbooks.parquet`,
`bettime_predictions.parquet`, `bettime_snapshot_rows_new_pass_only.
parquet`, `flagged_bets_with_clv.parquet`, `gate_predictions.parquet`,
`metadata.json`, `output_manifest.json`,
`p2_primary_betonlineag_universe.parquet`, `p2_allbooks_universe.
parquet`, `run_log.txt`. `p2_primary_betonlineag_universe.parquet`:
1,719 rows / 135 columns; `is_model_arm` is `True` for exactly 473 rows
and `False` for 1,246; `cluster_id` has 1,719 unique values (already
1-to-1 with rows, confirming the artifact's "quote" unit is already
one-row-per-goalie-night, matching this registration's own UNDER-only
collapse); `ev_under`, `is_model_arm`, `profit_if_under`, `cluster_id`,
`event_id`, `game_id`, `goalie_id`, `book_key`, `betting_line`,
`odds_over_decimal`, `odds_under_decimal` are all present, exactly as
the task described. `flagged_bets_with_clv.parquet`: 2,539 rows / 140
columns, including `bet_side`, `devig_prob_over`, `devig_prob_under`,
`consensus_prob_over`, `consensus_prob_under`, `n_closing_books`,
`closing_consensus_prob_chosen_side`, `bettime_devig_prob_chosen_side`,
`clv_prob`, `clv_prob_net_of_drift` -- larger than `p2_primary` (2,539
vs. 1,719 rows), so 17.5's CLV secondary must join-filter this file down
to each arm's exact `(event_id, goalie_id, book_key)` keys rather than
assume row alignment (recorded in 17.5 itself, restated here as a
verified structural fact).

No contradiction of any assumption in the task brief was found: the
2025-26 archive has both `betonlineag` bettime and closing rows at
usable scale; the 2024-25 closing archive figures (3,912 rows / 1,094
events) match exactly; the new pass's `betonlineag` saves figures (1,050
events) match exactly; and the Experiment 11 artifact has every column
the task named. The one genuine surprise -- within-`bettime`-pass
duplicate fetches in the EXISTING `saves_lines_snapshots.parquet` at a
7.68% natural-key group rate in 2025-26 bettime -- was not previously
documented anywhere
in this document family and required the fresh dedup rule registered in
17.2.

**17.9 Implemented result -- LEAD CLOSED (Sonnet sub-agent execution
under lead-reviewer direction, independently verified, 2026-07-14).**
`scripts/experiment_14_w6_betonline_convergence.py` completed both
phases and 17.5 under the exact 17.2 definitions. All 11 pre-statistic
structural reconciliation checks against 17.8's registered counts
passed exactly, independently recomputed by the lead reviewer from the
persisted row-level universes and matching to the row: 2,662 -> 2,474
dedup; 2,063 joined bettime/closing pairs (1,927 line-identical, 136
line-changed); 1,032 goalie-nights; 3,498 rows / 1,050 events for the
new-pass `betonlineag` saves population; 1,719 rows / 473 model-arm
bets in the reused Experiment 11 universe.

Phase A (2025-26 clustered re-derivation): Pearson `r =
-0.05019347165148147`, cluster-bootstrap CI95
`[-0.10550077988050377, +0.005427384587194252]`, n = 931 goalie-nights
(one row per night, PRIMARY universe). OLS slope secondary
`-0.07148410669176927`. Zero degenerate resamples of 10,000. Exclusion
funnel from 1,237 paired `betonlineag` bettime UNDER quotes: 60 failed
the >=2-other-books gate, 186 had no closing-side quote at all, 60 were
line-changed, leaving 931 primary. The CI includes zero -> **Phase A
verdict CLOSED**. Context that must stay attached to this number: the
original scratchpad discovery statistic was `r = -0.147` (nominal
`p = 2.0e-10`) on 1,851 unclustered correlated quote rows; under the
registered definitions (goalie-night units, UNDER-side collapse,
within-pass dedup, >=2-book same-line consensus) the same season yields
roughly one third that magnitude and does not clear zero. The lost
scratchpad's exact definitions are unrecoverable, so how much of the
shrinkage is pseudo-replication versus definitional difference cannot
be decomposed.

Phase B (2024-25 second-season replication, new-pass bettime joined to
the existing closing archive): `r = -0.05829157793567338`, CI95
`[-0.12429245812046397, +0.0048848362618009]`, n = 1,380 goalie-nights.
OLS slope secondary `-0.04590003547182036`. Zero degenerate resamples.
Exclusion funnel from 1,749 new-pass `betonlineag` UNDER quotes: 302
failed the books gate, 23 had no closing data, 44 were line-changed,
leaving 1,380 primary. The CI includes zero -> **Phase B verdict
FAIL**. The sign was negative in both seasons (a consistent direction)
but neither CI95 excludes zero at the registered bar -- that is
reported here exactly as that, not rounded up to a near-miss.

17.5 EV-stacked filter test (EXPLORATORY-ONLY per 17.7, since Phase A
did not pass): of the 473 frozen Experiment 11 model-arm bets, 102
lacked a computable `deviation_under` (books gate) and were excluded,
leaving agree-arm n = 260 (`deviation_under > 0`) and non-agree n = 111.
Full-473 reference ROI `+12.2895%`. Agree-arm ROI `+3.1282%`, delta
`-9.1613` points, CI95 `[-20.384, +1.761]` -> **FAIL**. Non-agree-arm
ROI `+21.4765%`, delta `+9.1871` points, CI95 `[-7.688, +25.661]` ->
**FAIL**. CLV secondary: agree-arm mean `clv_prob_net_of_drift =
-0.00398` (252 of 260 matched), non-agree-arm `+0.00528` (107 of 111
matched). Both arms' CIs cross zero; the numeric ordering ran OPPOSITE
to the registered 17.5 sign rationale, but per 17.6.7 the convention is
not revisited after the fact, and per 17.7 an exploratory 17.5 cannot
reopen or promote anything regardless. The non-agree arm's `+21%` is
not a lead -- it is an n=111 exploratory subsample on an already-viewed
season with a CI spanning `[-7.7, +25.7]`.

Disclosed judgment calls (from the run's own `metadata.json`,
summarized faithfully): goalie-night identity for within-file grouping
used `goalie_name_raw` directly where 17.2's dedup key names it
literally, while the cross-parquet `goalie_key` (falling back through
`goalie_id`, `goalie_name_matched`, `goalie_name_raw`) used for
consensus grouping and the Phase B join was verified to reproduce the
identical structural counts either way, so no ambiguity resulted; the
runner added its own `n_excluded_no_closing_data_available` count
(distinct from "line-changed," which presupposes a closing quote
exists) so each phase's exclusion funnel sums exactly to its
population, since 17.3/17.4 name only the books-gate and line-changed
buckets explicitly; 17.5's ROI-delta bootstrap resamples only the arm
under test against the FIXED (non-resampled) full-473 ROI rather than
Experiment 11's paired-both-arms convention (14.6), read as unambiguous
from 17.5's own "fixed reference" language rather than as a flagged
deviation; 17.5's per-arm PASS/FAIL bar (CI95 entirely above zero,
gated by the 100-bet floor) was inferred from 17.5's own descriptive
aside since no bar is named explicitly the way 17.3/17.4 do; and every
bootstrap used `numpy.random.default_rng(42)` per this task's own
instruction, where section 17 pins the seed and resample count but not
the RNG class, and where Experiments 11 and 13 had used
`np.random.RandomState` instead.

Registered consequence (17.7): either phase's CI95 includes zero, so
the W6 BetOnline convergence lead is **CLOSED this cycle**, matching
the steam-recon (section 8) and DFS-census (section 16.8) precedents.
No shadow-candidate registration follows. It does not reopen without a
new architecture or a new season of BetOnline bettime coverage.
Artifacts:
`models/trained/experiment_14_w6_betonline_convergence_20260714_142506/`.

---

## 18. Experiment 15 -- W3 saves-market microstructure feature block (juice skew)

Registered 2026-07-14 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, before any microstructure feature value was computed on any
season. This is `BREAKTHROUGH_MODEL_PLAN.md` section 10 NEXT WAVE's W3
entry (zero credit): "Market-microstructure feature block. Juice skew
... plus related bettime-observable price-shape features, tested as
model inputs against the no-pace control under Gate-A-style bars," and
`HISTORICAL_DATA_ANALYSIS.md` section 8's "Juice skew" exploratory
lead, operationalized as a binding registration per that section's own
statistical-standard caveat (added 2026-07-13). Architecturally this
experiment is a direct sibling of Experiment 5 (section 6, the
`control_plus_market_state` block) -- same variant mechanics, same
script lineage (`scripts/experiment_market_state_features.py`) --
extended with a second, book-agnostic feature family instead of the
market-game-state one.

**Amendment (2026-07-14, lead-reviewer directed, still before any
microstructure feature value was computed on any season).** As first
registered, this section gated on Origin A and Origin B agreeing.
The lead reviewer independently verified the structural disclosure in
18.1 item 4 (Origin A's train+val pool, 2022-10-07 to 2023-04-14,
predates the entire bettime saves archive, which starts 2023-11-02;
Experiment 5's own Origin A market-state result never beat control
there: `brier_vs_control_closing` mean `+0.000537819408872642`, CI95
`[-0.0001351793045926375, +0.0012345894289141777]`) and ruled that a
registration must not gate on an origin that cannot train the
features -- disclosure of a structurally unclearable bar is not
enough. The project's own precedent resolves this: the two origins on
which the market-state block actually demonstrated a CI-excluding-zero
improvement are Origin B (Experiment 5, section 10 table row 5) and
Origin C (Experiment 8, section 11.12 P1 PASS). Under this amendment
the GATING origin set is **{Origin B, Origin C}**; Origin A is
reclassified as a registered PLACEBO / NEGATIVE CONTROL (18.3a),
explicitly non-gating. Everything below is written in the amended
form; no feature value, correlation, Brier, or ROI had been computed
when the amendment was made, so this is a pre-execution registration
change, not a post-result revision.

**18.1 Hypothesis and honesty note.** Saves-market microstructure at
bettime (juice skew and price-shape) carries predictive information
about the saves outcome beyond the no-pace control's features. This
must be read against the same kind of honesty constraints section
17.1 stated for BetOnline convergence, restated for this lead
specifically:

1. The discovery statistics (same-book same-line over/under price
   asymmetry predicts the outcome at r=0.032 overall; 2023-24 r=0.039,
   nominal p=0.0005; 2025-26 r=0.028, nominal p=0.019) were computed by
   an unpersisted scratchpad script that no longer exists and was
   never clustered by goalie-night (`HISTORICAL_DATA_ANALYSIS.md`
   section 8's own caveat: "rows share goalie-nights across books, so
   these p-values are inflated"). There is no prior artifact to
   reproduce a wiring gate against for the discovery statistic itself
   -- only Experiment 5's architecture is reproducible, not the
   original juice-skew number. Whatever this experiment measures is a
   RE-DERIVATION under new, stricter, feature-block definitions
   (18.2), not a confirmation of r=0.032.
2. All three test seasons are already-viewed. 2024-25 outcomes
   (Origin B's test fold) have been opened repeatedly by this document
   family (Experiments 5, 8, 11, 12, 14 all touched 2024-25 in some
   form). 2023-24 (Origin A's placebo test fold) has been a viewed
   development fold since the rolling-origin experiment (section 0.1).
   2025-26 (Origin C's test fold, per the amendment) is the worst of
   the three in honesty terms: it is simultaneously the juice-skew
   DISCOVERY season (the original scratchpad recon ran on it) and the
   live-bet season -- doubly viewed. Any Origin C result is therefore
   IN-SAMPLE with respect to the original juice-skew lead, exactly the
   caveat section 17.1 item 2 attached to Experiment 14's Phase A on
   the same season, even though no MODEL in this repo was ever trained
   or validated on 2025-26 (Origin C's folds, 18.3). There is no
   untouched season available for this lead anywhere in this project's
   current data. Per plan section 6.1's interpretation hierarchy, the
   strongest possible outcome of this experiment is promotion of the
   block to 2026-27 shadow candidacy for a future model rebuild --
   never confirmation of edge on an untouched season.
3. The standalone-bet result is already known and is NOT being
   re-tested: betting the skew-favored side loses -6.98% ROI against a
   ~7.1% average vig, negative in every bucket tested
   (`HISTORICAL_DATA_ANALYSIS.md` section 8). This experiment tests
   only whether the underlying price-shape signal improves the
   distributional saves MODEL as an input feature -- a materially
   different and weaker claim than "this beats the vig on its own,"
   and section 18.5 computes no betting-policy ROI at all.
4. A structural fact that shaped the amended architecture: Origin A's
   train+val pool lies entirely inside the 2022-23 season (2022-10-07
   to 2023-04-14), and the saves-market bettime archive this block's
   training-season features come from (`saves_lines_snapshots.parquet`)
   does not begin until 2023-11-02 (18.8) -- fully seven months after
   Origin A's pool ends. Origin A's `control_plus_microstructure`
   shots model therefore has 0% real training exposure to any `juice_*`
   value, exactly the same structural problem Experiment 5's
   `control_plus_market_state` block had on Origin A (section 6's
   module docstring: "the tree has no non-missing training exposure to
   split on"), and exactly why Experiment 5's block failed its own
   both-origins bar despite a real, CI-excluding-zero Origin B result.
   Per the 2026-07-14 amendment, Origin A therefore does NOT gate this
   experiment: it is a registered PLACEBO / NEGATIVE CONTROL (18.3a)
   whose paired deltas estimate the procedure's noise floor, and the
   gating set is {Origin B, Origin C} -- the two origins on which
   market-state actually passed (Experiment 5 Origin B; Experiment 8
   Origin C P1), both of which have real `juice_*` training exposure:
   Origin B 29.79% of train rows / 86.81% of val rows, Origin C 49.34%
   of train rows / 80.89% of val rows (verified join counts, 18.8).

**18.2 Registered feature block.** Fixed here, before any feature
value is computed on any row. All features derive ONLY from bettime
`player_total_saves`-market quotes for the goalie's own saves market
(no `player_shots_on_goal` rows, no closing-pass quotes, no
cross-market machinery -- W1's cross-market coherence model, section
15, owns that territory and is closed to further historical touches).
All features are book-agnostic aggregates across whichever qualifying
books are present on a given goalie-night -- never keyed to
`betonlineag` or any other single book -- because `betonlineag` is
verified absent (zero rows, both `bettime` and `closing`) from the
entire 2023-24 `saves_lines_snapshots.parquet` archive (18.8): a
venue-specific feature would be an all-NaN column throughout training
on that origin's test fold, and this document family already flagged
the identical problem for Component G (section 10, "Consequence for
Component G": "that book has zero quotes in the archive for that
season").

*De-vig method (reused verbatim from section 17.2, restated as binding
here).* Proportional (multiplicative) normalization of the two-sided
quote at the SAME book, SAME line, SAME bettime snapshot, SAME
goalie-night -- never mixing books or sides (the odds-averaging-bug
rule, `HISTORICAL_DATA_ANALYSIS.md` section 1). From a qualifying
book's own paired decimal prices: `raw_p_over = 1 / price_decimal_over`,
`raw_p_under = 1 / price_decimal_under`, `overround = raw_p_over +
raw_p_under`, `p_under_devigged = raw_p_under / overround`. A book
contributes only when BOTH sides of the exact same line are present
for that book at that goalie-night (14.5 rule 1's pairing convention,
reused via `experiment_rolling_origin.build_season_multibook_frame` /
`clv_audit_pace_policy.pivot_both_sides`, unchanged); a single-sided
quote is not de-vigged and does not contribute anything to any
feature below -- not to a median, not to `juice_n_books`, not to the
modal-line computation (17.2's precedent, restated as binding).
DFS venues (`prizepicks`, `underdog`) are excluded from the book
universe at every step, per 17.2's Other-books/DFS-exclusion rule,
restated as binding here: they do not run genuine two-sided vig-priced
markets (`underdog` carries zero saves rows in any pass; the new
pass's `prizepicks` saves quotes carry simultaneous alternate lines on
the same side for the same player, 17.8/18.8).

*Modal line.* For a goalie-night with at least one qualifying
(two-sided) book, `L*` = the line value quoted by the most qualifying
books; ties among tied-for-most lines are broken by the LOWEST
numeric line value, fixed now and never revisited after seeing any
feature value. "At-modal" qualifying books = the subset whose own line
equals `L*`.

*The six registered features plus one indicator (within the 5-8
range; `juice_matched` is registered as an indicator, not counted
toward the 5-8, mirroring how `mkt_matched` was not counted among
Experiment 5's "7 `mkt_*`"):*

1. `juice_p_under_consensus` -- MEDIAN of `p_under_devigged` across
   at-modal qualifying books. This is the skew itself: >0.5 means the
   book-median devigged price currently favors UNDER being priced as
   more likely than a fair coin at that line; <0.5 the reverse. No
   separate centered ("minus 0.5") column is registered -- an XGBoost
   tree splits identically on `p` or `p - 0.5`, so a second derived
   column would be redundant fishing surface, not a new degree of
   freedom.
2. `juice_overround_median` -- MEDIAN of `overround` across at-modal
   qualifying books (vig magnitude at the consensus line).
3. `juice_p_under_dispersion` -- population standard deviation
   (`ddof=0`) of `p_under_devigged` across at-modal qualifying books;
   `0.0` (NOT NaN) when exactly one at-modal qualifying book exists,
   mirroring `mkt_h2h_dispersion`/`mkt_total_dispersion`'s
   `.fillna(0.0)` convention exactly (`build_market_state_events`,
   section 6's script) -- a single observation has zero empirical
   spread by construction, which is a fact about the data, not a
   missing value.
4. `juice_n_books` -- COUNT of qualifying (two-sided) books at the
   goalie-night, ANY line (not restricted to the modal line) --
   overall market depth, independent of line agreement.
5. `juice_line_dispersion` -- population standard deviation (`ddof=0`)
   of the LINE value across qualifying books at the goalie-night, ANY
   line; `0.0` when only one distinct line is quoted or only one
   qualifying book exists (same `fillna(0.0)` convention as #3). This
   averages LINE point values across books, not odds/probabilities --
   explicitly NOT the odds-averaging bug, per Experiment 5's own
   module docstring precedent (section 6's script, point 3: "Averaging
   LINE values across books is not the odds-averaging bug ... which
   was arithmetic averaging of vig-inclusive American odds").
6. `juice_line_minus_baseline` -- `L* - saves_rolling_5`, where
   `saves_rolling_5` is the goalie's own trailing-5-game rolling saves
   average, a column already present in the no-pace-control 104-column
   shots feature list (verified in `experiment_market_state_
   20260710_213106/metadata.json`'s `feature_sets.no_pace_control`,
   18.8) -- a plain arithmetic difference between two already-existing
   pregame-known quantities, expressible without any new modeling
   machinery, per the task's own instruction that this candidate is
   admissible "only if expressible without new modeling machinery."
   NaN if `L*` is undefined (no qualifying book) OR `saves_rolling_5`
   itself is NaN (goalie has fewer than 5 rolling starts of history) --
   the second condition is inherited from an existing column, not
   newly introduced here.
7. `juice_matched` -- 0/1 indicator, `1` iff the goalie-night has at
   least one qualifying (two-sided, non-DFS) bettime saves quote at
   any line (i.e., `L*` is defined and features 1-5 are computable),
   mirroring `mkt_matched` exactly.

*Registered missing-data rule.* Features 1, 2, 3, 6 are NaN wherever
`juice_matched == 0`; feature 6 is additionally NaN wherever
`saves_rolling_5` is NaN even if `juice_matched == 1`. For features 4
and 5, `juice_matched == 0` is reachable only through "no qualifying
book," and the binding rule in that case is: `juice_n_books` is `0` (a
real count, not missing) when no qualifying book exists;
`juice_line_dispersion` is NaN in that same case (no line data exists
to compute a spread over). All NaN routing uses XGBoost's native
missing-value handling; nothing is ever imputed, mirroring Experiment
5's convention exactly (module docstring point 5: "A market feature is
left as NaN ... Never silently imputed").

*Fail-closed rules for degenerate cases.* One-sided quotes (a book
posts only Over or only Under, not both) contribute NOTHING to any
feature -- not counted in `juice_n_books`, not eligible for modal-line
selection, not part of any median or dispersion (17.2's precedent,
restated as binding). Single-book nights are NOT excluded --
`juice_matched = 1` and features 1, 2, 6 are computed normally from
that one book's own de-vigged pair; only the dispersion features (3,
5) fail closed to `0.0` rather than NaN or undefined, per the
`fillna(0.0)` convention above. Zero-qualifying-book nights:
`juice_matched = 0`, features 1, 2, 3, 6 all NaN, feature 4 = `0`,
feature 5 = NaN (no line data). `betonlineag` receives no special
handling anywhere in this block -- it is one qualifying book among
however many are present on a given night, which is exactly what
makes the block trainable on 2023-24 without an all-NaN column.

**18.3 Architecture and origins (amended 2026-07-14).** Three rolling
origins. Origins A and B mirror Experiment 5 exactly; Origin C mirrors
Experiment 8 (section 11.3) exactly. All three are carved with
`experiment_rolling_origin.py`'s `carve_origin_split`/`season_date_
range`, `VAL_WINDOW_DAYS=49` (val = final 49 days of each origin's
pool date range). Origin A/B boundaries cross-checked against
`experiment_market_state_20260710_213106/metadata.json`'s
`fold_boundaries`; Origin C boundaries registered BY VALUE from
`models/trained/experiment_market_state_origin_c_20260713_140706/
metadata.json`'s `origin_c_fold_boundaries` (18.8):

| | Origin A (PLACEBO, 18.3a) | Origin B (GATING) | Origin C (GATING) |
|---|---|---|---|
| Pool | 2022-10-07 to 2023-04-14 | 2022-10-07 to 2024-04-18 | 2022-10-07 to 2025-04-17 |
| Train | 2022-10-07 to 2023-02-24 (1,864 rows) | 2022-10-07 to 2024-02-29 (4,528 rows) | 2022-10-07 to 2025-02-27 (7,134 rows) |
| Val | 2023-02-25 to 2023-04-14 (760 rows) | 2024-03-01 to 2024-04-18 (720 rows) | 2025-02-28 to 2025-04-17 (738 rows) |
| Test | season 2023-24 (2,624 rows) | season 2024-25 (2,624 rows) | season 2025-26 (2,624 rows) |

Origin C pool = seasons 2022-23 + 2023-24 + 2024-25, exactly as
Experiment 8 registered it (11.3: "validation = final 49 days of the
pool date range; train = the rest. Test = season 20252026"). Same
hyperparameter search space and procedure for Origin C as for A/B
(`SHOTS_CONFIGS`/`SAVE_RATE_CONFIGS` unchanged, val-MAE /
weighted-log-loss selection), same val-fitted dispersion convention
(one alpha per variant, fit on `val_idx` residuals) -- all identical
to what Experiment 8's own run recorded for its two Origin C variants
(winner config `shallow_highreg` for both; val-fitted alphas
`0.027718386433224374` control / `0.026939644644863918` market-state,
18.8).

**Leakage guard for Origin C (registered explicitly).** Origin C
variants are evaluated ONLY on Origin C's own registered test fold --
season 2025-26, strictly after its train+val window -- and NEVER on
the 2024-25 season in any form: the ENTIRE 2024-25 season lies inside
Origin C's train+val (train ends 2025-02-27, val ends 2025-04-17), so
any 2024-25 evaluation of an Origin C model would score the model on
its own training/validation data. This is the standing reason
Experiment 11 was required to reuse the frozen ORIGIN B artifact for
its 2024-25 P2 test, never Origin C (section 14.2's fold rationale:
"a season the frozen model never trained or validated on"). Origin
B's 2024-25 test fold and Origin C's train window overlap by
construction -- that is a property of the rolling-origin design shared
with Experiments 5/8, not leakage, because no statistic ever compares
an Origin C prediction against a 2024-25 outcome.

Four shots-model variants are trained per origin, all sharing ONE
save-rate model trained once per origin on the no-pace-control feature
list (mirrors section 6's "shared by both variants... literally trained
once, not twice," extended from 2 variants to 4):

- `no_pace_control` -- the literal 104-column control feature list,
  RETRAINED FRESH within this experiment's own script run (not read
  from the frozen artifact directly), same `SHOTS_CONFIGS`/`SAVE_RATE_
  CONFIGS` grid and val-MAE/weighted-log-loss selection procedure,
  unchanged (`src/experiments/distributional_saves.py`'s
  `train_shots_model`/`train_save_rate_model`, both with `random_
  state=42` fixed in the reused code, verified at lines 366/419).
  Needed both as the PRIMARY comparison baseline and as one leg of the
  mandatory wiring gate below.
- `control_plus_microstructure` -- `no_pace_control`'s 104 columns
  plus the 6 `juice_*` features plus `juice_matched` (7 new columns,
  111 total shots features), added to the SHOTS-AGAINST model ONLY.
  Verified this is where `mkt_*` actually went for Experiment 5 (not
  assumed): `experiment_market_state_20260710_213106/metadata.json`'s
  `feature_sets.no_pace_control` has 104 entries, and `origin_a`/
  `origin_b`'s `control_plus_market_state` variant's
  `shots_feature_count` is 112 (104 + 7 `mkt_*` + `mkt_matched`); the
  save-rate model's `feature_cols` in both origins' `rate_model` block
  is the 104-column no-pace list, confirming it is shared and
  unchanged across variants. No surprise or contradiction found here.
- `control_plus_market_state` -- RETRAINED FRESH (not reused from
  either frozen artifact), 112-column list unchanged (104 + 7 `mkt_*` +
  `mkt_matched`), same grid/procedure. Retraining, rather than reusing
  `experiment_market_state_20260710_213106`'s (Origins A/B) or
  `experiment_market_state_origin_c_20260713_140706`'s (Origin C)
  frozen shots-model JSONs directly, is registered because a fair
  PAIRED comparison (18.5) requires all variants' predictions to come
  from the SAME script execution on the SAME row indices with the SAME
  bootstrap draws -- exactly the property Experiment 5's own internal
  `paired_brier_delta_vs_variant`/`paired_shots_mae_delta` functions
  already assume for its two variants. This is explicitly allowed by
  the task's own instruction ("training is not rationed; grading is")
  and is bound by the mandatory wiring gate below rather than trusted
  blind.
- `control_plus_market_state_plus_microstructure` -- the 112-column
  `control_plus_market_state` list plus the 7 `juice_*`/`juice_matched`
  columns (119 total shots features), same placement, same shared
  save-rate model, its own val-fitted dispersion.

**Mandatory wiring gate (before any microstructure quote is loaded).**
The runner must first retrain `no_pace_control` and `control_plus_
market_state` fresh on ALL THREE origins through this experiment's own
code path and reproduce the recorded numbers to within `1e-4` on every
mean (the same tolerance Experiments 11/14 used; because
`random_state=42` is fixed and no feature or data change is involved,
bit-identical reproduction is actually expected, and `1e-4` is
registered as the acceptance bar, not evidence that looser
reproduction is tolerated).

From `experiment_market_state_20260710_213106/metadata.json`
(Experiment 5):
- Origin A `brier_vs_control_closing`: mean `0.000537819408872642`,
  CI95 `[-0.0001351793045926375, 0.0012345894289141777]`, n_bets=8,880,
  n_clusters=2,298.
- Origin B `brier_vs_control_closing`: mean `-0.0041404240194266384`,
  CI95 `[-0.007196770975912929, -0.0011800274158189096]`, n_bets=7,463,
  n_clusters=2,510.
- Origin A `shots_mae_delta_vs_control`: mean `0.009711408033603576`,
  CI95 `[-0.0032097097758839767, 0.02256776296147486]`, n=2,624.
- Origin B `shots_mae_delta_vs_control`: mean `-0.07375802354114812`,
  CI95 `[-0.12168046450469552, -0.023554074546185938]`, n=2,624.

From `experiment_market_state_origin_c_20260713_140706/metadata.json`
(Experiment 8):
- Origin C P1 paired Brier delta vs control (closing, val-fitted
  headline): mean `-0.003111099251412182`, CI95
  `[-0.005038647618850264, -0.0011919729523759111]`, n_bets=5,729,
  n_clusters=2,070, n_push_excluded=0.
- Origin C `no_pace_control` shots workload on test: mean bias
  `+0.23460506447931614`, MAE `5.407952885075313`, n=2,624.
- Origin C `control_plus_market_state` shots workload on test: mean
  bias `+0.42204831794994635`, MAE `5.3598914618899185`, n=2,624.

Registered caveat on the Origin C gate targets, so the runner is not
later accused of inventing one: Experiment 8 did NOT record an
Experiment-5-style paired shots-`|error|`-delta-vs-control CI for
Origin C (its `secondaries.shots_bias_mae` carries only the
per-variant point values gated on above), so the Origin C wiring gate
reproduces the P1 Brier delta (mean to `1e-4`, n_bets and n_clusters
exact) plus both variants' workload bias/MAE point values (each to
`1e-4`); the fresh paired shots-delta CI this experiment computes on
Origin C is a NEW statistic with no reproduction target, and is gated
only prospectively by 18.5's bar.

If ANY origin's reproduction fails, STOP and report -- do not load any
`juice_*` quote. This mirrors the Experiment 11 (14.4) / Experiment 8
(11.5) wiring-gate convention exactly, extended to all three origins.

Dispersion: val-idx-fitted NB2 alpha per VARIANT (four separate fits
per origin, one per shots-variant, each via `fit_dispersion(shots_
model, df_full, val_idx, ...)`), matching Experiment 5's dispersion
procedure exactly (module docstring point 6: fit on validation
residuals, not training residuals, per the Experiment 3 correction,
section 10.1 item 1). Recorded reference alphas from the frozen
artifacts for the wiring gate's two retrained variants: Origin A
`no_pace_control` alpha `0.033265130249687844`, `control_plus_market_
state` alpha `0.03312151062457906`; Origin B `no_pace_control` alpha
`0.02852173299997726`, `control_plus_market_state` alpha
`0.026775577916660614`; Origin C `no_pace_control` alpha
`0.027718386433224374`, `control_plus_market_state` alpha
`0.026939644644863918`. `ORIGIN_CAP=90`, fixed EV threshold `0.05`
(unused for any policy decision in this experiment, carried only
because `join_and_price`/`grade_bets` are reused unchanged), goalie-
night cluster bootstrap 10,000 resamples seed 42 throughout, matching
section 1's shared conventions exactly.

**Correction on the task's framing of Experiment 8's bar.** The
original task brief states "the block must beat the control on BOTH
origins to PASS (that is what Experiments 5 and 8 required)." This is
verified TRUE for Experiment 5 (section 6.4's bar, and the actual
computed `pre_registered_pass_bar.overall_pass` in
`experiment_market_state_20260710_213106/metadata.json`, which
literally contains a `passes_on_both_origins` boolean per metric). It
is NOT accurate for Experiment 8 as stated: Experiment 8 (section 11)
is a SINGLE-origin replication (Origin C only) whose registered bar
was "P1 AND P2 both pass" on that one origin (section 11.7), not
"both origins" -- there was only one origin to test. Under the
2026-07-14 amendment, this experiment retains Experiment 5's
TWO-ORIGIN-AGREEMENT structure but re-bases it onto {Origin B, Origin
C} -- which happens to make the gating set exactly the pair of origins
on which the market-state block's improvement was actually
demonstrated (Experiment 5's Origin B row; Experiment 8's Origin C
P1). Flagged here as a correction rather than silently designing
around the imprecise premise.

**18.3a Origin A: registered PLACEBO / NEGATIVE CONTROL (added by the
2026-07-14 amendment, non-gating).** Origin A runs the identical
four-variant pipeline as B and C (same shared save-rate model, same
grid, same val-fitted alphas, same metrics computed), but its results
CANNOT pass or fail this experiment. Rationale, restated from 18.1
item 4: Origin A's training pool has ZERO `juice_*` exposure (the
archive starts 2023-11-02, seven months after its pool ends), so its
`control_plus_microstructure` shots model cannot have learned anything
from the block -- every `juice_*` column is all-NaN across its entire
train+val, and XGBoost's trees have no non-missing value to split on.
Its paired deltas vs control on the 2023-24 test fold therefore
estimate the PROCEDURE'S NOISE FLOOR: the spread injected purely by
adding seven never-informative columns to the feature matrix (tree
tie-breaking, column subsampling under `colsample_bytree=0.8`, and
downstream dispersion refitting), with no possible signal content.

Registered placebo expectation: both Origin A paired-delta CI95s
(shots `|error|` delta and closing Brier delta,
`control_plus_microstructure` minus `no_pace_control`) CONTAIN zero.
Registered interpretation rule, fixed now: if either Origin A placebo
CI95 EXCLUDES zero in EITHER direction, that is a red flag for a
pipeline bug or evaluation artifact (feature leakage through the join,
row misalignment, dispersion contamination, or an evaluation-code
defect), and the runner must STOP-AND-INVESTIGATE before any Origin B
or Origin C result is trusted or any verdict is issued -- the placebo
cannot pass or fail the experiment itself, and a placebo anomaly does
not become a "finding" about microstructure under any reading. The
SECONDARY comparison (`control_plus_market_state_plus_microstructure`
minus `control_plus_market_state`) is ALSO computed on Origin A and is
part of the same placebo readout under the same anomaly rule (both
variants have zero `juice_*` training exposure there, so this delta is
equally signal-free by construction); it is not dropped, because a
second placebo statistic doubles the wiring-check surface at zero
marginal cost.

**18.4 Feature data sources and joins.** 2023-24 bettime features
(feeding Origin A's placebo test fold, Origin B's train-window
2023-24 portion, and Origin C's train-window 2023-24 portion):
`data/processed/saves_lines_snapshots.parquet`,
`snapshot_pass == "bettime"`, season 2023-24 (`game_date_eastern`
2023-10-10 to 2024-04-18) -- 15,682 raw rows / 1,125 events (18.8).
Apply section 17.2's max-`resolved_ts`-within-pass dedup rule by
reference (natural key `(event_id, goalie_name_raw, book, side)`);
18.8 additionally verifies this pass carries no DFS venue at all, so
the DFS-exclusion rule is trivially satisfied for 2023-24 without
dropping anything.

Test-season (2024-25) bettime features for Origin B: `data/processed/
core_bettime_202607_snapshots.parquet`, `pass_name == "combined-2024-
25"`, `market_key == "player_total_saves"` -- 16,820 raw rows / 1,244
events (18.8), non-DFS subset (`book_key != "prizepicks"`) for feature
construction per 18.2's DFS-exclusion rule. Per section 17.4's
11-event-overlap-dedup precedent (identical to 14.3a's resolution),
the pre-existing 21-event 2024-25 `bettime` fragment already inside
`saves_lines_snapshots.parquet` (258 rows, season 2024-25,
`snapshot_pass == "bettime"`) contributes ZERO rows to this
experiment's 2024-25 bettime population -- TOTAL EXCLUSION of the old
fragment, not a row-level merge or anchor tie-break, bound now by
reference to 17.4/14.3a exactly as section 17.4 itself bound it.

Origin B's closing-pass and (new, for the first time in this document
family) bettime-pass BETTING/GRADING universes for the Gate-A-style
metrics (18.5) are built the same way Experiment 11 already built its
own bettime frame from this exact purchased pass
(`bettime_frame_allbooks.parquet`,
`models/trained/experiment_11_frozen_origin_b_p2_20260714_090012/`):
reuse `experiment_rolling_origin.build_season_multibook_frame`
(which calls `clv_audit_pace_policy.pivot_both_sides`) unchanged,
rather than re-deriving new plumbing, per 14.5 rule 1's own citation
of this pattern.

Outcomes: `data/processed/clean_training_data.parquet` ONLY, `saves`
column. `data/betting.db` is FORBIDDEN, reads included (mirrors 17.6
item 2's stricter carve-out, not 16.3's read-allowed exception).
Goalie matching: reuse established conventions (`goalie_id` where
resolved; last-name-plus-opponent matching per `scripts/build_odds_
snapshots.py` for any unresolved rows) by reference, per 14.5 rule 1;
join key `(event_id, goalie_id)` with `attach_game_id`'s `(goalie_id,
game_date_eastern)` `+/-1`-day fallback (`clv_audit_pace_policy.py`),
by reference.

**2024-25 new pass: dual role, registered explicitly.** Under the
amended architecture the `combined-2024-25` pass serves TWO distinct
roles: (a) Origin B's TEST-fold feature source (above), and (b) Origin
C's TRAIN- and VAL-window feature source for its 2024-25 rows (train
window rows with `game_date_eastern <= 2025-02-27`; val window rows
2025-02-28 to 2025-04-17). The same parquet, the same dedup and DFS
rules, split by date against Origin C's registered fold boundaries.
This dual use is a property of the rolling-origin design (Origin B's
test season is inside Origin C's pool by construction, exactly as in
Experiments 5/8) and is not leakage under the 18.3 leakage guard,
because no Origin C prediction is ever scored against a 2024-25
outcome.

**Test-season (2025-26) bettime features for Origin C (amended
2026-07-14; the original registration excluded 2025-26 entirely, the
amendment brings it in as Origin C's test fold).** Source:
`data/processed/saves_lines_snapshots.parquet`, `snapshot_pass ==
"bettime"`, 2025-26 season window (`game_date_eastern` 2025-10-07 to
2026-04-19) -- 12,811 raw rows / 781 events, books `{betmgm,
draftkings, betonlineag, bovada, fanduel, fanatics}` (18.8; no DFS
venue exists anywhere in this parquet, so the DFS rule is trivially
satisfied). Section 17.2's max-`resolved_ts` within-pass dedup rule
applies -- this is the exact pass 17.2's rule was originally written
for (914 of 11,897 natural-key groups duplicated, 7.68%), and 18.8
records the dedup's row effect (12,811 -> 11,897). Coverage honesty,
registered up front: this archive covers only 781 of the season's
events (the live tracker fetched bettime snapshots only on days the
pipeline ran), so Origin C's test-fold `juice_matched` coverage is
structurally the lowest of the three origins -- 53.12% verified
(18.8), clearing 18.5's registered 50% COVERAGE-INSUFFICIENT floor,
but narrowly; that floor is NOT being adjusted to accommodate this
fact, and if a production run's join conventions land below 50%, the
floor triggers as registered. The in-sample honesty caveat attached to
any Origin C result (2025-26 is the juice-skew discovery season and
the live-bet season, 18.1 item 2) travels with every Origin C number.

Origin C's closing-pass grading universe:
`multibook_classification_training_data.parquet`, season `20252026`
(5,729 paired quotes / 2,070 goalie-night clusters, per Experiment 8's
recorded P1 -- section 11.4's inventory: 2025-10-07..2026-04-13). Its
bettime grading universe for the SECONDARY reporting is built from the
deduped 2025-26 snapshot rows via the same `build_season_multibook_
frame`/`pivot_both_sides` path as the other origins (Experiment 8's
own recorded bettime frame was 5,763 rows / 1,370 clusters, 18.8).

**18.5 Registered metrics and Gate-A-style bars.** PRIMARY universe:
CLOSING pass, mirroring Experiment 5's own precedent -- verified
directly from `experiment_market_state_20260710_213106/metadata.json`:
the actual computed `pre_registered_pass_bar` dict contains only
`brier_vs_control_closing` and `shots_mae_delta_vs_control` keys; there
is no bettime entry in the pass bar anywhere (Origin A's bettime frame
was priced only as an unconditional `price_passes["bettime"]` entry,
never fed into `pre_registered_pass_bar`; Origin B's Experiment 5 run
had no bettime frame available at all, section 1.2). Experiment 8's
Origin C P1 was likewise gated on the CLOSING pass ("model probability
at each posted line, closing pass, all books," section 11.7), with
bettime confined to its P2/secondaries -- so the closing-primary
convention holds independently on both frozen precedents. CLOSING is
therefore PRIMARY here too, by direct mirroring, not a fresh choice.

Two metrics per origin, both mirrored from Experiment 5's own two
pass-bar metrics, computed on ALL THREE origins for
`control_plus_microstructure` MINUS `no_pace_control`:
- (a) paired shots `|error|` delta, goalie-night cluster bootstrap
  (10,000 resamples, seed 42), same `paired_shots_mae_delta`
  construction (negative = microstructure variant more accurate).
- (b) paired per-quote Brier delta on the CLOSING quote universe, same
  `paired_brier_delta_vs_variant` construction, goalie-night cluster
  bootstrap (negative = microstructure variant better).

PASS on a given metric requires CI95 upper bound `< 0` on BOTH GATING
origins -- Origin B AND Origin C -- independently. This retains
Experiment 5's `passes_on_both_origins` two-origin-agreement logic
verbatim (verified in `metadata.json`: Experiment 5's actual computed
`overall_pass` was `False` because neither metric's `ci_excludes_zero_
improvement` was `True` on both of ITS origins simultaneously -- its
Origin A failed both metrics, its Origin B passed both, 18.8),
re-based per the 2026-07-14 amendment onto the two origins that can
actually train the block. Origin A's deltas are computed and reported
unconditionally but are the 18.3a placebo readout -- they enter no
pass computation. **PRIMARY PASS** (`control_plus_microstructure` vs
`no_pace_control`) = TRUE if EITHER metric clears BOTH Origin B and
Origin C, exactly mirroring `metadata.json`'s
`overall_pass = any(...)` logic verbatim.

**Origin C scale honesty, registered in advance.** Origin C's
registered universes are smaller than Origin B's but of the same order
-- verified from Experiment 8's recorded artifacts, not guessed:
closing 5,729 paired quotes / 2,070 goalie-night clusters (vs. Origin
B's 7,463 / 2,510), bettime 5,763 rows / 1,370 clusters, shots-delta
n=2,624 (identical row count on every origin's test fold). (For the
record: Experiment 8's much-quoted 85 was its P2 EV-THRESHOLD-QUALIFIED
BetOnline bettime UNDER count -- a betting-policy universe this
experiment does not use anywhere, since 18.5 computes no policy ROI;
it does not describe the Brier universes above.) Bound now: if either
registered PRIMARY CI on Origin C is too wide to exclude zero --
whether from scale, from the test fold's lower `juice_matched`
coverage (53.12%, 18.8), or from genuine absence of signal -- that is
a FAIL under the bar as registered, not an excuse, and no
post-hoc "underpowered, rerun with more data" reframing is available
within this registration.

BETTIME is reported as a SECONDARY, unconditionally, on all three
origins (Origin B's is newly possible in this document family, since
Experiment 5 had no 2024-25 bettime frame at all; Origin C's mirrors
Experiment 8's own bettime frame) -- identical paired-delta
construction against the bettime quote universes built per 18.4, no
gate.

**SECONDARY comparison** (the incremental-over-current-best question,
registered per the task's own instruction): `control_plus_market_
state_plus_microstructure` MINUS the RETRAINED `control_plus_market_
state` (18.3), identical construction (both metrics, Origins B and C;
Origin A's copy is part of the 18.3a placebo readout), CLOSING primary
+ bettime secondary, reported unconditionally, NO PASS/FAIL gate on
this document's PRIMARY consequence mapping (18.7).
It answers whether microstructure is REDUNDANT with the already-
promoted market-state block (no further CI-excluding-zero gain on
either metric on either origin) or ADDITIVE (a further gain survives
on top of market-state) -- informs, does not gate, promotion.

No betting-policy ROI is computed anywhere in this experiment -- this
is a feature-block gate only, exactly matching how Experiment 5 itself
reported policy ROI only as an unconditional secondary under section
6.3, never as part of its own pass bar. Betting-policy testing (EV
threshold selection, bet-level ROI, CLV) of any block promoted here is
explicitly a SEPARATE future registration, not covered by this one --
mirroring the family's existing separation between feature-gate
registrations (Experiment 5, this one) and policy registrations built
on top of an already-gated block (Experiment 8's P1-vs-P2 split,
Experiment 11's entirely separate P2 registration).

**Coverage floor and UNSTABLE/degenerate rules.** The cluster-
bootstrap MAE/Brier deltas here cannot produce the kind of degenerate
(undefined-statistic) resample that Experiment 14/17's Pearson-`r`
bootstrap could -- `cluster_bootstrap_mean_ci` always returns a
defined mean as long as at least one cluster is drawn. The registered
degenerate case for THIS experiment is instead a coverage floor,
applied to the GATING origins (B and C): if a gating origin's
TEST-fold `juice_matched` rate is below 50% (a coverage floor, not a
p-value bound -- chosen because a feature block with less than half
real-quote exposure on its own test fold cannot be interpreted as
evidence about the features regardless of which way the point estimate
lands), that origin's PRIMARY result is reported as
COVERAGE-INSUFFICIENT rather than PASS or FAIL and excluded from the
two-gating-origins computation. 18.8's verified TEST-fold coverage:
Origin B 81.94% all-books / 81.67% non-DFS (comfortably above the
floor); Origin C 53.12% (above the floor, but NARROWLY -- stated
plainly, and the floor is not being adjusted to accommodate it; if a
production run's join conventions land Origin C below 50%, the floor
triggers as registered). Origin A's test-fold coverage (75.23%) is
recorded for completeness but the floor is moot there -- Origin A is
the 18.3a placebo and enters no pass computation regardless; its
genuinely zero TRAIN+VAL coverage is the placebo's defining property,
not a COVERAGE-INSUFFICIENT trigger.

**18.6 Forbidden.** No feature additions, removals, or reweighting of
the 18.2 block after seeing any result. No hyperparameter search
beyond `SHOTS_CONFIGS`/`SAVE_RATE_CONFIGS` reused unchanged. No
threshold or policy search of any kind -- this experiment computes no
ROI and selects no EV threshold. No touching `data/betting.db`, reads
included. No Odds API credit or network use. No modification of any
pre-existing `models/trained/` directory, including
`experiment_market_state_20260710_213106` and
`experiment_market_state_origin_c_20260713_140706` themselves -- their
`metadata.json` files are read-only reused for the wiring-gate
expected values (18.3) only. New artifacts only under a new
`models/trained/experiment_15_*` directory. No evaluation of any
Origin C variant against any 2024-25 quote or outcome, in any form,
primary or secondary or diagnostic -- the 18.3 leakage guard restated
as a prohibition. No reclassifying Origin A out of placebo status, and
no issuing any Origin B/C verdict while an 18.3a placebo anomaly
(either placebo CI95 excluding zero) stands uninvestigated -- the STOP
is mandatory, and the investigation's outcome must be reported before
any verdict language is used. Crash-rerun rule identical to 17.6 item
8: one registered execution; if the runner crashes mid-run, it may be
fixed and rerun ONLY if NO phase's registered statistic (any origin's
wiring gate, either PRIMARY metric on either gating origin, the
placebo readout, the SECONDARY comparison) was yet computed and
printed/logged; otherwise the computed phases' numbers stand as-is and
must be reported. Unlike 17.6 item 8's Phase B (which explicitly
waived one-shot protection because 2024-25 was already-viewed
development data with no virgin touch to protect), this experiment's
test folds (2023-24, 2024-25, 2025-26) are ALL already-viewed
development data throughout (plan section 6.1; this document's own
section 0.1; 18.1 item 2 for 2025-26's doubly-viewed status) -- so
this crash-rerun rule exists for procedural discipline and artifact
hygiene, not because a virgin touch needs protecting, and there is no
12R-style touch-consumption machinery to invoke either way. No
post-hoc slicing (by book, date sub-window, or threshold) reported as
a result once any origin's metrics are computed.

**18.7 Consequence mapping (fixed in advance; amended 2026-07-14 to
the {Origin B, Origin C} gating set).** PRIMARY PASS
(`control_plus_microstructure` beats `no_pace_control` with CI95
entirely below zero on either registered metric, on BOTH Origin B and
Origin C) -> the block is promoted to candidate status for the 2026-27
model rebuild and queued for a SEPARATE future betting-policy
registration -- NOT promoted to live betting by this registration
alone, mirroring 17.7's shadow-candidate language exactly. The
SECONDARY comparison (on B and C) is then read (not gated) to decide
whether the block is REDUNDANT with the already-promoted market-state
block or ADDITIVE on top of it. PRIMARY FAIL on BOTH gating origins ->
the juice-skew feature lead is CLOSED this cycle, joining the section
8 closure precedents (steam-recon, DFS-census section 16.8,
BetOnline-convergence section 17.9) -- it does not reopen without a
new architecture or a genuinely new season of bettime coverage.
ONE-OF-TWO (clears the bar on exactly one of Origin B / Origin C, on
whichever metric) -> recorded as NOT REPLICATED, CLOSED, per
Experiment 5's own two-origin-agreement standard -- a single-origin
result cannot promote the block, exactly as Experiment 5's own real
Origin B improvement did not survive its both-origins bar. PLACEBO
ANOMALY (either Origin A placebo CI95 excludes zero, 18.3a) -> STOP:
no verdict of any kind is issued for the experiment until the
investigation required by 18.3a is completed and reported; the
computed B/C numbers stand as artifacts (17.6.8-style) but carry no
verdict language while the anomaly is open, and if the investigation
finds a pipeline defect, the entire run's numbers are reported as
INVALIDATED-BY-WIRING, not as pass or fail. COVERAGE-INSUFFICIENT on a
gating origin (18.5's 50% floor) -> that origin is excluded from the
PRIMARY verdict computation, and with only one usable gating origin
remaining the block-level verdict is reported as INSUFFICIENT SAMPLE
rather than forced to PASS or FAIL, mirroring the family's general
"report and stop" discipline (Experiments 8/11's INSUFFICIENT SAMPLE
handling) -- a single surviving origin cannot satisfy a
two-origin-agreement bar and is not permitted to try.

**18.8 Data inventory (verified 2026-07-14, read-only Python; no
skew value, correlation, Brier, or ROI was computed against any
season).**

`data/processed/saves_lines_snapshots.parquet` (79,884 rows / 15
columns, independently reconfirmed): 2023-24 season window
(`game_date_eastern` 2023-10-10 to 2024-04-18) `bettime` = 15,682 rows
/ 1,125 events; books `{barstool, betmgm, bovada, draftkings, fanduel,
williamhill_us}` (6, no DFS venue). **`betonlineag` is confirmed
ABSENT with ZERO rows from BOTH the 2023-24 `bettime` (0 of 15,682) AND
2023-24 `closing` (0 of 17,959) passes** -- the task's assumption is
verified, not contradicted, and independently confirms section 10's
"Consequence for Component G" note ("that book has zero quotes in the
archive for that season"). 2024-25: `bettime` = 258 rows / 21 events
(pre-purchase fragment, excluded per 18.4), `closing` = 14,954 rows /
1,288 events, `betonlineag` closing = 3,912 rows / 1,094 events.
2025-26 (Origin C's test-fold feature source under the 2026-07-14
amendment): `bettime` = 12,811 rows / 781 events, books with row
counts `{betmgm: 2,964; draftkings: 2,902; betonlineag: 2,662;
bovada: 2,377; fanduel: 1,878; fanatics: 28}` (independently
reconfirming section 17.8's book list for this cut exactly);
`betonlineag` bettime = 2,662 rows / 725 events; 17.2's
max-`resolved_ts` dedup takes this pass 12,811 -> 11,897 rows
(consistent with 17.8's 914-duplicate-group / 7.68% finding).

**Surprise #1 (not previously documented anywhere in this document
family, disclosed per the task's contradiction-reporting instruction
even though it does not contradict any stated assumption):** the
2023-24 `bettime` pass has its own within-pass duplicate natural-key
groups -- 126 of 15,556 `(event_id, goalie_name_raw, book, side)`
groups (0.81%) -- smaller than 2025-26 `bettime`'s 7.68% rate (17.2)
but nonzero and not covered by 17.2's own dedup table, which
enumerated 2025-26 `bettime`/`closing` and "both 2024-25 passes"
explicitly but was silent on 2023-24. Section 17.2's max-`resolved_ts`
dedup rule is registered here as applying uniformly to 2023-24 as well
(by reference, generalized), and this 126-group/0.81% figure is now
the first record of that fact.

**Surprise #2 (a structural coverage gap, distinct from and in
addition to the `betonlineag`-absence confirmation the task asked
for):** the 2023-24 `bettime` archive's own earliest `game_date_
eastern` is 2023-11-02, even though the season (and Origin A's test
fold) begins 2023-10-10 -- 288 of 2,624 (10.98%) `clean_training_data`
2023-24 goalie-games fall in the 2023-10-10-to-2023-11-01 window and
therefore cannot have ANY bettime saves quote from ANY book,
structurally, regardless of the microstructure block's design. This is
folded into the coverage join count below, not a separate exclusion
rule.

**Coverage JOIN COUNT, 2023-24 (training-season features, a join
count, not a statistic).** Starting from the 15,682 `bettime` rows:
220 rows (1.40%) have null `goalie_id` and are dropped (no name-based
fallback attempted in this inventory-only check; a production run
would apply 14.5 rule 1's name-matching fallback and could recover
some of these). The remaining 15,462 rows all resolve a `game_id` via
`attach_game_id`'s `(goalie_id, game_date_eastern)` `+/-1`-day lookup
against the 2,624 2023-24 `clean_training_data` goalie-games (0
unmatched). Pivoting to `(event_id, game_id, goalie_id, book, line)`
groups yields 7,678 groups, of which 7,670 have both Over and Under
present (8 single-sided groups dropped, per the fail-closed rule).
Collapsing to unique `(game_id, goalie_id)`: **1,974 of 2,624 (75.23%)
2023-24 goalie-games have at least one qualifying two-sided bettime
saves quote at some book/line.**

`data/processed/core_bettime_202607_snapshots.parquet` (413,758 rows /
23 columns, matches the task's stated schema exactly: `pass_name`,
`season`, `event_id`, `book_key`, `market_key`, `player_name_raw`,
`side`, `line`, `price_decimal`, `goalie_id` all present;
`snapshot_pass` is `bettime` ONLY for all 413,758 rows). `pass_name`
values `{sog-2023-24: 214,252; combined-2024-25: 199,506}`,
independently confirming `player_total_saves` rows (16,820) are 100%
`season == "2024-25"` / `pass_name == "combined-2024-25"` -- ZERO
`player_total_saves` rows exist anywhere under `sog-2023-24`,
independently confirming the task's assumption that 2023-24 saves
features must come from the old `saves_lines_snapshots.parquet`
archive only, not this purchase. `player_total_saves` = 16,820 rows /
1,244 events (matches the task's stated figures exactly). Saves
`book_key` values: `{betmgm: 4,172; prizepicks: 3,827; betonlineag:
3,498; bovada: 2,561; williamhill_us: 1,912; draftkings: 850}` -- 6
books, no `fanatics`, no `underdog`, no `barstool`, matching section
17.8 exactly. `betonlineag` saves = 3,498 rows / 1,050 events, matching
the task's stated figure and section 14.3 exactly. Duplicate-key check
independently reconfirmed: 23 duplicate `(event_id, player_name_raw,
book_key, side)` groups in the full saves population, ALL 23
`prizepicks`, ZERO for `betonlineag` or any other book; `betonlineag`
carries exactly one `requested_ts`/event (mean = max = 1.0) -- exact
match to section 17.8, no contradiction.

**Coverage JOIN COUNT, 2024-25 new pass (test-season features for
Origin B, a join count, not a statistic).** Starting from the 16,820
`player_total_saves` rows: 180 rows (1.07%) have null `goalie_id` and
are dropped (same caveat as above -- no name-fallback attempted here).
The remaining 16,640 rows all resolve a `game_id` via the same `+/-1`-
day lookup against 2024-25's 2,624 `clean_training_data` goalie-games
(0 unmatched). Pivoting to `(event_id, game_id, goalie_id, book_key,
line)` groups across ALL 6 books yields 8,329 groups, of which 8,311
have both sides present. Collapsing to unique `(game_id, goalie_id)`:
**2,150 of 2,624 (81.94%) 2024-25 goalie-games have at least one
qualifying two-sided bettime saves quote at ANY book (including
`prizepicks`)**; restricting to the registered non-DFS universe
(excluding `prizepicks` per 18.2's DFS-exclusion rule): **2,143 of
2,624 (81.67%)** -- `prizepicks` contributes only 7 additional
goalie-nights uniquely, confirming the DFS exclusion costs negligible
coverage.

**Experiment 5 metadata facts relied on (`experiment_market_state_
20260710_213106/metadata.json`, independently re-read, not assumed):**
`feature_sets.no_pace_control` has exactly 104 entries, including
`saves_rolling_5` (confirming feature 6's baseline reference is a real,
already-available column, not a new one). `market_feature_cols` has
exactly 7 entries plus the separate `market_indicator_col` (`mkt_
matched`); `control_plus_market_state`'s `shots_feature_count` is 112
in both origins, confirming `mkt_*` went into the SHOTS model only and
the save-rate model's `feature_cols` is the 104-column no-pace list in
both origins, shared and unchanged across variants -- no contradiction
of the task's assumption. `join_coverage`: Origin A train 0/1,864
(0.00%), val 0/760 (0.00%), test 2,622/2,624 (99.92%); Origin B train
1,902/4,528 (42.01%), val 720/720 (100.00%), test 2,622/2,624 (99.92%).
Recorded bars, independently re-read from `pre_registered_pass_bar`:
`brier_vs_control_closing` Origin A mean `+0.000537819408872642` (NOT
a CI-excluding-zero improvement), Origin B mean
`-0.0041404240194266384` (IS a CI-excluding-zero improvement);
`shots_mae_delta_vs_control` Origin A mean `+0.009711408033603576`
(NOT an improvement), Origin B mean `-0.07375802354114812` (IS an
improvement); `overall_pass: false` (neither metric cleared BOTH
origins) -- exactly reproducing section 10's summary-table verdict
("FAIL (both-origins bar)"), no contradiction.

**Experiment 8 metadata facts relied on for the amended Origin C leg
(`models/trained/experiment_market_state_origin_c_20260713_140706/
metadata.json`, independently re-read 2026-07-14, not assumed).**
`origin_c_fold_boundaries`: pool 2022-10-07 to 2025-04-17; train
2022-10-07 to 2025-02-27 (7,134 rows); val 2025-02-28 to 2025-04-17
(738 rows, `val_window_days` 49); test season 20252026 (2,624 rows) --
independently re-derived from `clean_training_data.parquet` for this
registration: the same date windows reproduce exactly 7,134 / 738 /
2,624 rows. P1 (val-fitted headline): mean `-0.003111099251412182`,
CI95 `[-0.005038647618850264, -0.0011919729523759111]`, n_bets=5,729,
n_clusters=2,070, n_push_excluded=0, `pass: true`; train-fitted
sensitivity mean `-0.003170803847077501` (sign-stable). Both Origin C
variants selected winner config `shallow_highreg` (`max_depth=2,
learning_rate=0.05, min_child_weight=30, subsample=0.8,
colsample_bytree=0.8, n_estimators=400, reg_lambda=5.0`); shots
feature counts 104 (`no_pace_control`) and 112
(`control_plus_market_state`), confirming the identical
shots-model-only placement on Origin C as on A/B. Val-fitted alphas:
`no_pace_control` `0.027718386433224374`, `control_plus_market_state`
`0.026939644644863918` (both `fit_on: val_idx`). Workload on test
(`secondaries.shots_bias_mae`): `no_pace_control` mean bias
`+0.23460506447931614` / MAE `5.407952885075313`;
`control_plus_market_state` mean bias `+0.42204831794994635` / MAE
`5.3598914618899185` (both n=2,624). Experiment 8's own recorded
bettime frame: 5,763 rows / 1,370 goalie-night clusters
(`brier_vs_market_bettime_allbooks`). Its `join_coverage_origin_c`
(market-game-state features, NOT this experiment's juice features):
train 4,506/7,134 (63.16%), val 738/738 (100%), test 2,622/2,624
(99.92%). No Experiment-5-style paired shots-`|error|`-delta CI was
recorded for Origin C anywhere in this metadata -- the basis for the
Origin C wiring-gate caveat in 18.3.

**Coverage JOIN COUNTS for the gating origins' train/val windows and
Origin C's test fold (verified 2026-07-14, join counts, not
statistics; same method as the season-level counts above --
`goalie_id`-resolved rows only, no name-fallback attempted in this
inventory check, `attach_game_id` +/-1-day lookup, two-sided
`pivot_both_sides` presence).**
- Origin B TRAIN window (2022-10-07 to 2024-02-29, 4,528 rows):
  **1,349 (29.79%)** goalie-games with at least one qualifying
  two-sided bettime saves quote (all from the 2023-24 archive; the
  2022-23 portion is structurally zero). Origin B VAL window
  (2024-03-01 to 2024-04-18, 720 rows): **625 (86.81%)**.
- Origin C TRAIN window (2022-10-07 to 2025-02-27, 7,134 rows):
  **3,520 (49.34%)** -- the union of the 2023-24 archive's 1,974
  qualifying goalie-nights and the new pass's non-DFS 2024-25 rows
  with `game_date_eastern <= 2025-02-27` (9,597 rows -> 4,759
  two-sided groups -> 1,546 qualifying goalie-nights). Origin C VAL
  window (2025-02-28 to 2025-04-17, 738 rows): **597 (80.89%)** from
  the new pass's 3,396 val-window non-DFS rows. This verifies the
  amendment's premise: Origin C's training window has real
  bettime-archive coverage, spanning the 2023-24 archive from
  2023-11-02 onward plus the new pass's 2024-25 rows through its
  train-end date.
- Origin C TEST fold (season 2025-26, 2,624 rows): after the 17.2
  dedup (12,811 -> 11,897 rows), 148 rows dropped for null
  `goalie_id`, 0 game_id-unmatched, 5,876 `(event, game, goalie, book,
  line)` groups of which 5,873 are two-sided: **1,394 of 2,624
  (53.12%)** goalie-games with at least one qualifying two-sided
  bettime quote -- above 18.5's 50% COVERAGE-INSUFFICIENT floor, but
  narrowly, as 18.4 discloses. The structural cause is archive
  breadth (781 of the season's events have any bettime snapshot), not
  book pairing.

**One correction of the task's own framing, not a contradiction of
verified data:** the original task brief states this "P1-style"
both-origins bar is "what Experiments 5 and 8 required." Verified true
for Experiment 5. Verified FALSE as stated for Experiment 8, which
tested a single fresh origin (Origin C, 2025-26 only) against a "P1
AND P2 both pass" bar on that one origin (section 11.7) -- there were
never two origins to require agreement across. Under the 2026-07-14
amendment this experiment retains Experiment 5's two-origin-agreement
STRUCTURE but applies it to {Origin B, Origin C} (18.3), so the
correction is doubly relevant: the "Experiments 5 and 8" framing must
not be repeated as fact in a future document, and the agreement bar in
force here derives its authority from Experiment 5's registered logic
alone, applied to the amended origin set.

No other contradiction of any assumption in the task brief was found:
`core_bettime_202607_snapshots.parquet`'s schema matches the stated 23
columns exactly; `betonlineag` absence from 2023-24 is confirmed, not
contradicted; the new pass's `betonlineag` saves figures (1,050
events) match exactly; and Experiment 5's `mkt_*` placement (shots
model only) matches exactly.

**18.9 Implemented result -- LEAD CLOSED (Sonnet sub-agent execution
under lead-reviewer direction, independently verified, 2026-07-16).**
`scripts/experiment_15_w3_microstructure.py` completed all three
origins' wiring gates, the 18.3a placebo, and both the PRIMARY and
SECONDARY comparisons in one execution, wall clock 183.1s, no crash.
Every number below was independently recomputed by the lead reviewer
from the persisted `bootstrap_cluster_inputs.parquet` and quote
universes, point estimates matching to 10 decimals, and the 18.2
feature formulas verified by hand against raw snapshot rows for a
sample goalie-night.

Wiring gate: BIT-IDENTICAL (abs diff exactly 0.0) against every
recorded Experiment 5 Origin A/B and Experiment 8 Origin C value --
Brier-vs-control means, ns, cluster counts, workload bias/MAE pairs,
all six val-fitted alphas. Coverage reconciliation exact on all seven
registered 18.8 counts. Coverage floor: Origin B 81.67% non-DFS,
Origin C 53.125% -- both SUFFICIENT.

Placebo (Origin A, zero `juice_*` training exposure): NO ANOMALY -- all
four registered anomaly-surface CIs (closing Brier and shots `|error|`,
PRIMARY and SECONDARY) contain zero. PRIMARY closing Brier
`+0.00022468387361090333`, CI95 `[-0.00044703079794508187,
+0.0008851610428282828]`. One disclosure: the NON-registered placebo
SECONDARY bettime Brier CI `[-0.0014621365511030798,
-0.000016680192124909426]` marginally excludes zero by 1.7e-5 -- outside
the registered 18.3a surface (which names closing Brier and shots
`|error|` only), one of six placebo CIs at 95%, recorded as a
noise-floor calibration fact, triggers no rule.

PRIMARY (`control_plus_microstructure` minus `no_pace_control`;
negative = block helped; PASS required CI95 upper `< 0` on both gating
origins on either metric):
- Origin B closing Brier: mean `-0.0010586603482554456`, CI95
  `[-0.0029536449440400203, +0.0008505503511136891]`, n=7,463 / 2,510
  clusters -- right direction, does not clear.
- Origin B shots `|error|`: mean `-0.028319315939414794`, CI95
  `[-0.058235809243306874, +0.0015836850717300132]`, n=2,624 -- right
  direction, narrowly does not clear.
- Origin C closing Brier: mean `+0.0004815237825549298`, CI95
  `[-0.0005084216363560196, +0.0014548538581875285]`, n=5,729 / 2,070
  -- wrong direction.
- Origin C shots `|error|`: mean `+0.011691479421243435`, CI95
  `[-0.0036416111559402654, +0.02691749608734759]`, n=2,624 -- wrong
  direction.

Bettime secondaries agreed with the closing readout: Origin B mean
`-0.0004335398581977066`, CI95 `[-0.0025017811918127656,
+0.0016738888469758511]`; Origin C mean `+0.001133996788670537`, CI95
`[-0.0003436422726257154, +0.002663959185821192]`.

SECONDARY (`control_plus_market_state_plus_microstructure` minus
`control_plus_market_state`, the redundancy question): no gain
anywhere -- all Origin B/C means weakly positive (weakly worse), e.g.
Origin B closing `+0.0004724257158626342`, CI95
`[-0.00047223716260126326, +0.0014621754617122253]`; Origin C closing
`+0.0006841048542424979`, CI95 `[-0.000052200637443091104,
+0.0014507269606154575]`. Read plainly: on top of the already-promoted
market-state block, microstructure is redundant-to-slightly-harmful.

**Verdict (18.5/18.7): PRIMARY FAIL on both gating origins** -- not
ONE-OF-TWO; neither origin cleared either metric. The juice-skew
feature lead is **CLOSED this cycle**, joining the section 8 closure
precedents (steam recon, DFS census 16.9, BetOnline convergence 17.9).
It does not reopen without a new architecture or a genuinely new
season of bettime coverage.

Context, stated plainly: the placebo arm's own noise floor (all-NaN
columns moving Brier by up to ~0.0007 in either direction) means
Origin B's `-0.0011` closing mean is barely above procedure noise; the
original unclustered discovery `r=0.032` is consistent with
pseudo-replication inflation, the same lesson as W6; and the SECONDARY
readout suggests the game-level `mkt_*` block already carries whatever
market information the saves-market microstructure would add.

Disclosed judgment calls (full list in the run's `metadata.json`
`judgment_calls` key and the lead reviewer's records): before any
statistic was computed, the runner's own pre-execution self-review
caught and fixed a pass-bar direction bug -- the PASS criterion had
been implemented two-sided where 18.5 registers a one-sided CI95-
upper-bound-below-zero bar -- and corrected it prior to computing
anything. Other disclosed calls: all bettime-archive loading deferred
until after the wiring gate passed (stricter than the registered gate
targets strictly require); `juice_n_books` computed via `nunique()` on
book, verified empirically equivalent to a row count under the 17.2
dedup key; and Origin C's SECONDARY bettime grading universe built via
`build_season_multibook_frame`'s own internal cleaning rather than
pre-applying the 17.2 dedup, per 18.4's own anchor to Experiment 8's
recorded frame -- this reproduced Experiment 8's recorded 5,763/1,370
bettime frame exactly. Artifacts:
`models/trained/experiment_15_w3_microstructure_20260716_124811/`.

---

## 19. Experiment 16 -- Alternate-saves one-sided-ladder feasibility pilot

Registered 2026-07-16 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, before any credit is spent, any alternate-ladder quote is
loaded, or any coverage/calibration statistic is computed on any season.
This operationalizes the remainder decision flagged repeatedly since the
2026-07-13 probe: `BREAKTHROUGH_MODEL_PLAN.md` section 5.7's "Flexible
remainder: targeted over-only alternate-saves pass ... up to 12,125
[credits]," `HISTORICAL_DATA_ANALYSIS.md` sections 9.3/9.5's "alternate-
saves remainder is capped at a 1-2k pilot until a one-sided ladder model
is shown to work," and section 5.7's own core-purchase note rejecting an
"~11,000-credit alternate-ladder purchase" outright because "over-only
quotes need a one-sided vig model that does not exist yet; pilot 1-2k
first." This is that pilot's registration. Unlike sections 17 and 18, it
authorizes a real, bounded purchase (max 2,000 credits of the 12,895
remaining, expiring 2026-07-31) AND fixes the feasibility analysis's
every formula and pass bar before that purchase's data is seen -- both
halves fixed together, before either is executed.

### 19.1 Hypothesis and honesty notes

Hypothesis: BetOnline's over-only alternate-saves ladder, de-vigged
against its own same-snapshot standard two-sided line at the anchor rung
and extended across the ladder under a simple, fixed, one-sided
vig-extension assumption, produces exceedance probabilities at
NON-anchor rungs that are (a) available on a large-enough fraction of
goalie-nights to justify a full-season purchase, and (b) more
informative about the actual saves outcome than a baseline built from
the standard line alone. This is a FEASIBILITY question, not an edge
claim. Per plan section 6.1's interpretation hierarchy, the strongest
possible outcome of this experiment is authorization to DRAFT a
full-season purchase registration -- never a betting-policy result,
never "edge" language, and never an automatic further spend.

Honesty constraints, stated plainly rather than rounded up:

1. Alternate saves lines do not exist historically before 2024-25
   (probe-verified: `BREAKTHROUGH_MODEL_PLAN.md` section 5.7,
   `HISTORICAL_DATA_ANALYSIS.md` section 9.3, reconfirmed in 19.8). There
   is no untouched season available for this market anywhere in this
   project's current data, and there never will be one -- both 2024-25
   and 2025-26 outcomes have already been opened repeatedly by this
   document family for other purposes (Experiments 5, 8, 11, 12, 14, 15).
   Any PASS this pilot produces is development evidence on already-viewed
   seasons, exactly the caveat sections 17.1/18.1 attach to their own
   touches of the same two seasons.
2. 2025-26 carries the same doubled honesty burden 18.1 item 2 flagged
   for the juice-skew lead: it is simultaneously the alternate-saves
   DISCOVERY season (the W1 probe that first found alternate-saves
   coverage sampled 2025-26 alongside 2024-25) and the live-bet season.
   Nothing below treats a 2025-26 result as confirmatory in any stronger
   sense than a 2024-25 result for that reason.
3. Twenty-six BetOnline goalie-nights of alternate-ladder quotes are
   ALREADY OWNED (the W1 probe: 15 events -- 7 in 2024-25, 8 in 2025-26 --
   12 and 14 goalie-nights respectively, 19.8) and were already seen at
   the raw-quote level during probe design (section 9.3's "all 455
   alternate-saves outcomes were over-only" finding). Critically, no
   prior document in this family has ever joined those 26 nights' ladder
   quotes to actual saves OUTCOMES -- the W1 probe was "a data-
   availability and billing probe, not a model experiment"
   (`HISTORICAL_DATA_ANALYSIS.md` section 9's own framing) and never
   touched `clean_training_data.parquet`. 19.3 registers exactly how this
   pilot reuses those 26 nights (folded into the primary universe at zero
   additional cost, flagged as prior-exposure quotes) rather than
   silently treating them as fresh.
4. This registration itself was drafted from read-only inspection of
   already-owned data only: no network call, no credit spend, and no
   outcome-linked statistic (no Brier, no calibration check, no
   coverage-vs-outcome join) was computed anywhere in producing this
   section. Every count in 19.8 is a coverage, schema, or timing join
   count, not a statistic, exactly matching 17.8/18.8's own distinction.
5. The one-sided vig-extension assumption (19.4) is genuinely new
   machinery in this document family -- there is no prior artifact to
   reproduce a wiring gate against (unlike Experiments 11/14/15, which
   each reused a frozen or previously-registered component). Its own
   coverage and calibration gates (19.5) are this pilot's entire reason
   to exist; a FAIL here is a legitimate, expected, and useful outcome,
   not a bug.

### 19.2 Registered definitions

**Reused verbatim, by reference, not restated in full:**

- *De-vig method.* Proportional (multiplicative) normalization of a
  two-sided quote at the SAME book, SAME line, SAME snapshot, SAME
  goalie-night (17.2, restated as binding in 18.2) -- the
  odds-averaging-bug rule (`HISTORICAL_DATA_ANALYSIS.md` section 1).
  `raw_p_over = 1/price_decimal_over`, `raw_p_under =
  1/price_decimal_under`, `overround = raw_p_over + raw_p_under`,
  `p_over_devigged = raw_p_over/overround`.
- *DFS exclusion.* `prizepicks` and `underdog` never contribute to any
  anchor, ladder, or baseline computation here (17.2's Other-books rule;
  18.2's restatement). PrizePicks' own alternate-saves quotes were
  independently found in 17.8/18.8 to carry simultaneous same-side
  alternate lines for the same player -- not a genuine fixed-line
  two-sided market, and structurally the same "ladder-shaped but not
  vig-priced" product this experiment is trying to price properly for a
  real sportsbook, making PrizePicks doubly inappropriate as either an
  anchor or a ladder source here.
- *Dedup.* Within a single `snapshot_pass`/pass label, rows sharing a
  natural key are deduped to the MAXIMUM `resolved_ts` (ties by max
  `requested_ts`, then row order), applying uniformly to any archive row
  this experiment reads (17.2, extended to 2023-24 in 18.8's Surprise
  #1).
- *Never average raw odds.* Restated because this experiment's ladder-
  extension mechanism (19.4) could easily be misread as odds-averaging if
  implemented carelessly -- it is not: every division in 19.4 divides by
  a SINGLE book's SINGLE anchor-rung overround, never a cross-book or
  cross-rung average.

**New for this experiment:**

- *Primary book scope: `betonlineag` ONLY*, for both the anchor
  (standard two-sided line) and the ladder (alternate over-only rungs)
  at every step of the PRIMARY universe. Deliberate, not an oversight:
  (a) 17.2/18.2's same-book-same-line-same-snapshot de-vig rule already
  forbids mixing books between an anchor and a ladder within one
  computation; (b) the W1 probe found BetOnline is the only
  alternate-saves provider with near-universal, consistently-shaped
  coverage (7/8 2024-25, 8/8 2025-26 events; 7-8 rungs per goalie-night
  at a uniform 2-point spacing every time it is present, 19.8) --
  FanDuel and Fanatics appear on far fewer nights with far less
  consistent rung counts (19.8), and PrizePicks is DFS-excluded above;
  (c) BetOnline is the only one of these venues that is an actual
  bettable sportsbook for this project, consistent with the BetOnline
  focus of Experiments 11 and 14 (sections 14 and 17). Unlike 18.2's microstructure block,
  this is NOT a 2023-24-trainability workaround -- alternate saves do not
  exist in 2023-24 at ANY book, so there is no book-agnostic-for-
  trainability motive here; the choice is purely "which book's ladder is
  usable," and BetOnline is the only real answer the data gives.
- *Goalie-night unit.* `(event_id, goalie identity)`, resolved via
  `goalie_id` where both sides resolve it, else
  `goalie_name_matched`/`goalie_name_raw` fallback (14.5 rule 1, by
  reference), joined to `clean_training_data.parquet` via
  `attach_game_id`'s `(goalie_id, game_date)` +/-1-day lookup (18.4's
  convention, by reference; `clean_training_data.parquet` uses a plain
  `game_date` column, not `game_date_eastern` -- verified in 19.8, not
  assumed).
- *Anchor rung.* The point value `L_std` of BetOnline's own two-sided
  `player_total_saves` (standard) quote for that goalie-night at the
  SAME snapshot as the ladder (19.3 registers exactly how "same
  snapshot" is verified/enforced per season). Requires BOTH sides
  present at `L_std` for BetOnline (17.2's pairing rule) -- a
  single-sided BetOnline standard quote yields no anchor and the
  goalie-night is excluded.
- *Ladder rungs.* The set of distinct `point` values `{L_1 < L_2 < ... <
  L_k}` for which BetOnline posts an `Over` alternate-saves outcome for
  that goalie-night at the SAME snapshot as the anchor. Per-rung raw
  probability `raw_p_over(L_i) = 1/price_decimal_over(L_i)`. Verified in
  19.8: `L_std` is one of `{L_i}` in 26 of 26 already-owned goalie-nights
  where both a BetOnline standard line and a BetOnline alternate ladder
  exist at the same snapshot -- the anchor rung is expected, not merely
  hoped, to sit exactly on the ladder itself; 19.4's consistency
  diagnostic treats a night where `L_std` is NOT found among the ladder
  rungs as an anomaly to flag, not silently drop.
- *Outcome.* `data/processed/clean_training_data.parquet`, `saves`
  column, ONLY -- `data/betting.db` is forbidden, reads included (17.6
  item 2 / 18.4's stricter carve-out, restated as binding). Settlement:
  `label_over(L_i) = 1{saves_actual > L_i}` -- unambiguous because every
  observed rung in the owned data is an `X.5` value (19.8), so no push
  case exists.

### 19.3 Purchase design

**Script and cache discipline** (mirrors `scripts/purchase_core_bettime_
passes.py` exactly, by reference, restated as binding requirements for
the new script):

- New dedicated script: `scripts/purchase_alt_ladder_pilot.py`. New
  dedicated append-only cache directory:
  `data/raw/betting_lines/passes/alt_ladder_pilot_202607/`, record
  naming `altladder_event={event_id}_signature={signature}.json` -- a
  shape that cannot collide with `probe_opening_markets.py`'s
  `w1_event=...` or `purchase_core_bettime_passes.py`'s `core_event=...`
  records, per the established never-shared-cache-naming convention.
- DRY-RUN IS THE DEFAULT. No network call without `--execute`, and
  `--execute` additionally requires `--max-credits`.
- `--max-credits 2000` (the registered hard cap for this pilot -- not a
  default the runner may raise). `--credit-floor 10895` (= 12,895 -
  2,000, the registered floor; the script aborts, exactly as
  `purchase_core_bettime_passes.py.execute()` already does, the instant
  a call's `x-requests-remaining` header falls below this floor or below
  zero -- a live-header check, never a locally-estimated one).
- Plans are built entirely from the cached events-list envelopes under
  `data/raw/betting_lines/cache/` (`events_date=*.json`), never from
  `data/betting.db`, reusing `purchase_core_bettime_passes.py`'s own
  `load_cached_events`/`_season_events` verbatim rather than
  reimplementing a second, potentially divergent, season-window/Eastern-
  date calculation.
- API key loaded from `API_KEY`/`THE_ODDS_API_KEY` env var or `.env`,
  exactly as the existing scripts do; never printed, logged, or written
  into any cache record.
- Worst-case per-event credit reservation happens BEFORE dispatch, not
  after -- reused verbatim from `execute()`'s existing "may have billed
  even if the connection dies" reasoning.
- A previously-recorded signature (any complete response, 200 or
  non-200) is NEVER re-requested -- reused verbatim.

**Season mix and call-type decision, reasoned explicitly.** The choice
between an alt-only call (`markets=player_total_saves_alternate` only,
up to 10 credits/event, relies on aligning with an ALREADY-OWNED
standard quote from a separate call) and a combined call
(`markets=player_total_saves,player_total_saves_alternate`, up to 20
credits/event, guaranteed same-envelope anchor, half the sample for the
same budget) depends on whether each season's EXISTING standard-saves
archive is a trustworthy anchor for a freshly-purchased ladder at the
same requested bet-time anchor. Checked directly, not assumed (full
detail in 19.8):

- *2024-25.* `core_bettime_202607_snapshots.parquet`'s `combined-2024-25`
  pass was built by `purchase_core_bettime_passes.py` using the exact
  same `compute_bettime_ts` anchor formula this pilot will reuse. Of the
  16 events where the already-owned W1 probe (issued 2026-07-13) and the
  already-owned core pass (issued 2026-07-14, a separate day, a separate
  call) both happened to sample the same event id, the two calls'
  requested `date` params were identical in all 16 cases (by
  construction) AND the returned envelope `timestamp` fields were
  BYTE-IDENTICAL in all 16 cases; of the 7 of those 16 with a BetOnline
  standard-saves quote present in both calls, all 24 shared `(player,
  side, point)` price outcomes matched exactly (zero diffs) across the
  two independently-issued calls (19.8; lead-reviewer recount 2026-07-16:
  7 events, 2+2+4+4+4+4+4 = 24 shared outcomes, correcting the draft's
  event count of 6 -- the 24-outcome total was and remains exact). This is direct, verified
  evidence that the historical-odds endpoint resolves an identical
  requested `date` to the identical archived snapshot regardless of
  which call requests it or when. **Decision: 2024-25 uses the alt-only
  call type**, doubling the achievable sample for the same credit spend,
  with alignment risk empirically bounded at near-zero and still checked
  per-night by the alignment rule below (never trusted blind).
- *2025-26.* `saves_lines_snapshots.parquet`'s 2025-26 `bettime` rows do
  NOT uniformly follow `compute_bettime_ts` -- checked directly against
  all 781 events with a 2025-26 bettime row (19.8): only 725/781
  (92.83%) have `requested_ts` exactly equal to the computed anchor; the
  remaining 56 (7.17%) diverge, in some cases by up to 4.5 hours (16,200
  seconds) -- meaning at least part of this archive's 2025-26 bettime
  rows were captured by a mechanism other than a purchase-script-style
  historical-anchor call. Separately, this archive covers only 781 of
  the season's 1,232 in-window events (63.7%) -- most 2025-26 events
  have no existing standard-saves quote to anchor against at all,
  regardless of alignment. An alt-only design for 2025-26 would
  therefore (a) need to restrict its sampling frame to the well-aligned
  subset, discarding most of the season's coverage before drawing a
  single sample, and (b) still carry residual per-night alignment risk
  the 2024-25 leg does not have. **Decision: 2025-26 uses the combined
  call type** -- a single call fetches both the standard line and the
  alternate ladder together, guaranteeing same-envelope alignment by
  construction and reaching events the existing archive never touched,
  at the cost of half the per-credit sample size. The more expensive
  option, bought deliberately for the season whose existing archive
  cannot supply a reliable free anchor.

This produces a genuine per-season mix with a stated, data-grounded
reason for each leg, not a single uniform policy applied without
justification.

**Registered sample sizes (fixed now, before any purchase):**

| Season | Call type | Markets requested | Credits/event (worst case) | Sampling pool (candidates) | Target N (new purchases) | Worst-case credits |
|---|---|---|---:|---:|---:|---:|
| 2024-25 | alt-only | `player_total_saves_alternate` | 10 | 1,043 (BetOnline-standard-covered events, minus the 7 already-probed with BetOnline coverage; 19.8) | 120 | 1,200 |
| 2025-26 | combined | `player_total_saves,player_total_saves_alternate` | 20 | 1,224 (all in-window events, minus the 8 already-probed; 19.8) | 35 | 700 |
| **Total** | | | | | **155** | **1,900** |

Worst-case total (1,900) is deliberately 100 credits below the
registered `--max-credits 2000` safety cap, so the cap is a genuine
backstop, not a value the plan expects to hit exactly; it is NOT itself
a target, and the runner may not raise the target N opportunistically
because headroom remains (19.6). Bookmakers requested: the same nine
named books used by every prior pass in this family (`draftkings,
fanduel, betmgm, williamhill_us, fanatics, bovada, betonlineag,
underdog, prizepicks`) plus `includeMultipliers=true`, purely for
schema/signature consistency -- billing depends only on distinct MARKETS
returned, never on book count (verified repeatedly, 9.1/17.8), so
requesting all nine costs nothing extra and preserves optionality for a
future non-primary characterization of FanDuel/Fanatics ladder coverage
(19.5's coverage diagnostics, non-gating).

**Seeded random sampling rule (fixed now).** For each season's candidate
pool (table above, sorted deterministically by `(commence_time,
event_id)`, exactly as `_season_events` already sorts): assign each
candidate a 0-based index in that sorted order; draw
`numpy.random.default_rng(42).permutation(pool_size)` (seed 42, this
document family's uniform bootstrap/sampling seed); the target sample is
the first `N` candidates in permutation order. The FULL permutation (not
just the first N) is persisted in the run's plan artifact, so that if a
run needs to extend the sample later within the same registered N and
credit cap (e.g. a `--limit` run stopped early, or a handful of 404s
need replacing), the NEXT events are drawn from the same frozen
permutation in order -- never redrawn, never cherry-picked. The sample
may not be redrawn, reordered, or resized after this registration is
filed (19.6).

**Already-owned probe events: reuse for free, exclude from new
sampling (fixed now).** The 15 events (7 in 2024-25, 8 in 2025-26) the
W1 probe already purchased are (a) EXCLUDED from both seasons' new
candidate pools (already reflected in the pool sizes above) --
re-requesting them would be a redundant paid call for data already on
disk; and (b) FOLDED INTO the primary analysis universe at zero
additional cost, read directly from the existing
`data/raw/betting_lines/probes/w1_market_coverage/` records, flagged in
every report as `source = "probe_reuse"` versus `source =
"new_purchase"` for the 155 newly purchased events -- a provenance flag,
not a different formula; the SAME 19.2/19.4 definitions apply
identically to both sources. These 26 already-owned BetOnline
goalie-nights (12 in 2024-25, 14 in 2025-26; 19.8) are same-snapshot-
aligned by construction (one combined call already fetched both markets
together), so they pass the alignment check below trivially, and this is
reported as such rather than silently omitted from the check.

**Alignment verification rule (fixed now, applies uniformly to every
goalie-night in the primary universe, including trivially-passing
probe-reuse and 2025-26-combined nights).** `alignment_gap_seconds =
|ladder_envelope_timestamp - standard_quote_observed_timestamp|`, using
OBSERVED (actually-returned) timestamps on both sides -- the ladder
call's own returned envelope `timestamp` field, and the standard
quote's own observed capture time (`resolved_ts` for
`saves_lines_snapshots.parquet`-sourced anchors, `fetched_at`/envelope
`timestamp` for `core_bettime_202607_snapshots.parquet`-sourced anchors)
-- never the nominal REQUESTED `date` param on either side, consistent
with 17.2's observed-over-nominal precedent. Tolerance:
`alignment_gap_seconds <= 300` (5 minutes), chosen to match the
existing project's own pre-established "material drift" threshold
(`audit_core_bettime_passes.py`'s `check_anchor_integrity`'s
`n_drift_gt_5min` bucket, reused as a natural precedent rather than
inventing a new number). A goalie-night with `alignment_gap_seconds >
300`, or where either side's observed timestamp cannot be resolved at
all, is EXCLUDED from the primary universe -- fail-closed, counted and
reported, never imputed or tolerated with a wider window after the fact.
For 2025-26 combined-call and probe-reuse nights this gap is 0 by
construction and the check is expected to pass unconditionally, but it
is still COMPUTED and reported for every night, not skipped for the
cases expected to trivially pass.

**Independent audit requirement** (mirrors
`scripts/audit_core_bettime_passes.py`, mandatory before any analysis
script trusts the new cache). New dedicated script
`scripts/audit_alt_ladder_pilot.py`, read-only, recomputing from the raw
`data/raw/betting_lines/passes/alt_ladder_pilot_202607/` records (never
trusting the purchase script's own run-log summary): (1) record
integrity -- parses, signature recomputation, filename-embeds-signature,
no `apiKey` substring anywhere in raw text, season-window membership;
(2) billing arithmetic -- `x-requests-last == 10 * distinct markets
actually returned` on every 200, running total reconciled against
claimed spend, monotonically non-increasing `x-requests-remaining`,
implied starting balance reconciled against the pre-purchase 12,895; (3)
non-200s -- enumerate, confirm zero-cost, confirm `EVENT_NOT_FOUND`
where applicable; (4) anchor integrity -- the SAME `alignment_gap_
seconds` check as above, independently recomputed from raw records
rather than trusted from the purchase script's own log; (5) coverage --
rungs-per-goalie-night distribution, per-book breakdown, exact-duplicate
and one-sided-outcome schema traps (mirroring `check_coverage`'s
existing structure); (6) pairing potential -- read-only join against
`core_bettime_202607_snapshots.parquet` (2024-25) and
`saves_lines_snapshots.parquet` (2025-26) to independently size the
qualifying primary universe before the feasibility analysis script runs.
This audit MUST run and its `all_clean`-equivalent integrity flag MUST
be true (or every violation individually reconciled and disclosed)
before `scripts/experiment_16_alt_ladder_pilot.py` (19.4/19.5) is
permitted to load a single cached record -- an audit FAILURE is a STOP,
not a warning, mirroring the STOP-AND-INVESTIGATE discipline 18.3a
already established for a different kind of anomaly.

### 19.4 One-sided ladder model and baseline

Fully specified before any purchase. Kept deliberately simple and fully
deterministic per the task's own instruction -- this is a feasibility
pilot, not an optimization.

**Anchor vig.** For a qualifying goalie-night (19.2/19.3), from
BetOnline's own two-sided standard quote at `L_std`: `overround_std =
raw_p_over(L_std) + raw_p_under(L_std)` (both from the STANDARD market's
own paired prices, not the ladder), `p_over_devigged(L_std) =
raw_p_over(L_std) / overround_std` (17.2's proportional method,
restated).

**Vig-extension assumption (the one new modeling choice this pilot
registers).** The ladder gives only `raw_p_over(L_i)` at every rung -- no
paired under price to de-vig each rung independently. Registered
assumption: BetOnline's OVERROUND is CONSTANT across all rungs of the
same goalie-night's ladder, equal to `overround_std` (the anchor's own
measured overround). This is the constant-multiplicative-factor
extension the task named as admissible, and the simplest one that (a)
requires no new free parameter beyond what the anchor already measures,
and (b) is automatically monotonicity-preserving before any enforcement
step (dividing an already-decreasing sequence by one positive constant
cannot reorder it): `p_over_devigged_raw(L_i) = raw_p_over(L_i) /
overround_std` for every rung `L_i`, INCLUDING `L_std` itself (which
recovers 19.4's anchor value exactly when the ladder's own quote at
`L_std` matches the standard market's quote at `L_std` -- checked, not
assumed: the raw discrepancy between the ladder's own `L_std`-implied
value and the standard market's `p_over_devigged(L_std)` is reported as
a wiring/consistency diagnostic, non-gating, because these are two
DIFFERENT outcomes objects -- alternate-market vs. standard-market --
even when priced at the identical point and snapshot, and a small
discrepancy is expected book behavior, not necessarily an error).

**Monotonicity enforcement.** Sort rungs ascending by `L_i`. Walk
forward: `p_over_devigged(L_1) = p_over_devigged_raw(L_1)`; for `i =
2..k`, `p_over_devigged(L_i) = min(p_over_devigged_raw(L_i),
p_over_devigged(L_{i-1}))`. A plain forward-min clip -- deterministic,
auditable in one line, no fitting or optimization (isotonic regression
was deliberately NOT used, to keep this a fixed formula rather than a
fitted procedure). The number of rungs actually clipped (where
`p_over_devigged_raw(L_i) > p_over_devigged(L_{i-1})`) is counted and
reported per goalie-night and in aggregate -- a diagnostic of raw ladder
noise, not itself a pass/fail criterion.

**Nights lacking a usable anchor.** A goalie-night with a BetOnline
alternate ladder but no qualifying BetOnline standard two-sided quote at
the same aligned snapshot (missing entirely, one-sided only, or failing
the 19.3 alignment tolerance) has no `overround_std` to divide by and is
EXCLUDED from the primary universe outright -- fail-closed, counted,
never imputed from a different book, a different snapshot, or a
league-average overround.

**Baseline (uses ONLY the standard line, fit on 2023-24, which has no
alternates).** `betonlineag` is CONFIRMED ABSENT (zero rows) from
2023-24 in BOTH `bettime` and `closing` (18.8, reconfirmed 19.8) -- so
the baseline's shape parameter cannot be fit from BetOnline-only
2023-24 data, and this experiment does not pretend otherwise. Registered
fix, disclosed as a structural asymmetry rather than hidden: the
baseline's fit input on 2023-24 reuses Experiment 15's own book-agnostic
construction (18.2) BY REFERENCE -- `p_over_std_2023-24(goalie-night) =
1 - juice_p_under_consensus`, i.e. the MEDIAN de-vigged probability
across whichever qualifying (two-sided, non-DFS) books quote the modal
line `L*` for that goalie-night (18.2's `juice_p_under_consensus`/
modal-line machinery exactly; median commutes through the strictly
monotonic `1-x` transform, so this equals the median of `p_over_devigged`
across the same books). This machinery was chosen specifically because
it was already built, already registered, and already independently
executed and verified (18.9) for exactly the same "betonlineag doesn't
exist in 2023-24" problem. Implementation may source it either by
reading it directly (read-only) from Experiment 15's persisted
per-goalie-night frame in
`models/trained/experiment_15_w3_microstructure_20260716_124811/` if a
suitable full-season 2023-24 frame exists there, or by recomputing it
fresh from `saves_lines_snapshots.parquet`'s 2023-24 `bettime` rows
using the identical 18.2 formula; either path must reproduce identical
values on a spot-check sample before being trusted (a lightweight wiring
gate, mirroring this family's standing discipline) -- the FORMULA, not
the file path, is the registered source of truth. This is a deliberate
asymmetry versus the ladder model's betonlineag-only scope, disclosed
rather than hidden: there is no other way to fit anything on 2023-24
using betonlineag specifically, since it has zero rows there.

Fit procedure (closed-form, no iteration): for every 2023-24
goalie-night with `juice_matched == 1` (18.2) and `0 < p_over_std_2023-24
< 1` (exact 0/1 excluded, where the implied Normal quantile is
undefined; counted, not imputed), let `z_i = Phi^-1(1 - p_over_std_2023-
24,i)`, `y_i = saves_actual,i - (L*_i + 0.5)`, `x_i = -z_i`. `sigma_0 =
sum(x_i * y_i) / sum(x_i^2)` -- the OLS slope through the origin of `y`
on `x`, a single global scalar (deliberately not bucketed by workload or
goalie, per the "keep it simple" instruction; a bucketed version is
explicitly out of scope for this pilot and may not be added post-hoc,
19.6). If `sigma_0 <= 0` (a pathological/degenerate fit), the baseline is
UNDEFINED and 19.5's calibration test is reported as INSUFFICIENT SAMPLE
rather than computed with a nonsensical shape parameter.

Baseline translation, applied to every non-anchor rung `L_i` of every
primary-universe 2024-25/2025-26 goalie-night (using that night's OWN
`L_std` and `p_over_devigged(L_std)` -- the standard line only, never any
ladder rung, satisfying "uses only the standard line"): `mu_hat = L_std
+ 0.5 - sigma_0 * Phi^-1(1 - p_over_devigged(L_std))`; `BASELINE_p_over
(L_i) = 1 - Phi((L_i + 0.5 - mu_hat) / sigma_0)`. This is a deliberately
simple Normal-shape translation, disclosed explicitly as simpler than
the project's own production NB2 distributional model
(`src/experiments/distributional_saves.py`) -- intentional, since this
baseline exists only to give the ladder model a genuine, pre-registered
comparison point for this pilot, not to be a candidate feature or model
itself.

### 19.5 Registered metrics and pass bars

**Coverage gates.** Denominator: every goalie-night (`game_id`,
`goalie_id`) in `clean_training_data.parquet` whose game falls among the
SAMPLED events -- the 155 newly purchased events (19.3's table) plus the
15 already-owned probe events, 170 events total across both seasons --
regardless of whether that particular call actually returned alternate
data (a join count against the SAMPLING DESIGN, not a post-filtered
count, mirroring 18.8's "Coverage JOIN COUNT" framing exactly). A
goalie-night QUALIFIES for the primary universe iff ALL of: (a) a
resolved goalie identity; (b) a computable BetOnline anchor (19.4,
including the 19.3 alignment check); (c) at least 5 distinct BetOnline
ladder rungs (including the anchor rung) at the same aligned snapshot.

- Rung-depth floor: `>= 5` distinct rungs. Chosen below the probe's own
  observed BetOnline range (7-8 rungs on all 26 already-owned nights,
  19.8) specifically to allow real-purchase attrition without instantly
  failing a night with slightly thinner coverage than the probe's small
  sample happened to show, while still requiring genuine ladder breadth.
- Coverage-gate PASS bar: `>= 70%` of the 170-event-derived goalie-night
  population qualifies. This directly reuses `BREAKTHROUGH_MODEL_PLAN.md`
  section 5.1's own original go/no-go convention for "is this market's
  coverage good enough to justify buying more" decisions ("at least two
  usable books on 70% or more of sampled events") -- the same style of
  decision this pilot is making, at the same threshold, not a fresh
  number invented for this document.
- Reported unconditionally, non-gating: rungs-per-goalie-night
  distribution (min/median/mean/max) split by season and by `source`
  (`new_purchase` vs `probe_reuse`); per-book breakdown (BetOnline vs.
  FanDuel vs. Fanatics vs. excluded PrizePicks) for future full-purchase
  planning value; the exclusion funnel (no resolved identity / no anchor
  / alignment failure / rung-depth failure), summing exactly to the
  170-event-derived population, mirroring this family's standing
  exclusion-funnel-must-reconcile discipline (17.9's Phase A funnel is
  the direct precedent).

**Calibration/informativeness primary.** Cluster floor: `>= 50`
qualifying goalie-nights (a scaled-down analogue of 17.5's 100-bet
floor, proportional to this pilot's roughly 5x-smaller credit budget
relative to the core passes; the unit here is `(goalie-night,
non-anchor-rung)` rows, and each qualifying night contributes several
such rows given the established 5-8 rung depth, so 50 clusters yields
materially more graded rows than 17.5's 100 single-bet clusters did). If
`n_qualifying < 50`, the calibration test is INSUFFICIENT SAMPLE
regardless of the coverage-gate percentage (a small absolute purchase
outcome, e.g. from an early credit-floor stop, can technically clear 70%
of a tiny denominator without supporting a trustworthy bootstrap).

Population: every `(goalie-night, L_i)` pair in the qualifying primary
universe where `L_i != L_std` (the anchor rung is excluded from the
calibration population -- it is de-vigged directly from a genuine
two-sided market and would trivially "calibrate," which is not the
question; the test is specifically about NON-anchor rungs, per the
task's own framing).

Per row: `label_over(L_i) = 1{saves_actual > L_i}` (19.2). LADDER
prediction = `p_over_devigged(L_i)` (19.4, monotonicity-enforced).
BASELINE prediction = `BASELINE_p_over(L_i)` (19.4), evaluated on the
exact same rows (a genuinely paired comparison).

PRIMARY metric: rung-level Brier score delta, `delta = Brier_ladder -
Brier_baseline` where `Brier_x = mean((p_x - label_over)^2)` over the
population, computed within each bootstrap resample. Negative = ladder
more accurate than baseline (matches this family's established sign
convention, e.g. 18.5's "negative = microstructure variant more
accurate"). Goalie-night CLUSTER bootstrap: 10,000 resamples, seed 42,
resampling goalie-night clusters WITH replacement (every resampled
night's full multi-rung row set travels together, this family's
standing multi-row-per-cluster convention). No degenerate-resample rule
is needed -- a Brier-delta mean is always defined for any nonempty
resample, exactly the reasoning 18.5 already gave for its own Brier/MAE
bootstrap (not a Pearson-`r`-style statistic that can be undefined).

**PASS bar, stated with the exact arithmetic and the exact one-sided
direction** (registered precisely because 18.9 disclosed a near-miss: an
early implementation of this exact bar shape was caught pre-execution
having been coded two-sided by mistake). Compute the standard percentile
CI95 (2.5th/97.5th percentile of the 10,000 resampled deltas). PASS
requires `ci_upper < 0` -- the UPPER bound of the two-sided-computed
interval below zero, a one-sided-in-effect bar from a two-sided-computed
CI, identical in construction to 17.3's and 18.5's own bars. The runner
MUST implement this as `ci_upper < 0`, never as `not (ci_lower <= 0 <=
ci_upper)` (equivalent in principle but registered explicitly given the
disclosed 18.9 near-miss).

SECONDARY, reported unconditionally, non-gating: the identical
construction using rung-level log-loss instead of Brier (`-[label*ln(p)
+ (1-label)*ln(1-p)]`, both `p` clipped to `[1e-6, 1-1e-6]` to avoid an
unbounded value from a single degenerate prediction -- a numerical
safeguard, not a modeling choice); the raw (non-clustered) count of
monotonicity clips applied (19.4); the anchor-rung wiring/consistency
diagnostic (19.4).

**Exact PASS/FAIL/INSUFFICIENT-SAMPLE arithmetic.**

1. If the sampled population yields zero qualifying goalie-nights: HARD
   STOP, overall verdict INSUFFICIENT SAMPLE, no bootstrap attempted.
2. Compute `coverage_rate = n_qualifying / n_total_sampled_goalie_
   nights`.
   - `coverage_rate < 0.70` -> coverage = INSUFFICIENT. The calibration
     statistic is STILL computed and reported (this family's standing
     "report and stop" discipline, e.g. 17.7's EXPLORATORY-ONLY handling
     of a failed-gate Phase B/17.5), but labeled EXPLORATORY-ONLY and
     does NOT gate anything. Overall verdict: **INSUFFICIENT SAMPLE**.
   - `coverage_rate >= 0.70` -> coverage = SUFFICIENT, proceed to step 3.
3. If `n_qualifying < 50` (the cluster floor): overall verdict
   **INSUFFICIENT SAMPLE** regardless of `coverage_rate`; point
   estimates reported, no CI trusted.
4. Else (`coverage_rate >= 0.70` AND `n_qualifying >= 50`): compute the
   PRIMARY Brier-delta cluster-bootstrap CI95.
   - `ci_upper < 0` -> **PILOT PASS**.
   - Otherwise (`ci_upper >= 0`) -> **PILOT FAIL**.

No other combination is possible; every branch above is exhaustive and
terminal.

### 19.6 Forbidden

1. No purchase beyond `--max-credits 2000` or below `--credit-floor
   10895` under this registration; raising either requires a NEW
   registration, not a flag change under this one.
2. No touching `data/betting.db`, reads included (17.6 item 2 / 18.4's
   stricter carve-out, restated as binding).
3. No modification of `src/betting/predictor.py`, any pre-existing file
   under `models/trained/` (including
   `experiment_15_w3_microstructure_20260716_124811/`, reused strictly
   read-only for its `juice_*` formula/values), or `.github/workflows/`.
4. No changing the 19.2 book scope (BetOnline-only), the vig-extension
   assumption, the monotonicity-clip rule, or the baseline formula/
   `sigma_0`-fit procedure (19.4) after seeing any purchased ladder quote
   or any outcome. All are fixed by this registration.
5. No changing the 19.3 sample sizes (120 / 35), the seeded sampling
   permutation, the 19.3 alignment tolerance (300 seconds), the 19.5
   rung-depth floor (5) or coverage-gate bar (70%), or the 19.5 cluster
   floor (50) or PASS bar (`ci_upper < 0`) after seeing any result. No
   opportunistic use of the 100-credit headroom (19.3) to enlarge the
   target sample mid-run or on a rerun.
6. No re-sampling, reordering, or resizing the seeded sample to chase a
   better-looking coverage or calibration number; a run that
   under-delivers (e.g., stops early at the credit floor) is reported
   exactly as it landed, including as INSUFFICIENT SAMPLE if that is
   what results.
7. No post-hoc slicing (by book, date range, or rung position) reported
   as a result once 19.5's statistics are computed.
8. No treating the 26 already-owned probe goalie-nights, or either
   season generally, as an untouched or confirmatory sample in any
   report -- the 19.1 honesty notes travel with every number this pilot
   produces.
9. The independent audit (19.3) must run and its integrity checks must
   be clean (or every violation individually reconciled and disclosed)
   before the analysis script (19.4/19.5) loads a single cached record;
   an audit failure is a STOP pending investigation, not a warning to
   note and proceed past.
10. Crash-rerun rule, in two parts, since this is the first experiment in
    this family to spend real credits as part of its own registration:
    - PURCHASE script: resumable by construction via its append-only
      cache (identical mechanism to `purchase_core_bettime_passes.py`)
      -- a crash simply means the next invocation skips already-cached
      signatures and continues under the SAME frozen sample plan,
      `--max-credits`, and `--credit-floor`. This is a structural
      property, not a special exception, and does NOT permit redrawing
      the sample (item 6) or raising the cap (item 1).
    - ANALYSIS/statistic script: mirrors 17.6 item 8 / 18.6's rule
      exactly -- one registered execution; if it crashes mid-run, it may
      be fixed and rerun ONLY if NO registered 19.5 statistic (the
      coverage rate, the calibration PRIMARY CI, or the SECONDARY
      log-loss delta) was yet computed and printed/logged; otherwise the
      computed numbers stand as-is and must be reported. As with 17.6
      item 8 and 18.6, there is no virgin season to protect here (19.1),
      so this rule exists for procedural discipline and artifact
      hygiene, not touch-consumption protection, and no 12R-style
      recovery machinery applies.
11. New artifacts only, under a new directory
    (`models/trained/experiment_16_alt_ladder_pilot_<timestamp>/`); no
    writes to any existing parquet or any existing `models/trained/`
    directory.

### 19.7 Consequence mapping (fixed in advance)

- **PILOT PASS** -> unlocks DRAFTING a full-season alternate-ladder
  purchase registration as a candidate next document, using this
  pilot's own verified season-mix reasoning (19.3), ladder model (19.4),
  and any calibration lessons learned, as its starting point. This does
  NOT itself authorize the further purchase -- exactly as section 5.7's
  own core-purchase authorization required a separate, explicit user
  allocation decision after its probe cleared its gates, the full-season
  purchase remains a future decision requiring the user's explicit
  sign-off, never automatic. Given the balance's 2026-07-31 expiration, a
  PASS creates real time pressure to draft that follow-up promptly, but
  this registration does not pre-authorize anything beyond the
  2,000-credit pilot itself.
- **PILOT FAIL** -> the alternate-saves ladder remainder use is **CLOSED
  this cycle**, joining this document family's existing closure
  precedents (steam-recon section 8, DFS-census 16.9, BetOnline-
  convergence 17.9, juice-skew 18.9's PRIMARY FAIL). It does not reopen
  without a new architecture or a genuinely new season of alternate-
  saves coverage.
- **Coverage SUFFICIENT but calibration PRIMARY FAILS** (the explicit
  "coverage passes but model fails" case) -> **PILOT FAIL** exactly as
  above; the ladder EXTENSION ASSUMPTION (19.4), not the market's
  existence, is what failed, and this is reported precisely that way
  rather than blurred into a data-availability finding.
- **INSUFFICIENT SAMPLE** (either the coverage gate or the cluster
  floor) -> neither closes nor opens the remainder decision; it is
  reported as a scale/design finding (mirroring 17.7's identical
  handling of 17.5's own INSUFFICIENT SAMPLE outcome). Given this pilot
  already spends the entire registered 2,000-credit budget attempting to
  avoid exactly this outcome, an INSUFFICIENT SAMPLE result is a
  genuine, disclosed possibility that must be reported as such, not
  re-run with a larger sample under this same registration (19.6) -- any
  follow-up would require a fresh registration.
- **Coverage INSUFFICIENT but the (non-gating, EXPLORATORY-ONLY)
  calibration number would have cleared PASS anyway** -> still
  **INSUFFICIENT SAMPLE** overall (per 19.5 step 2's explicit "does NOT
  gate anything" rule) -- a favorable point estimate on a too-thin or
  too-narrow sample cannot promote the pilot, mirroring 18.7's identical
  treatment of a single-surviving-gating-origin case.

### 19.8 Data inventory (verified 2026-07-16, read-only Python; no
coverage-vs-outcome, calibration, or ladder-vs-baseline statistic was
computed against any season -- every count below is a coverage, schema,
or timing join count, exactly matching 17.8/18.8's own distinction).

**Events cache.** `data/raw/betting_lines/cache/` holds 560 distinct
`events_date=*.json` envelope stamps spanning 2023-10-10 to 2026-04-16,
3,870 distinct events overall. 2024-25 window (2024-10-04..2025-04-17):
1,313 events -- matches `audit_core_bettime_passes.py`'s own
`expected_events=1313` exactly. 2025-26 window (2025-10-07..2026-04-19,
matching 18.4's registered window rather than
`probe_opening_markets.py`'s narrower 2026-04-16 end; checked both:
ending 2026-04-16 yields 1,226 events / 156 distinct game dates, ending
2026-04-19 yields 1,232 events / 158 distinct game dates -- this
registration uses the wider, more authoritative 18.4 window): **1,232
events**. Missing calendar-date stamps inside each window: 18 inside
2024-25 (holiday/All-Star-break dates, e.g. 2024-12-24/25/26,
2025-02-10..21) and 2 inside 2025-26 (2025-10-10, 2025-11-27) -- both
sets consistent with scheduled off-days rather than acquisition gaps,
and neither reduced the actual per-season event count below the true
schedule size. The events cache is a complete-enough basis to plan a
full-season-scale random sample from, for both seasons.

**Probe alternate-saves structure**
(`data/raw/betting_lines/probes/w1_market_coverage/`, 24 records, 455
alternate-saves outcome rows, all `side == "Over"`, matching 9.3 exactly).
Books present: `{betonlineag, prizepicks, fanduel, fanatics}` (no
Underdog alternate saves, matching 9.3). Rows by season: 2024-25 = 100,
2025-26 = 355. Distinct `(event, player, book)` goalie-night-book
groups: 52; rungs-per-group min 2 / max 24 / mean 8.75. Per book:
`betonlineag` n=26 goalie-nights, rungs 7-8 (mean 7.81, uniform 2.0-point
spacing on every sampled night); `fanduel` n=7, rungs 23-24 (mean
23.57); `prizepicks` n=14, rungs 2-5 (mean 4.43); `fanatics` n=5, rungs
exactly 5. BetOnline goalie-nights split by season: 2024-25 = 12 (from
7 of 8 sampled events), 2025-26 = 14 (from 8 of 8 sampled events) -- 26
total, matching 9.3's "7/8 ... 8/8" event-level finding exactly at the
goalie-night level too.

**Anchor-rung-on-ladder check.** Of the 26 already-owned BetOnline
goalie-nights with both a standard line and an alternate ladder at the
same snapshot, `L_std` is found EXACTLY among the ladder's own rungs in
26 of 26 cases (0 mismatches) -- direct evidence for the 19.2
anchor-rung definition and the 19.4 extension assumption's starting
premise.

**Cross-call snapshot-alignment check** (the load-bearing evidence for
the 19.3 alt-only-vs-combined decision). Of the probe's 24 events, 16
also appear in the separately-issued `core_bettime_202607` pass (2,624
unique event ids). For all 16 overlapping events, the two
independently-issued calls' requested `date` params were identical (by
construction) AND their returned envelope `timestamp` fields were
byte-identical in all 16 cases; of the 7 with a BetOnline standard-saves
quote present in both calls (lead-reviewer recount 2026-07-16: seven
events contributing 2+2+4+4+4+4+4 = 24 shared outcomes; the draft's
count of 6 events was an error, the 24-outcome total was exact), all 24
shared `(player, side, point)` price outcomes matched exactly (zero
diffs).

**2025-26 bettime anchor-alignment audit** (`saves_lines_snapshots.
parquet`, 781 unique 2025-26 bettime events). `requested_ts` equals
`compute_bettime_ts(commence_time)` exactly in 725/781 (92.83%); the
remaining 56 (7.17%) diverge, diff-seconds ranging -720 to +16,200
(up to 4.5 hours), median 0, mean +328.8s, std 2,016s. `resolved_ts`
trails `requested_ts` by a stable 259.6s mean (min 19s, max 263s, std
19.9s) -- the archive's usual "actual capture slightly before nominal
anchor" pattern, but this does not explain the 56-event divergence from
the anchor FORMULA itself, a different quantity. For contrast, the OLD
pre-purchase 21-event 2024-25 `bettime` fragment (already excluded from
every other experiment in this family per 14.3a/17.4/18.4) shows ZERO
exact matches to the modern anchor formula (mean diff +8,600s) across
its 21 events -- confirming that fragment's exclusion is warranted on
this same anchor-timing basis, independently of the reasons already on
record.

**`saves_lines_snapshots.parquet` structure** (79,884 rows / 15 columns,
independently reconfirmed, no new facts beyond 17.8/18.8's own
inventory): `snapshot_pass` values `{bettime: 28,751; closing: 51,133}`;
rows by pass x season: `bettime` 2023-24 15,682 / 2024-25 258 / 2025-26
12,811; `closing` 2023-24 17,959 / 2024-25 14,954 / 2025-26 18,220 --
matches 17.8/18.8 exactly.

**2024-25 anchor-availability restriction sizing**
(`core_bettime_202607_snapshots.parquet`): 1,050 events carry a
`betonlineag`/`player_total_saves` row (matches 9.4/9.5/18.8 exactly).
Of the 8 already-probed 2024-25 events, 7 fall inside this 1,050-event
set and 1 (`e1dd2bc0fa38ee53116f047cf3d0327e`) does not -- the same
event that also returned no BetOnline alternate-saves data in the probe
(9.3's "7/8" finding and this finding describe the SAME missing event,
not two independent gaps). Restricted 2024-25 sampling pool
(BetOnline-covered, minus already-probed): **1,043 events**.

**2025-26 sampling pool sizing:** 1,232 in-window events minus the 8
already-probed = **1,224 events**, unrestricted by prior standard-saves
coverage (by design, 19.3).

**`clean_training_data.parquet` schema-only check** (10,496 rows; no
`saves` values inspected or aggregated): contains `game_id`, `game_date`
(not `game_date_eastern` -- a naming difference from the other parquets,
noted for the join implementation), `season`, `goalie_id`, `saves`,
`saves_rolling_5`, and no other saves-market-derived column -- confirms
the join key and the outcome column this pilot needs both exist as
expected, and that `game_date` (not `game_date_eastern`) is the correct
column name for the `attach_game_id`-style join here.

**`betonlineag` absence reconfirmed.** Zero rows in BOTH 2023-24
`bettime` (0 of 15,682) and 2023-24 `closing` (0 of 17,959) in
`saves_lines_snapshots.parquet` -- reconfirms 18.8's own finding, the
direct basis for 19.4's book-agnostic baseline-fit workaround.

No contradiction of any established fact in this document family was
found. The one genuinely new finding not previously documented anywhere
-- the 2025-26 bettime archive's 7.17% divergence from the modern
`compute_bettime_ts` anchor formula, with some events off by hours --
directly shaped the 19.3 season-mix decision (alt-only for 2024-25,
combined for 2025-26) and is the first record of that fact in this
document family.

### 19.9 Implemented result (2026-07-17)

**PILOT FAIL (Sonnet sub-agent execution under lead-reviewer direction,
independently verified, 2026-07-17).** `scripts/purchase_alt_ladder_pilot.py`
executed under explicit user authorization against the frozen 19.3 seeded
plans -- no redraw. Leg `alt_only_2024_25` completed 120/120 planned calls
for 1,200 credits; leg `combined_2025_26` completed 35/35 planned calls for
640 credits (3 events returned zero markets, free; 32 returned both
markets). Total spend 1,840 of the registered 2,000-credit cap; balance
12,895 -> 11,055, reconciled exactly against response headers; the 10,895
floor was never approached; zero non-200 responses.

`scripts/audit_alt_ladder_pilot.py` (independent, read-only) returned
integrity CLEAN on all 155 records: every signature recomputes, the billing
formula (`x-requests-last == 10 * distinct markets actually returned`)
holds exactly on every call, the balance chain has zero breaks, zero
`apiKey` leakage anywhere in raw text, the on-disk event set equals exactly
the frozen plans' first-N permutation samples, and `alignment_gap_seconds`
is 0.0 on all 155 events -- the 2024-25 alt-only envelopes are byte-
identical in snapshot timestamp to the existing core pass, exactly as
19.8's cross-call evidence predicted. One disclosed audit correction: the
audit's first draft ordered by `fetched_at` (1-second granularity, 36
same-second collisions) and reported 22/66 false-positive chain
inversions; the constructive chain check reconciled all 155 records with
zero breaks, and the audit was rerun with `--force`.

`scripts/experiment_16_alt_ladder_pilot.py` ran as a single registered
execution, no crash; the runner's own pre-execution self-review caught and
fixed a NaN-dedup bug and a JSON-serialization issue before any statistic
was computed. Denominator: 338 goalie-nights from 169 of the 170 sampled
events -- 1 event (`bedd26005f98ad01c4994501ba28ddf3`, MTL @ TBL,
2026-04-19, the final day of the registered window) does not map into
`clean_training_data.parquet`, a disclosed deviation. Exclusion funnel
(sums exactly to 338): qualifying 264; `no_resolved_identity_or_ladder` 74;
`no_anchor` 0; `alignment_failure` 0; `rung_depth_failure` 0.

`coverage_rate = 0.7810650887573964` (264/338) -> coverage **SUFFICIENT**
(bar 0.70; the 50-night cluster floor also cleared at 264). `sigma_0 =
6.345402303064149` (fit n = 1,974 2023-24 goalie-nights, zero exact-0/1
exclusions; the registered two-path wiring gate passed with 0.0 max abs
diff between Experiment 15's persisted frame and a fresh recompute).

PRIMARY (rung-level Brier delta, ladder minus baseline; 1,823 non-anchor
rows, 264 goalie-night clusters, 10,000 resamples, seed 42): `delta =
+0.0011871742823691134`, CI95 `[-0.001705079310562343,
+0.003972697810631795]`. `ci_upper >= 0` -> **FAIL** under the registered
`ci_upper < 0` bar. SECONDARY (log-loss, identical construction): `delta =
+0.005570703535905996`, CI95 `[-0.005115547032554023,
+0.015731626782081978]`.

Monotonicity clips: 0 across all nights. Anchor-rung consistency: `L_std`
found on the ladder in 264/264 qualifying nights, with the ladder's own
price at `L_std` IDENTICAL to the standard market's price at `L_std` (0.0
discrepancy everywhere) -- BetOnline prices the alternate market's anchor
rung exactly like its standard line. Probe-reuse: 25 of the 26
already-owned probe goalie-nights qualified.

**Verdict (19.5/19.7): coverage SUFFICIENT, calibration PRIMARY FAILS ->
PILOT FAIL**, exactly the branch 19.7 names in advance -- the ladder
EXTENSION ASSUMPTION, not the market's existence, is what failed. The
market itself is excellent: 7-8 rungs on essentially every BetOnline night
in both seasons, perfect (0.0-second) snapshot alignment, and an anchor
rung priced identically to the standard line -- but the registered
constant-overround one-sided vig-extension produced non-anchor-rung
probabilities slightly WORSE than the simple `sigma_0` Normal baseline
built from the standard line alone, on both the primary and secondary
metric. Per 19.7, the alternate-saves ladder remainder use is **CLOSED
this cycle**, joining the section 8 / 16.9 / 17.9 / 18.9 closure
precedents; it does not reopen without a new architecture or a genuinely
new season of alternate-saves coverage. Per 19.1, this is development
evidence on already-viewed seasons in any case -- not stronger than that.
The lead reviewer independently reproduced every statistic above from the
persisted row-level artifacts (point estimates exact, bootstrap CIs
matching to 1e-15) and hand-verified one goalie-night's full chain from
raw record to labels at 1e-12.

Disclosed judgment calls (from the run's own `metadata.json`, summarized
faithfully): (1) the one unmapped event
(`bedd26005f98ad01c4994501ba28ddf3`) noted above; (2) the 19.3 alignment
check's standard-quote-side observed timestamp was read from `resolved_ts`
rather than `fetched_at`, per 19.3's own observed-over-nominal definition
-- an implementation reading, not a deviation from the registered rule,
recorded here because it is the same field the audit's own draft got
wrong in the opposite direction (above); (3) the anchor rung was read as
counting toward 19.5's `>= 5`-distinct-rungs floor, which proved moot (0
goalie-nights turned on the distinction either way); (4) the 2024-25
sampling-pool exclusion used all 8 already-probed 2024-25 events rather
than only the 7 falling inside the 1,050-event BetOnline-covered set,
numerically identical to the registered 1,043-event pool (19.8) either
way; (5) the audit's `fetched_at`-ordering artifact described above,
caught and corrected before being trusted; (6) zero goalie-nights carried
multiple BetOnline standard lines at the same snapshot, so the
multi-line-anchor tie-break rule was never invoked; (7) probe-reuse
qualified at 25 of the 26 already-owned goalie-nights, not 26 of 26 (the
remaining one fell into the `no_resolved_identity_or_ladder` funnel
bucket).

Consequence for the credit remainder: 11,055 credits remain, expiring
2026-07-31. With the alternate-ladder use closed, there is no
currently-registered candidate use for the remainder. Artifacts:
`models/trained/experiment_16_alt_ladder_pilot_20260717_130952/`
(`metadata.json`, `rung_level_paired_frame.parquet` 2,087 rows,
`exclusion_funnel_frame.parquet` 338 rows, `sigma0_fit_inputs.parquet`
1,974 rows, plan copies, `run_log.txt`) and
`data/raw/betting_lines/passes/alt_ladder_pilot_202607/` (155 raw records,
2 plan files, `run_log.jsonl`, `audit_summary.json`).

## 20. 2025-26 bet-time saves archive completion purchase (data acquisition, not a hypothesis test)

Registered 2026-07-24 by a Sonnet sub-agent under lead-reviewer (Claude)
direction, before any network call, any purchase, or any dry-run of any script
named in this section. This section registers a DATA-ACQUISITION purchase, not
an experiment. There is no model under test here, no metric, no pass/fail gate,
and no outcome-linked statistic computed anywhere in this section -- stated
explicitly because every other numbered section in this document (1 through 19)
registers a hypothesis, a metric, and a pass bar, and this one deliberately
does not. Its only success criterion is completeness and correctness of an
archive: does the 2025-26 `player_total_saves` bet-time line archive end up
covering the full in-window schedule at the project's own bet-time anchor, or
does it not. Nothing computed under this registration may be reported as
evidence of an edge, a calibration result, or a model improvement of any kind;
any future use of the completed archive for modeling is a separate, later
decision requiring its own registration under this document's established
discipline.

### 20.1 Purpose and honesty notes

**Purpose.** Complete and re-anchor the 2025-26 `player_total_saves` bet-time
line archive in `data/processed/saves_lines_snapshots.parquet`. As of this
registration, 2025-26 bet-time saves coverage is 781 of 1,232 in-window cached
events (~63.4% of the season), and 30 of those 781 owned events are
mis-anchored under the registered cache-anchored, min-gap-over-all-snapshots
test (20.2): none of that event's owned bettime snapshots falls within the
project's own `compute_bettime_ts` tolerance of the anchor computed from the
CACHE's own `commence_time` -- large enough, on every owned snapshot for that
event, that the archived quote was very plausibly not captured at the
intended bet-time snapshot at all. By contrast the same
archive's 2024-25 bet-time saves coverage is roughly 95% (1,244 of 1,313 events
carry a saves quote in `core_bettime_202607_snapshots.parquet`,
docs/CURRENT_HISTORICAL_DATA.md section 4.2) and 2023-24 is roughly 86% (1,125
of 1,313 in-window events, `saves_lines_snapshots.parquet`'s own `bettime` rows
-- both percentages independently reproduced while drafting this registration,
20.8; section 19.8's own 2025-26 anchor-alignment audit is the direct
predecessor finding this registration extends). This buy brings 2025-26 to
parity with the other two owned seasons so all three form a uniform,
correctly-anchored bet-time saves archive -- the substrate
`docs/CURRENT_HISTORICAL_DATA.md` sections 5 and 6 name as THE binding
constraint on this project's model training and evaluation: section 5 calls the
current two-full-season archive "workable, but thin," with a held-out
chronological test fold of only ~950-1,380 rows carrying real sampling
uncertainty on any single reported ROI number; section 6 states plainly that
walk-forward validation -- multiple rolling chronological train/test splits,
the standard mitigation for a single-cut test window being at the mercy of
whatever happened to occur in it -- cannot be run meaningfully with only two
seasons of history. A complete, uniformly-anchored three-season bet-time
archive is a direct, disclosed prerequisite for eventually attempting that
walk-forward evaluation. This purchase does not itself run that evaluation,
train any model, or compute any statistic against it; it only builds the raw
substrate.

This is explicitly value-preservation, not an edge test, and every honesty
constraint below is stated plainly per CLAUDE.md's real-money standard rather
than rounded up into something it is not:

1. This purchase will not itself produce a betting edge, a model, or a
   calibration result of any kind. Success is defined entirely as archive
   completeness and correct anchoring -- nothing more. Any future modeling use
   of the completed archive is a separate decision, requiring its own
   registration, its own hypothesis, and its own pass bar under this document's
   established discipline; this section pre-authorizes none of that.
2. The worst-case credit figure for this buy set is 4,810 (481 x 10) -- a
   true ceiling, not an estimate of expected spend, because a zero-market or
   event-not-found response bills zero credits (verified repeatedly elsewhere
   in this document family: 9.1, 17.8, 19.3) -- so the actual number of
   newly-usable events this purchase produces is genuinely uncertain and will
   very likely land below the full 481-event buy set actually returning usable
   data, even though all 481 will be attempted. `REGISTERED_MAX_CREDITS = 5000`
   (20.3) is a separate, round safety cap set deliberately above this 4,810
   worst case, not a claim that 5,000 credits of spend is expected. The
   expected YIELD, separately, is well-supported rather than a guess: of the
   451 truly-missing events in the buy set (defined in 20.2/20.3), 431 (95.6%)
   already carry a *closing*-pass saves line in the same
   `saves_lines_snapshots.parquet` archive, which is direct, already-owned
   evidence that the `player_total_saves` market existed and was quoted for
   that game -- it just was not captured at the bet-time snapshot. This is not
   proof every one of those 431 will also return a bet-time quote (closing and
   bet-time are different snapshots of a possibly different-shaped market), but
   it is a real, disclosed basis for expecting meaningful yield rather than a
   hopeful one.
3. The 2023-24 season's already-owned bet-time saves lines are NOT currently
   ingested into `clean_training_data.parquet` or any training pipeline
   (docs/CURRENT_HISTORICAL_DATA.md, `merge_betting_lines.py`'s current scope).
   That ingestion is free -- no credits, no new purchase -- and is a separate,
   later follow-on task, explicitly out of scope for this purchase
   registration. This registration buys new 2025-26 raw archive coverage only;
   it does not ingest anything, 2023-24 or otherwise, into any training-facing
   parquet.
4. This registration itself was drafted from read-only inspection of
   already-owned data only: no network call, no credit spend, and no purchase
   or dry-run of any script was executed anywhere in producing this section.
   Every count in 20.8 is a coverage or schema join count, exactly matching the
   distinction this document family has drawn since 17.8/18.8/19.8.
5. The Odds API credit balance funding this purchase (11,055 as of section
   19.9, 2026-07-17) expires 2026-07-31. As of this registration's filing date
   (2026-07-24), seven days remain. This creates real, disclosed time pressure
   to execute promptly once authorized, but time pressure is not a reason to
   relax any locked number below -- the credit floor, the cap, and the buy set
   are fixed regardless of how many days remain.

### 20.2 Registered definitions

**Universe.** Every in-window 2025-26 cached event from
`data/raw/betting_lines/cache/`, computed via the existing
`_season_events(load_cached_events(CANONICAL_EVENTS_CACHE), "2025-10-07",
"2026-04-19")` helper, imported (not reimplemented) from
`scripts/purchase_core_bettime_passes.py` -- the same window section 19.8
already established as the wider, more authoritative 2025-26 window versus
`probe_opening_markets.py`'s narrower cutoff, restated as binding here. Sorted
by `(commence_time, event_id)`, exactly as `_season_events` already sorts.
Verified size: **1,232** (reproduced exactly via read-only Python while
drafting this registration, 20.8).

**Correctly-anchored owned set.** From
`data/processed/saves_lines_snapshots.parquet`, ALL rows (no dedup -- see
below for why) with `snapshot_pass == "bettime"` and `event_id` in the
universe above. The anchor is computed from the CACHE's own `commence_time`
-- `compute_bettime_ts(commence_time)` using the `commence_time` field from
the cached events-list envelope (`data/raw/betting_lines/cache/`), the same
source the purchase itself anchors from -- NOT the `commence_time` column
carried inside `saves_lines_snapshots.parquet` itself. This is a deliberate
correction, not a stylistic choice: the snapshot archive's own `commence_time`
disagrees with the cache's `commence_time` by up to 30 minutes on 85 in-window
events, and that disagreement changes which side of the 300-second tolerance
an event lands on for 36 of them (20.8) -- an "already own it?" test must test
the anchor the purchase itself will buy at, not a possibly-stale value carried
in the archive being audited. For each event, let its OWNED SNAPSHOTS be every
`bettime` row for that `event_id` (an event may carry more than one, 20.8's
68-event finding), and `gap_seconds(row) = |Timestamp(row.requested_ts) -
Timestamp(compute_bettime_ts(cache_commence_time))|` for each. An event is
correctly-anchored iff AT LEAST ONE of its owned snapshots has `gap_seconds <=
300` (5 minutes, the same tolerance this document family already established
as its "material drift" threshold in `audit_core_bettime_passes.py`'s
`check_anchor_integrity` and reused verbatim in 19.3, not a new number
invented here) -- a MIN-GAP-OVER-ALL-SNAPSHOTS test, not a
single-representative-row test. No dedup rule is applied or needed: this test
is order-independent by construction, since it asks whether ANY owned
snapshot qualifies, not which ONE row a dedup convention happens to keep. The
19.2 natural-key dedup rule (max `resolved_ts`, tie-break max `requested_ts`)
remains correct and binding for its own original purpose elsewhere in this
document family, but is explicitly NOT used for this "already own it?" test
-- it is the wrong tool here, since it answers "what value should stand in
for this event's row" (a downstream-consumer concern) rather than "do we
already own a correctly-anchored snapshot for this event" (this test's only
question). Verified size: **751**.

**Buy set.** Universe event ids MINUS correctly-anchored event ids, preserving
the universe's `(commence_time, event_id)` sort order -- a plain
set-difference, not a random sample, and not itself re-sorted by any other
criterion. Verified size: **481**, composed of two disjoint groups: 451 events
with NO existing 2025-26 bettime row at all ("truly missing"), plus 30 events
where EVERY existing owned bettime snapshot is mis-anchored (`gap_seconds >
300` on all of that event's owned snapshots, per the min-gap-over-all-
snapshots test above; correctly-anchored count 751 plus mis-anchored count 30
equals the full owned 781, and universe 1,232 minus correctly-anchored 751
equals the buy set 481 -- all three identities checked and hold exactly). The
30 mis-anchored events are deliberately re-bought under a fresh,
correctly-anchored call rather than left as-is or silently dropped: their
existing rows remain on disk untouched (this is an append-only purchase, 20.5),
so after this purchase completes, a downstream consumer must still apply the
SAME min-gap test (not assume every 2025-26 bettime row is trustworthy, and
not assume any single dedup convention picks the right row) to select a
usable snapshot per event -- a follow-on selection-logic note, not something
this purchase itself resolves.

`EXPECTED_BUYSET_SIZE = 481` is registered as a fail-loud hard-stop constant:
if a freshly built buy set (recomputed by the purchase script from the live
cache and archive state at invocation time) does not equal 481 exactly, the
script MUST raise and halt before any network call, rather than silently
proceeding with a different-sized set -- the identical STOP-and-investigate
discipline `purchase_alt_ladder_pilot.py`'s own `EXPECTED_POOL_SIZE` gate
already established in this document family (section 19.3): either the buy-set
logic here has a bug, or the underlying cache/archive has drifted since this
registration was filed, and either way that is a stop, never a silent
continue. A sha256 checksum of the frozen buy set's own sorted event-id list
(20.8) provides a second, stronger reproducibility check beyond the count
alone.

### 20.3 Purchase parameters (locked)

- **Market: `player_total_saves` ONLY.** No other market is requested by this
  purchase -- not `player_total_saves_alternate`, not `player_shots_on_goal`,
  not any combined-market call. Up to 10 credits per event (10 credits x 1
  distinct market actually returned, the same per-market billing rate verified
  repeatedly elsewhere in this family; an event returning zero markets bills
  zero).
- **Anchor: `compute_bettime_ts(commence_time)`** -- the min of 22:30Z on the
  Eastern game date, or commence time minus 30 minutes. Imported, not
  reimplemented, from `scripts/purchase_core_bettime_passes.py` (20.2).
- **Books: the nine-book `BOOKMAKERS` set**, imported verbatim from
  `scripts/purchase_core_bettime_passes.py`: `draftkings, fanduel, betmgm,
  williamhill_us, fanatics, bovada, betonlineag, underdog, prizepicks`. Billing
  depends only on distinct markets returned, never on book count (9.1/17.8,
  reconfirmed repeatedly), so requesting all nine costs nothing extra and
  preserves the full multi-book archive shape this project's other passes
  already carry.
- **Worst case: 481 x 10 = 4,810 credits.**
- **`REGISTERED_MAX_CREDITS = 5000`** -- a hard cap, never raisable by any
  command-line flag, and deliberately left at this round number even after
  the buy set's own worst case was corrected downward to 4,810 (from an
  earlier 500-event/5,000-credit design where the cap and the worst case
  coincided exactly): 5,000 is a safety cap with genuine headroom above the
  actual 4,810 worst case, not a value re-tuned to track the buy set size.
  Raising it requires a new registration, not a flag change under this one.
- **`REGISTERED_CREDIT_FLOOR = 6055`** (= the last recorded balance, 11,055 as
  of section 19.9 / 2026-07-17, minus the 5,000 `REGISTERED_MAX_CREDITS` cap
  -- not minus the 4,810 worst case, so the floor still reserves the full
  cap's headroom regardless of the buy set's actual size). The purchase
  script aborts the instant any call's LIVE `x-requests-remaining` response
  header falls below this floor -- never a locally-estimated or assumed
  figure, and never
  lowerable by any flag. Disclosed explicitly: if the account balance has
  already moved since 2026-07-17 for reasons outside this document's tracking,
  `REGISTERED_CREDIT_FLOOR` remains fixed at 6055 regardless -- it is not
  recomputed against whatever the live balance turns out to be at execution
  time. If the live starting balance is already at or below 6055 plus the
  worst-case reservation for even a single event, the very first dispatch
  attempt aborts immediately, exactly as intended; this is a floor on remaining
  credits, not a promise that 5,000 credits of headroom will still exist when
  the script runs.
- **New dedicated append-only cache directory:**
  `data/raw/betting_lines/passes/saves_fill_2526_202607/`. This directory name
  cannot collide with any prior pass's cache directory in this project
  (`core_bettime_202607`, `alt_ladder_pilot_202607`, `w1_market_coverage`), per
  this family's established never-shared-cache-naming convention.
- **Record naming:** `savesfill_event={event_id}_signature={signature}.json`,
  matching this family's existing
  `core_event=...`/`altladder_event=...`/`w1_event=...` shape while remaining
  structurally distinct from all three.
- **Frozen plan artifact:** `plan_saves_fill_2526.json`, written inside the
  pass cache directory on first invocation (dry-run or execute) and loaded,
  never recomputed, by every later invocation -- the identical frozen-plan
  mechanism `purchase_alt_ladder_pilot.py` already established
  (build-or-load-plan, 19.3), applied here to a single buy set rather than two
  per-season legs, since this purchase has exactly one market, one season, and
  one buy set.
- **Run log:** `run_log.jsonl`, appended once per dispatched call, inside the
  same pass cache directory.

### 20.4 Script and cache discipline

The purchase script (`scripts/purchase_2526_bettime_saves_fill.py`, a new
dedicated script, not a mode of any existing script) mirrors
`scripts/purchase_core_bettime_passes.py` and
`scripts/purchase_alt_ladder_pilot.py` exactly, restated here as binding
requirements rather than left implicit:

1. **Dry-run is the default.** No network call without `--execute`, and
   `--execute` additionally requires `--max-credits` to be passed explicitly
   (never assumed from `REGISTERED_MAX_CREDITS` silently) -- the runner must
   state the cap it intends to honor on every executing invocation, exactly as
   the two mirrored scripts already require.
2. **The plan is built entirely from the cached events-list envelopes** under
   `data/raw/betting_lines/cache/` (`events_date=*.json`), NEVER from
   `data/betting.db` -- reused verbatim from both mirrored scripts' own
   standing rule (17.6 item 2 / 18.4 / 19.6 item 2, restated as binding here).
3. **Worst-case per-event credit is reserved BEFORE dispatch, not after** --
   the identical "may have billed even if the connection dies" reasoning
   `purchase_core_bettime_passes.py.execute()` already implements, reused
   verbatim.
4. **A previously-recorded signature (any complete response, 200 or non-200) is
   never re-requested.** The signature is a SHA-256 hex digest of the request's
   own parameters (event id, market, anchor timestamp, book list), the
   identical construction `request_signature()` already implements in
   `purchase_core_bettime_passes.py`; a cache hit on a matching signature is
   treated as already spent regardless of which invocation produced it.
5. **The buy set is frozen to the plan artifact (`plan_saves_fill_2526.json`)
   on first invocation** and every later invocation loads that frozen file
   rather than recomputing the buy set from scratch -- so the buy set can never
   be reordered or resized after this registration is filed, mirroring
   `purchase_alt_ladder_pilot.py`'s own build-or-load-plan mechanism (19.3)
   exactly. The ONE exception to "recompute" is the `EXPECTED_BUYSET_SIZE`
   fail-loud check (20.2/20.3), which runs once, at first-build time only,
   before the plan is frozen -- never on a later invocation that is loading an
   already-frozen plan.
6. **The API key is loaded from the `API_KEY`/`THE_ODDS_API_KEY` env var or
   `.env`**, exactly as the existing scripts already do; it is never printed,
   logged, or written into any cache record, run-log line, or plan artifact.
7. **Resumable by construction.** Because the cache is append-only and keyed by
   signature, a crash mid-run simply means the next invocation skips
   already-cached signatures and continues under the SAME frozen plan,
   `--max-credits`, and `--credit-floor` -- a structural property of the
   append-only cache design, not a special exception, and it does not permit
   redrawing the buy set (20.2) or raising the cap (20.3) on resume. This is
   the identical crash-rerun property this document family established for its
   first real-money purchase script (section 19.6 item 10, purchase-script
   half).

### 20.5 Audit plan (mandatory before any downstream ingestion)

An independent, read-only audit script
(`scripts/audit_2526_bettime_saves_fill.py`, a new dedicated script,
mirroring `scripts/audit_core_bettime_passes.py` and
`scripts/audit_alt_ladder_pilot.py`) MUST run and its integrity checks MUST be
clean -- or every violation individually reconciled and disclosed -- before any
downstream ingestion (into `clean_training_data.parquet`, any training
pipeline, or any future model) consumes a single record from this purchase.
This mirrors the STOP-AND-INVESTIGATE discipline this document family has
enforced since section 18.3a and restated in section 19.3/19.6 item 9. The
audit recomputes everything from the raw records in
`data/raw/betting_lines/passes/saves_fill_2526_202607/` only -- it never trusts
the purchase script's own run-log summary. Specifically, read-only from the raw
records:

1. **Signature reproduction.** Every stored record's filename-embedded
   signature is recomputed from the record's own request parameters and must
   match exactly.
2. **Billing identity.** `x-requests-last == 10 * distinct markets actually
   returned` on every call, checked individually on every single record -- not
   just in aggregate.
3. **Balance chain, via a CONSTRUCTIVE/SEQUENTIAL check using response/sequence
   order** -- NOT by sorting on `fetched_at`. This is registered explicitly as
   a known pitfall, disclosed because it already produced a false-positive
   audit failure in this exact document family: `audit_alt_ladder_pilot.py`'s
   first draft ordered by `fetched_at` (1-second timestamp granularity) and
   reported 22 of 66 balance-chain steps as broken purely from same-second
   collisions reordering records that were actually dispatched and billed in
   the correct sequence (section 19.9's disclosed audit correction). The
   constructive/sequential chain check that replaced it is the one registered
   here from the start, not discovered after a false alarm this time.
4. **Zero `apiKey` leakage** in any stored record, checked as a raw-text
   substring search across every cached file, not merely the parsed JSON
   structure.
5. **On-disk event set exactly equals the frozen plan's buy set** -- no more,
   no fewer, no substituted event ids -- checked as an exact set-equality
   comparison against `plan_saves_fill_2526.json`, not a count-only check.
6. **Per-event `alignment_gap_seconds`** between the returned envelope's own
   observed timestamp and `compute_bettime_ts(commence_time)`, independently
   recomputed from the raw records for every purchased event -- the same
   anchor-gap quantity 20.2 uses to define correct anchoring, now verified
   end-to-end on what was actually captured rather than merely what was
   requested.

### 20.6 Forbidden

1. No re-request of a previously-recorded signature (20.4 item 4).
2. `--max-credits` may not exceed 5,000 and `--credit-floor` may not be set
   below 6,055 under this registration; raising either requires a new
   registration, not a flag change under this one (20.3).
3. The frozen buy set (20.2/20.4 item 5) is never redrawn, reordered, or
   resized after this registration is filed, on any invocation, for any reason
   including a partial or crashed run.
4. No reads or writes to `data/betting.db` anywhere in this purchase or its
   audit (17.6 item 2 / 18.4 / 19.6 item 2, restated as binding).
5. The audit (20.5) must pass -- or have every violation individually
   reconciled and disclosed -- before any ingestion consumes these records. An
   audit failure is a STOP pending investigation, never a warning to note and
   proceed past.
6. On any non-200 response (other than a cached, free `EVENT_NOT_FOUND` 404) or
   any post-dispatch network exception, the run aborts without retry -- a
   dispatched request may already have billed, and a blind retry risks
   double-billing an event that in fact succeeded server-side but failed to
   acknowledge client-side. This is the identical "may have billed even if the
   connection dies" caution already governing 20.4 item 3, extended here to the
   abort-without-retry consequence explicitly.
7. No modification of `src/betting/predictor.py`, any file under
   `models/trained/`, or any file under `.github/workflows/`.
8. No treating this purchase's completion, by itself, as authorization for any
   downstream modeling, ingestion, or walk-forward validation attempt -- 20.1
   item 1 is binding: this section registers the buy only, not any use of what
   it buys.

### 20.7 Consequence mapping (fixed in advance)

Because this is not a hypothesis test, there is no PASS/FAIL verdict to map.
The consequence mapping here is a completion/audit-clean gate, not a
statistical one:

- **Purchase completes the full 481-event buy set (worst case 4,810 credits,
  within the 5,000-credit `REGISTERED_MAX_CREDITS` cap), and the audit (20.5)
  is clean** -> the 2025-26 bet-time saves archive is complete and uniformly
  anchored alongside the existing 2023-24 and 2024-25 coverage. This does NOT
  itself authorize any ingestion, modeling, or walk-forward-validation attempt
  against the completed archive -- that remains a separate future decision
  requiring its own registration, exactly as section 19.7 held the
  ladder-pilot's own PASS did not itself authorize the further full-season
  purchase it would have unlocked. It also does not retroactively resolve the
  20.1 item 3 follow-on (2023-24 ingestion), which remains a distinct,
  separate task.
- **Purchase stops early at the credit floor (6,055) before completing the full
  481-event buy set** -> reported exactly as it landed: however many events
  were actually purchased, at whatever partial coverage that produces, with no
  re-draw and no opportunistic resizing of the buy set to "finish" under a
  different registration's authority. A genuine partial-completion outcome is a
  legitimate, disclosed result of this registration's own credit-floor safety
  design, not a failure to be hidden or quietly re-attempted outside this
  document's discipline. Completing the remainder, if desired, requires a new
  registration.
- **Audit (20.5) finds one or more violations** -> STOP pending investigation.
  No ingestion proceeds from any record until every violation is individually
  reconciled and disclosed, mirroring 20.6 item 5. A clean re-audit after a
  disclosed, reconciled correction (the same shape as section 19.9's own
  `--force` re-audit after its disclosed `fetched_at`-ordering artifact) is
  permitted; a re-audit that simply reruns without disclosing what was found or
  fixed is not.
- **The Odds API balance expires (2026-07-31) before this purchase executes or
  completes** -> the unspent portion of the 5,000-credit cap is simply lost
  with the rest of the expiring balance; this registration does not extend,
  renew, or reallocate credits, and a purchase that never executes under this
  registration produces no archive change and no consequence beyond the honesty
  note already disclosed in 20.1 item 5.

### 20.8 Data inventory (verified 2026-07-24, read-only Python; no purchase, dry-run, or network call was made in producing any count below)

**Universe.** `_season_events(load_cached_events(CANONICAL_EVENTS_CACHE),
"2025-10-07", "2026-04-19")`, reproduced by importing the unmodified helpers
directly from `scripts/purchase_core_bettime_passes.py`: **1,232 events**,
matching section 19.8's own count for the identical window exactly.

**Owned 2025-26 bettime coverage.**
`data/processed/saves_lines_snapshots.parquet` filtered to `snapshot_pass ==
"bettime"` and `event_id` in the universe above: **781 distinct events**
(`nunique()` on `event_id`), matching this registration's stated 781/1,232
(~63.4%) figure exactly and consistent with section 19.8's own 63.7% figure for
the same underlying quantity computed slightly differently there (781/1,224,
the pool size after excluding 8 already-probed events, versus 781/1,232 here
with no such exclusion -- both describe the same 781-event owned set).

**Truly-missing events.** Universe events with NO bettime row at all: **451**,
verified directly (1,232 universe events minus 781 owned events with any
bettime row, cross-checked by explicit set difference rather than arithmetic
alone).

**Closing-line overlap on truly-missing events.** Of the 451 truly-missing
events, **431** (95.6%) already carry at least one `closing`-pass row in the
same archive for that `event_id` -- direct, already-owned evidence the market
existed for that game, the basis for 20.1 item 2's yield disclosure.

**Cache-vs-snapshot `commence_time` disagreement (the basis for 20.2's
cache-anchoring correction; provided by the lead reviewer's corrected
resolution, not independently re-derived in this drafting pass).** Of the 781
owned 2025-26 bettime events, 85 carry a `commence_time` in
`saves_lines_snapshots.parquet` that disagrees with that same event's
`commence_time` in the cached events-list envelope
(`data/raw/betting_lines/cache/`) by a nonzero amount, up to 30 minutes. Of
those 85, 36 events flip their correctly-anchored/mis-anchored classification
depending on which `commence_time` (cache vs. snapshot) the anchor is computed
from -- confirming this is not a cosmetic discrepancy but one that materially
changes which events this purchase buys. 20.2's rule anchors from the cache's
`commence_time` exclusively, since that is the value the purchase script itself
anchors its live requests against; anchoring the exclusion test against a
different, possibly-stale `commence_time` would test the wrong thing.

**Mis-anchored count, correctly-anchored count, and buy-set size -- RESOLVED,
not an open ambiguity.** An earlier draft of this section computed the
correctly-anchored/mis-anchored split via a single-representative-row dedup of
each event's owned bettime snapshots (19.2's max-`resolved_ts` rule, applied
outside its original purpose) and anchored against the snapshot parquet's own
`commence_time` column. That approach was the WRONG TOOL for an "already own a
correctly-anchored snapshot for this event?" test, for two independent reasons,
both caught and corrected before any purchase: (1) which single row a dedup
convention keeps is an arbitrary implementation choice when an event carries
multiple owned snapshots at different `requested_ts` values (68 of the 781
owned events do); the right question is whether ANY owned snapshot qualifies,
not which one a dedup rule happens to surface -- quick checks under a couple of
plausible dedup orderings during initial drafting landed anywhere from roughly
683 to 751 correctly-anchored events depending purely on dedup-ordering choice,
which is exactly the symptom of using the wrong tool, not a genuine data
ambiguity; (2) anchoring against the snapshot parquet's own `commence_time`
rather than the cache's `commence_time` (the actual anchor the purchase itself
buys at) is independently wrong regardless of dedup, since the two disagree by
up to 30 minutes on 85 in-window events and that disagreement flips the
correctly-anchored/mis-anchored classification on 36 of them (above). The
AUTHORITATIVE rule, now locked (20.2): for each event, take the MINIMUM
`gap_seconds` over ALL of that event's owned bettime snapshots, computed
against `compute_bettime_ts` of the CACHE's own `commence_time`; the event is
correctly-anchored iff that minimum is `<= 300`. This rule is order-independent
by construction -- it does not dedup anything, so there is no ordering choice
left to make, and the earlier 683-751 spread cannot recur under it. Applied to
the 781 owned events: **751 correctly-anchored, 30 mis-anchored** (`751 + 30 =
781`, `1,232 - 751 = 481`, `451 + 30 = 481` -- all three identities hold
exactly), yielding buy set size **481** and `EXPECTED_BUYSET_SIZE = 481`. This
corrected rule and its resulting counts (751/30/481, and the 85/36
cache-vs-snapshot disagreement above) were independently reproduced and
verified deterministic by the lead reviewer, not separately re-derived by this
drafting pass: the sha256 digest of the buy set's `(commence_time,
event_id)`-sorted event-id list is `96163617c977a9c5`, fixed in the frozen plan
artifact (`plan_saves_fill_2526.json`) as the reproducibility check any future
recomputation must match. The universe (1,232), owned-event count (781),
truly-missing count (451), and closing-line-overlap count (431) were
independently reproduced directly against the live parquet and cache by this
drafting pass itself and are unaffected by this correction; only the
correctly-anchored/mis-anchored SPLIT of the 781 owned events, and the
resulting buy-set size, changed.

**2023-24 event-level bettime coverage, for context (20.1's ~86% figure).**
`saves_lines_snapshots.parquet` filtered to `snapshot_pass == "bettime"` and
`event_id` in `_season_events(load_cached_events(CANONICAL_EVENTS_CACHE),
"2023-10-10", "2024-04-18")` (1,313 in-window events, matching this
registration's own reproduction of `audit_core_bettime_passes.py`'s
`expected_events=1313`): **1,125 distinct events** carry a 2023-24 bettime row,
85.68% -- confirming the ~86% figure cited in 20.1. **2024-25 event-level saves
coverage (20.1's ~95% figure).** Cited, not independently rerun in this pass,
from docs/CURRENT_HISTORICAL_DATA.md section 4.2's own already-verified count:
1,244 of 1,313 events carry a saves quote in
`core_bettime_202607_snapshots.parquet`, 94.75%.

**Row-level bettime counts, for reference (not re-verified in this pass, cited
from docs/CURRENT_HISTORICAL_DATA.md section 4.2 and section 19.8, which
already independently verified the underlying
`saves_lines_snapshots.parquet` row counts by pass and season: `bettime` rows
2023-24 15,682 / 2024-25 258 / 2025-26 12,811).** 2024-25's low RAW bettime row
count (258) alongside its cited ~95% event-level coverage reflects that
2024-25's bettime saves data comes primarily from the separate
`core_bettime_202607_snapshots.parquet` archive (section 5/19.8), not from
`saves_lines_snapshots.parquet`'s own `bettime` rows -- a structural difference
from 2025-26, whose bettime saves data lives in `saves_lines_snapshots.parquet`
directly, exactly where this purchase's new records will also land.

No contradiction of any established fact in this document family was found
while drafting this registration.

### 20.9 Implemented result (2026-07-24)

Executed and audited the same day the registration was filed. This is a data
acquisition, so the 20.7 gate is completion + audit-clean, not a
calibration/edge verdict.

Command run (after explicit user authorization, following an explicit pause
before the dry-run AND the standard pre-execute pause):
`python scripts/purchase_2526_bettime_saves_fill.py --execute --max-credits
4810 --credit-floor 6055`.

Frozen plan: `plan_saves_fill_2526.json`, 481 events, sha256 of the
sorted-order event-id list `96163617c977a9c5` -- created at dry-run, verified
by the lead against an independent recomputation, and confirmed byte-stable on
a second dry-run (reload, not recompute). The 481 figure is the
cache-anchored min-gap-over-all-snapshots result of 20.2 (an earlier
snapshot-anchored / drop-first estimate had given 500; see 20.8).

Purchase actuals:
- 481 / 481 calls attempted and completed; `aborted_reason` None.
- **4,390 credits billed** (below the 4,810 worst case): 439 events returned
  the `player_total_saves` market and billed 10 each; 41 returned zero markets
  (free); 1 was a free HTTP 404 EVENT_NOT_FOUND (CBJ vs LAK, 2026-01-27).
- Balance **11,055 -> 6,665**; the 6,055 floor was never approached.
- 481 append-only `savesfill_event=*.json` records written; 0 non-200s other
  than the single free 404.

Audit (`scripts/audit_2526_bettime_saves_fill.py`): **VERDICT CLEAN** on all
four checks -- record integrity (481 records, 0 parse/signature/filename/param
failures, 0 apiKey leakage, on-disk event set == frozen plan buy set exactly),
billing arithmetic (billing identity `x-requests-last == 10 x distinct markets
returned` holds on every call; constructive balance-chain 0 breaks; the naive
`fetched_at`-order check logged 19 diagnostic-only inversions across 57
same-second groups, correctly ignored), non-200s (the 1 404 is zero-cost), and
alignment (all 480 resolved events at `alignment_gap_seconds` 0.0). The lead
independently recomputed every figure from the raw records: sum of
`x-requests-last` = 4,390; all 481 signatures reproduce; 0 apiKey leaks;
on-disk == plan; and the balance chain closes exactly (6,665 + 4,390 = 11,055,
matching the expected starting balance).

Consequence (20.7): completion gate PASSED. The 439 obtained saves lines are
the payoff -- once ingested into `saves_lines_snapshots.parquet`, 2025-26
bet-time saves coverage rises from ~60% (781 / 1,232) toward ~95%, uniformly
anchored across all three owned seasons and fixing the mis-anchored subset in
the same pass. Ingestion is a separate, zero-credit follow-on (not yet done at
the time of writing). **6,665 credits remain, expiring 2026-07-31, with no
further planned use.**

Artifacts (all under `data/raw/betting_lines/passes/saves_fill_2526_202607/`,
gitignored / local-only per `.gitignore`): 481 `savesfill_event=*.json`
records, `plan_saves_fill_2526.json`, `run_log.jsonl`, `audit_summary.json`.
Scripts: `scripts/purchase_2526_bettime_saves_fill.py`,
`scripts/audit_2526_bettime_saves_fill.py`.

### 20.10 Ingestion implemented (2026-07-24)

The zero-credit ingestion follow-on 20.9 flagged as not-yet-done is now
complete. `scripts/build_saves_fill_2526_snapshots.py` (a pure parser
mirroring `scripts/build_core_bettime_pass_snapshots.py`, importing
`scripts/build_odds_snapshots.py`'s canonical goalie-matching / snapshot-
classification helpers and reusing its 15-column `OUTPUT_COLUMNS` directly)
parsed the 481 raw records into
`data/processed/saves_fill_2526_202607_snapshots.parquet` -- **7,357 rows x 15
columns, schema and dtypes byte-identical to
`data/processed/saves_lines_snapshots.parquet`** (a drop-in `pd.concat`,
verified 79,884 + 7,357 = 87,241 combined rows with no dtype coercion). The
existing archive was NOT mutated -- this is a SIBLING parquet, exactly as the
core pass produced `core_bettime_202607_snapshots.parquet` (both are
`*.parquet`, gitignored / local-only per `.gitignore` line 14). The build did
NO analysis, grading, EV, or outcome join (a pure parser, 20.1 item 1).

Rows come from the 439 saves-market events only (41 zero-bookmaker + 1 free
404 produce no rows). Verification (the lead independently recomputed every
figure below from the raw records, not the build's own summary): all 7,357
rows classify as `bettime`; anchor alignment `|requested_ts -
compute_bettime_ts(cache commence)|` is 0.0s on every resolved event; goalie
match rate 97.39%; 0 null lines, 0 null prices, 0 rows with a side outside
{Over, Under}; 439 distinct events. Duplicate / conflicting-price rows in the
raw data (20 rows across 10 groups, all `prizepicks` or `bovada`) are
documented, not dropped, matching `build_odds_snapshots.py`'s keep-and-
document convention. `includeMultipliers=true` was requested but 0 outcomes
carried a non-null multiplier, so the 15-column schema's omission of that
field loses nothing here.

Coverage jump (registered 20.2 cache-anchored, min-gap-over-all-snapshots,
300s-tolerance test, independently reproduced by the lead directly from the
raw records): 2025-26 bet-time saves correctly-anchored events rise from **751
/ 1,232 (60.96%)** -- reproducing the registered 751 exactly -- to **1,190 /
1,232 (96.59%)** on the union, delta +439. That is above the ~95% target and
now the HIGHEST of the three owned seasons (2023-24 ~86%, 2024-25 ~95%).

Honesty refinement on 20.9's "fixing the mis-anchored subset in the same
pass": of the 30 previously mis-anchored owned events (all part of the buy
set), 24 received a new correctly-anchored bettime row and are now correct;
the other **6 returned zero bookmakers on the re-buy and remain mis-anchored**
(event ids `4c4565dff6298c0a63737d6d06135f30`,
`589e8e7f45909b6c4b71110f0aa61d3b`, `784eb5fb32a5372dd14e963999989b61`,
`9dbcbdb73b68acce76356cae763f2e8c`, `c3a3f59b3617c326c0ae314afa582a30`,
`f2bdb7d7350b912047bec931086476bd`). So the buy fixed 24 of 30 mis-anchored
events, not all 30 -- a disclosed data outcome (those 6 games simply had no
bet-time saves quote available at re-buy), not a script defect. Per 20.2's own
note, a downstream consumer must still apply the same min-gap selection to
pick a usable snapshot per event; the 6 remain flagged by that same test.

Artifacts: `scripts/build_saves_fill_2526_snapshots.py`,
`data/processed/saves_fill_2526_202607_snapshots.parquet` (gitignored),
`data/processed/saves_fill_2526_202607_snapshots_summary.json`. This ingestion
completes the archive substrate only; it does NOT itself run walk-forward
validation, train any model, or fold anything into a training-facing parquet
(20.1 item 3 / 20.6 item 8 -- that remains a separate future decision).
