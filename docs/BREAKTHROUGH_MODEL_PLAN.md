# Breakthrough Model Plan

Authored by Codex, 2026-07-09. Reviewed, independently spot-verified, and
merged with Claude's companion diagnostics on 2026-07-10 (sections 2 note,
3.1 note, 3.4, 4.6 additions, 4.7, 5.2a, 7 Gate C additions, and the revised
section 10 sequence). Operational constraints added 2026-07-10 after user
review (section 1a) -- they are binding on every component. This file is the
single canonical plan for the next research program; the underlying
data-acquisition record stays in `OFFSEASON_OPTIMIZATION_PLAN.md` section
3.15.

Status: proposed next research program. No new Odds API purchase, model
selection, or confirmatory test described here has been executed yet.

This plan follows the completed work in `OFFSEASON_OPTIMIZATION_PLAN.md`
sections 3.14 and 3.15. It is intentionally narrower than a general feature
roadmap. Its purpose is to pursue the most credible remaining route to an NHL
goalie saves edge without repeating the failed pattern of adding more
historical averages to the same model.

## 1. Executive decision

The next model should not primarily ask:

> Can historical goalie and team statistics predict the final saves result
> better than the closing market?

The fresh rolling-origin tests show that the current `pace_shots` recipe
cannot do that reliably. The better question is:

> Can a model combine information from several related markets and hockey
> mechanisms quickly enough to identify a stale goalie saves quote before
> that quote moves?

The proposed breakthrough candidate is an **exposure-aware, cross-market
opening-line model** with two outputs:

1. A fair goalie saves distribution built from expected goalie exposure,
   opponent shots on goal, and opponent goals.
2. A prediction of opening-to-closing line or price movement at each available
   sportsbook.

The central relationship is:

```text
Expected saves
~= expected opponent shots on goal during the goalie's exposure
   - expected opponent goals during the goalie's exposure
```

This attacks a different and more plausible source of edge: related markets
and sportsbooks incorporating the same new information at different speeds.

## 1a. Operational constraints (Claude-authored, added 2026-07-10 after user review)

This plan serves a specific bettor, not a generic one. These constraints are
binding on every component below; a strategy that violates them is
research-only, no matter how well it backtests.

- **One decision window per day.** The user works a 9-5 (US Central).
  There is no reliable intraday line watching, no bet-on-news capability, no
  multi-window execution. Occasional earlier-in-the-day looks are possible
  but inconsistent (work-schedule dependent), so nothing in this plan may
  REQUIRE an earlier window -- anything exploiting one is opportunistic
  bonus, never part of the evaluated strategy. The realistic routine is a
  single early-evening decision, which matches the existing 22:30Z
  (5:30pm ET / 4:30pm CT) pipeline anchor -- the same anchor behind the verified late-window CLV
  evidence, so the one window the user has is the one window with evidence.
  Everything upstream of it must be automated (the GitHub Actions
  fetch/predict pipeline already is).
- **Minnesota venue access only**: Underdog Fantasy and PrizePicks
  (fixed-multiplier parlay apps -- no price lever, only line selection and
  ticket construction) plus BetOnline (offshore, straight bets).
  DraftKings, FanDuel, Caesars, BetMGM, and Bovada quotes in the historical
  data are consensus inputs and research instruments, NOT executable
  venues.

Consequences:

1. The movement model (Component F) is operationally reframed: its job is
   to answer "does the 5:30pm ET window still capture the value the CLV
   audit measured, or is the move gone by then?" -- not to enable intraday
   trading.
2. Component G's implementable form is a **venue-relative filter** at the
   daily decision time: bet only when the accessible book's line/price sits
   on the favorable side of the translated cross-book consensus. BetOnline
   was the most frequent off-consensus book in the dispersion diagnostic
   (section 3.4), which cuts both ways -- the filter's job is to take only
   the favorable half and pass otherwise.
3. For Underdog/PrizePicks the only levers are line favorability vs
   consensus and ticket construction. The operational expression is the
   UNDER-weighted 3-leg construction result
   (`OFFSEASON_OPTIMIZATION_PLAN.md` section 4.2) gated by the same
   line-vs-consensus check. Whether UD/PP lines are ever exploitably stale
   at the decision time is now measurable in-season via the line-snapshot
   tracking built in roadmap item 6.
4. Context on the live record, so this plan is not misread as "start
   over": the Jan-Apr 2026 live UNDER results survived night-clustered
   re-testing and independent re-grading (section 3.11 of the offseason
   plan), and the frozen policy shows verified positive closing-line value
   net of drift. The honest status is unproven-but-supported, and the
   2026-27 shadow run with CLV tracking is the designated arbiter -- not a
   consolation prize.

## 2. Why the current recipe failed

The negative rolling-origin result is real. It is not explained by a small
test sample:

| Test season | Model-market Brier delta | Cluster 95% CI | Policy ROI |
|---|---:|---:|---:|
| 2023-24 | +0.0134 | [+0.0090, +0.0178] | -8.30% |
| 2024-25 | +0.0156 | [+0.0099, +0.0213] | -3.00% |

Positive Brier delta means the model was worse than the de-vigged market.

Further read-only diagnostics performed after that result found a specific
failure mechanism:

- Predicted shots against were approximately 1.95 shots too high in 2023-24
  and 1.85 shots too high in 2024-25.
- Expected saves were consequently approximately 1.79 and 1.75 too high.
- Save-rate bias was small. The dominant error was workload, not goalie save
  ability.
- A control ablation without the 41 MoneyPuck pace features reduced shot bias
  to approximately +0.44 and +0.03. The controls still trailed the market, so
  this is a diagnosis, not a discovered control-model edge.
- Starter shots against fell from 30.31 in 2022-23 to 29.18 in 2023-24 and
  27.41 in 2024-25, while the Corsi inputs did not fall in parallel. The
  attempt-to-SOG relationship drifted across seasons.
- Negative-binomial dispersion was fitted on in-sample training residuals.
  That made the distributions too narrow, particularly when a flexible shots
  model reduced its own training residuals.

The conclusion is not that shot-attempt data is useless. It is that raw pace
features cannot be allowed to determine the absolute SOG level without
season-aware conversion and calibration.

*Verification status (Claude, 2026-07-10):* the shots-bias, ablation, and
dispersion-fitting numbers in this section are Codex-reported from read-only
post-hoc diagnostics and have NOT yet been independently verified (doing so
requires reloading the origin model artifacts). Step 0 of the section 10
sequence covers this before any Gate A work builds on them. By contrast, the
market-data claims in section 3.1 have been verified exactly (see the note
there).

## 3. Additional diagnostic findings

These findings are exploratory and post-hoc. They motivate experiments but
must not be presented as confirmatory evidence.

### 3.1 The existing CLV test is a late-window test

Joining the frozen-policy CLV bets back to their source snapshots showed:

- Minimum lead time: approximately 15 minutes.
- Median lead time: approximately 1.7 hours.
- Maximum lead time: approximately 5.2 hours.

Therefore, the positive CLV result does not measure the first goalie saves
line posted 8-10 hours before puck drop. It shows that the policy anticipated
part of the late move.

Across all matched quotes, not only policy-selected bets:

| Season | Line changed before close | Same-line price moved at least 1 probability point |
|---|---:|---:|
| 2023-24 | 5.0% | 6.5% |
| 2025-26 | 4.2% | 14.0% |

The late market is not constantly moving, but there is enough movement to
justify testing whether direction and stale books are predictable. The true
opening window may contain more movement and remains unmeasured.

*Verified (Claude, 2026-07-10):* both claims in this subsection reproduce
exactly from `saves_lines_snapshots.parquet` with independent code: audit-
event bettime lead time min 0.25h / median 1.67h / max 5.17h (n=258 events),
and the movement-frequency table matches to the decimal (5.0%/6.5% for
2023-24, 4.2%/14.0% for the 2025-26 audit window).

### 3.2 Early exits materially affect the lower tail

In `clean_training_data.parquet`, 572 of 10,496 starts (5.45%) had goalie TOI
below 50 minutes. These include injuries and performance-related replacements;
the cause cannot be reliably separated from the current data, so the neutral
label should be `early_replacement` rather than `injury`.

Removing those games after the fact changed failed OVER results substantially:

| Test season | OVER ROI, all starts | OVER ROI excluding TOI < 50 |
|---|---:|---:|
| 2023-24 | -10.58% | -4.11% |
| 2024-25 | -8.40% | -3.49% |

This does not create an actionable filter because early replacements are not
known in advance. It shows that a single negative-binomial count process is a
poor representation of the saves distribution's lower tail.

A quick fixed exploratory classifier achieved only AUC 0.53-0.56 for
`TOI < 50`. Individual early exits are difficult to predict, but a calibrated
pooled exposure mixture can still improve the distribution.

### 3.3 New-season recalibration may recover weak ranking signal

A post-hoc diagnostic used the first 30 days of each test season to fit a
simple Platt calibration, then evaluated the remainder:

| Origin | Calibrated model Brier | Market Brier | Paired delta 95% CI |
|---|---:|---:|---:|
| 2023-24 remainder | 0.24859 | 0.24995 | [-0.00449, +0.00072] |
| 2024-25 remainder | 0.24621 | 0.24860 | [-0.00482, +0.00107] |

Both intervals cross zero. This is not proof of superiority, and the method
was chosen after inspecting the failed origins. It is evidence that the
model may contain weak ranking information hidden beneath a large seasonal
level error. A pre-registered shots-level burn-in correction is worth testing.

### 3.4 Claude-authored companion diagnostics (2026-07-10, verified, zero credits)

Three read-only measurements run independently of this plan, arriving at the
same executive decision. Full numbers and CIs live in
`OFFSEASON_OPTIMIZATION_PLAN.md` section 3.15 "Follow-up diagnostics"; the
consequences for this plan are:

1. **Fresh-origin CLV.** Origin A's 2023-24 bettime bets (the model that
   lost -6.5% ROI) still earned positive probability CLV: +0.037%, cluster
   95% CI [+0.013%, +0.061%], with the UNDER side carrying ~3x the OVER
   side's CLV. The front-running effect generalizes in sign and appears to
   scale with training-data volume (the frozen model's +0.33% came from ~3x
   the training seasons). Consequence: the movement model has a nonzero
   prior, and Gate C must report OVER/UNDER splits. Origin B cannot be
   CLV-audited until the Phase 1 purchase (2024-25 has no bettime pass).
2. **Unconditional drift baseline.** The 2025-26 audit window drifted
   +0.128% toward OVER market-wide between bettime and close (2023-24
   drifted zero). Mix-times-drift explains ~+0.07% of the frozen policy's
   +0.33% CLV; net selection is ~+0.26%. Consequence: any movement-model or
   CLV result in this program must be reported net of the unconditional
   drift baseline for its window, or it will claim market-wide drift as
   selection skill.
3. **Cross-book dispersion.** Same-line price dispersion across us books is
   far too small to exploit (mean absolute deviation from same-line
   consensus ~0.35-0.55 probability points against a typical ~3.5-point
   half-vig; only ~0.4% of bettime quotes sit 3+ points from the
   leave-one-out consensus). Line dispersion is the real raw material: ~11%
   of bettime goalie-nights (16-18% at closing) have books posting lines a
   full save or more apart, and one save of line is worth 5-6 probability
   points. Consequence: "stale book" in this plan means stale ACROSS LINES,
   requiring distribution-shape translation (Components E/G), not same-line
   price shopping. Closing-snapshot outliers (BetOnline most frequently) may
   be stale or suspended boards rather than bettable prices; bettime
   snapshots are the trustworthy pool for backtests.

## 4. Proposed model architecture

### 4.1 Component A: goalie exposure state

Model the starter's exposure as a mixture of states:

```text
early replacement
normal regulation start
regulation plus overtime exposure
```

The first implementation should remain simple:

- Binary calibrated probability of `TOI < 50`.
- Conditional TOI distribution for early replacements.
- Conditional exposure distribution for normal starts.
- Overtime probability derived from de-vigged game moneyline and total inputs,
  or a simpler historical game-state baseline if that derivation is unstable.

Use only pregame-safe features. Current-game goals, saves, shots, final starter
status, or postgame lineup information are forbidden.

The component must be evaluated on exposure log loss, Brier score, calibration,
and lower-tail saves calibration. AUC alone is not sufficient for a rare event.

### 4.2 Component B: season-aware shot funnel

Replace the direct mapping from raw pace features to starter shots against with:

```text
shot attempts
-> unblocked attempts
-> shots on goal per 60 minutes
-> shots on goal during projected goalie exposure
```

Candidate inputs already on disk include:

- MoneyPuck Corsi and score-adjusted Corsi.
- Fenwick and missed/blocked-shot information.
- Opponent and defending-team rolling SOG rates.
- Strength-state attempt volume.
- Penalty-kill and opponent power-play volume.
- Current-season league baselines and prior-season priors.

Each stage should predict a rate or a residual relative to the current league
environment. Do not train the absolute SOG level solely from pooled raw counts
across seasons.

Required ablations:

| Variant | Purpose |
|---|---|
| Existing no-pace control | Establish the lower-bias baseline |
| Raw `pace_shots` | Reproduce the known failure |
| Season-normalized pace | Test whether conversion drift is correctable |
| Explicit attempt-to-SOG funnel | Test the new mechanism |

### 4.3 Component C: market-implied game state

Use the existing timing-safe `market_game_features.parquet` to derive:

- Consensus game total.
- De-vigged home and away win probabilities.
- Favorite/underdog strength.
- Cross-book dispersion.
- Approximate opponent expected goals from total plus moneyline.

These features should enter the exposure and workload components. They should
not be treated as proof of edge by themselves. A simple exploratory linear
test found only marginal shots-MAE improvement.

### 4.4 Component D: opponent player SOG market

The highest-value proposed new feature source is historical
`player_shots_on_goal` pricing at the same timestamp as the candidate goalie
saves bet.

Construct a team-level market projection from individual skater lines:

- De-vig each player's Over/Under pair.
- Convert the line and probability into an expected SOG estimate under a
  simple count distribution.
- Aggregate listed players by team.
- Include number of listed players, role coverage, and missing-depth-player
  correction.
- Calibrate aggregate quoted-player SOG against actual team SOG using the
  development season only.

This data can encode expected lineup, injuries, scratches, power-play roles,
and usage changes without backfilling the actual postgame lineup.

The feature is valuable only if the probe shows broad enough player and book
coverage. Summing incomplete headline-player lines without a coverage model is
not acceptable.

### 4.5 Component E: fair saves distribution

Combine exposure, SOG rate, and expected goals into a simulation or analytic
mixture distribution:

```text
P(saves = s)
= sum over exposure states e:
    P(e) * P(opponent SOG - opponent goals = s | e)
```

Estimate dispersion from validation or rolling out-of-sample residuals, not
from the same training residuals used to fit the count model.

The model must output coherent probabilities for OVER, UNDER, and PUSH at any
line. Calibration and full-distribution scoring take precedence over ROI.

### 4.6 Component F: opening-to-close movement model

Train a separate model whose target is market movement, not game outcome:

```text
probability_clv = closing no-vig probability at the same line
                  - opening no-vig probability at that line
```

Secondary targets:

- Direction of price movement.
- Direction of full-save line movement.
- Whether the current book is stale relative to cross-book consensus.
- Expected best obtainable price before close.

Candidate features:

- Difference between synthetic fair probability and current book probability.
- Current cross-book line and price dispersion.
- The book's line relative to the cross-book consensus, translated across
  lines via the Component E distribution shape (per section 3.4, cross-line
  disagreement is where the raw material lives, not same-line prices).
- Which books have already moved.
- Time to puck drop.
- Player SOG, moneyline, and total movement.
- Starter-confirmation state when available live.
- Book identity and whether the user can actually bet that book.

Mandatory evaluation requirements (from section 3.4):

- Every CLV or movement result is reported net of the unconditional
  bettime-to-close drift baseline for its window, computed on all quoted
  goalie/lines, not only selected rows. A model whose CLV disappears after
  the drift correction has learned market-wide drift, not selection.
- OVER and UNDER results are reported separately. The verified prior says
  the UNDER side carries most of the recipe's CLV.

This model is successful only if it predicts movement on a later chronological
season. In-sample movement fit is not useful.

Operational framing (section 1a): the deployed question this model answers
is whether the user's single 5:30pm ET decision window still sits ahead of
the late move. If the answer is "the move happens earlier," that is a
research finding about anchor choice, not a mandate for intraday execution
the user cannot perform.

### 4.7 Component G: cross-line outlier pricing (Claude-authored, no purchase required)

A market-versus-market strategy test that needs no model edge over the
market, only over a single lagging book:

1. Pin each goalie's fair probability at each posted line from the
   cross-book consensus (the strongest predictor this project has ever
   measured -- no repo model has beaten the de-vigged consensus Brier on a
   fresh fold).
2. Translate that consensus across lines using a distribution shape (the
   Component E mixture once built, or the existing NB pmf as a first pass)
   so that a book posting 24.5 can be priced against books posting 25.5.
3. Flag quotes where a book's (line, price) pair implies a probability gap
   versus the translated consensus large enough to clear that book's vig.
4. Grade flagged bets by outcomes AND by CLV against the closing consensus.

Candidate pool from section 3.4: roughly 150-200 bettime goalie-nights per
season with a full-save line spread. Development on 2023-24 bettime and
closing passes; the single chronological test is the 2024-25 closing pass
(which already exists -- this experiment does not depend on any purchase).
The bettime version of the 2024-25 test becomes possible after Phase 1. Both
the threshold and the shape-translation method are locked on 2023-24 before
the test touch. If flagged quotes at closing turn out to be stale or
suspended boards (the BetOnline caveat in 3.4), the honest disposition is to
restrict the strategy to bettime snapshots and books the user can actually
bet.

Deployed form (section 1a): if the backtest passes its gate, this component
ships as the venue-relative filter in the daily workflow -- at the 5:30pm ET
decision time, compare BetOnline's quote and the Underdog/PrizePicks lines
against the translated cross-book consensus, and only allow bets on the
favorable side. The historical experiment exists to set that filter's
threshold honestly, not to trade books the user cannot access.

**Development-phase result (Claude, 2026-07-13, 2023-24 only, per
`docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 8): the threshold
lock FAILED -- insufficient sample.** No gap threshold on the pre-declared
grid produced the required minimum of 20 graded bettime bets; the
reported numbers at the loosest fallback threshold (0.02, 11 bettime
bets, 9 UNDER/2 OVER) are shown for visibility only and are explicitly not
a validly locked policy. Every ROI and CLV confidence interval at every
threshold crosses zero. The candidate-pool size matched the section 3.4
forecast closely (10.6% of bettime goalie-nights / 15.5% of closing
goalie-nights have a full-save-or-more line spread, vs. the ~11%/16-18%
estimate) -- the bottleneck is not pool size, it is that few individual
quotes within that pool clear a meaningful EV gap once properly
translated. Flagged bettime bets were 82% one book (fanduel), which the
book-concentration diagnostic correctly flagged as "a hypothesis about
that book," not a general cross-book mispricing finding, per the
pre-registration's own disposition rule. The mandatory drift-baseline
subtraction cross-checked the earlier diagnostic within 0.01 probability
points, confirming the join/de-vig/consensus pipeline is wired correctly
even though the headline result is weak.

**A separate, load-bearing finding: `betonlineag` has zero quotes
anywhere in the 2023-24 archive** -- BetOnline coverage in this dataset
only begins in 2024-25. This means Component G's entire deployed premise
(a venue-relative filter keyed on BetOnline, per section 1a) cannot be
evaluated at all on 2023-24, independent of whether the pricing
hypothesis itself has merit. **Per the pre-registration's single-touch
discipline, the 2024-25 closing-pass confirmatory touch does not proceed
on this unlocked policy.** Testing an ad hoc fallback threshold against
2024-25 now would be an uncontrolled second look, not a valid
confirmatory test. Before this component can be meaningfully tested
again, it needs either a materially larger flagged-bet sample (pooling
more seasons) or the 2025-26 data (where BetOnline coverage should
exist) to make the venue-accessible cut evaluable in the first place.

## 5. Odds API acquisition plan

Current documented balance: 52,265 credits.

The Odds API historical event endpoint costs up to 10 credits per returned
market, region, event, and snapshot. Empty-market responses may cost less, but
all budgets below use the conservative maximum.

No full purchase should occur until a probe is documented and reviewed.

### 5.1 Phase 0A: true-opening timing probe

Purpose: determine when goalie saves markets are actually available and choose
one reproducible opening anchor.

Sample 20-24 games, stratified by:

- 2023-24, 2024-25, and 2025-26.
- Eastern, Central, Mountain, and Pacific start times.
- Weekday and weekend.
- Matinee and normal evening games.

Query `player_total_saves` at:

```text
T-10 hours
T-8 hours
T-6 hours
T-3 hours
T-90 minutes
```

Maximum expected budget: approximately 1,000-1,200 credits.

Go gate:

- At least two usable books on 70% or more of sampled events at the chosen
  anchor.
- Both sides quoted for at least 95% of usable goalie-book lines.
- At least 10% of matched quotes show meaningful subsequent line or price
  movement.
- The chosen anchor reflects a time at which the user can realistically bet
  (the single early-evening window in section 1a), OR is explicitly
  designated research-only for measuring movement, with the operational
  anchor staying at the evening window.

If no early anchor has adequate coverage, do not purchase a full early pass.

### 5.2 Phase 0B: player SOG probe

Query `player_shots_on_goal` for 20 games at one or two candidate anchors.

Maximum expected budget: 200-400 credits.

Go gate:

- Multiple usable books on at least 70% of events.
- Both sides present on at least 95% of usable player lines.
- Reliable player-to-team matching above 98%.
- Enough listed skaters to explain a stable portion of actual team SOG.
- Coverage is not confined to only two or three star players per team.

Escalate to a 100-game pilot before a season-scale purchase if the 20-game
sample is borderline.

### 5.2a Phase 0C: alternate saves lines probe (Claude-authored)

Query the historical archive for alternate goalie saves lines (market key to
be confirmed against the API's market list, e.g. `player_total_saves_alternate`)
for 10-20 games at one anchor.

Maximum expected budget: 200-400 credits.

If alternate lines exist with usable coverage, each goalie gets a
market-quoted probability at several lines simultaneously -- the market's own
distribution curve. That directly supplies the shape-translation layer for
Components E and G without model assumptions, and multi-line quotes are
additional training rows for the fair-distribution model. If the market does
not exist historically or coverage is thin, nothing else in this plan
changes; Components E and G fall back to model-based shape translation.

### 5.3 Phase 1: 2024-25 opening saves pass

Purpose: complete a chronological opening-to-close dataset for the season that
currently has closing saves snapshots but no uniform earlier pass.

Maximum budget:

```text
1,311 events * 1 market * 1 region * 10 credits = 13,110 credits
```

Use 2023-24 to develop and lock the movement experiment. Touch 2024-25 once as
the chronological test.

### 5.4 Phase 2: player SOG development and test seasons

If the SOG probe passes, purchase one chosen anchor for:

| Season | Approximate events | Maximum credits |
|---|---:|---:|
| 2023-24 | 1,312 | 13,120 |
| 2024-25 | 1,311 | 13,110 |
| **Total** | 2,623 | **26,230** |

Use 2023-24 for feature construction and calibration. Use 2024-25 as the
single-touch chronological test.

### 5.5 Budget disposition

| Item | Maximum credits |
|---|---:|
| Opening timing probe | 1,200 |
| Player SOG probe | 400 |
| Alternate saves lines probe | 400 |
| 2024-25 opening saves | 13,110 |
| 2023-24 and 2024-25 player SOG | 26,230 |
| **Maximum planned** | **41,340** |
| **Reserve from 52,265** | **10,925** |

The reserve is intentional. Use it only for a pre-registered targeted follow-up
or preserve it for the 2026-27 shadow run. Do not spend it merely because it is
available.

### 5.6 Purchases not currently recommended

- `us2` saves history, unless a small probe finds unique books that the user
  can bet and that materially improve obtainable prices.
- More closing game totals. Existing h2h and totals coverage is sufficient for
  the first market-state experiment.
- Several full intraday snapshots. They can consume the entire balance without
  creating a clean new target.
- Team totals before testing whether existing total plus moneyline inputs are
  adequate.
- Another full season of the same conventional boxscore features.

### 5.7 Revised acquisition plan (2026-07-13: expiring balance)

The user reports approximately 50,000 credits remaining that EXPIRE at the
end of July 2026. That changes this section's decision calculus: sections
5.1-5.6 treated credits as a scarce persistent resource and gated every
purchase behind experiment evidence; expiring credits have zero value after
2026-07-31, so the correct policy is to spend nearly all of them on the
highest-information archives this month, keeping the probe-first discipline
but compressing the sequence. Verified facts feeding this revision (Claude,
cross-checked against an independent Codex research round, 2026-07-13):

- The Odds API has a `us_dfs` region with `underdog` and `prizepicks` as
  named bookmaker keys -- two of the user's three executable venues.
  Historical availability depth for these books is UNKNOWN (the props
  archive starts 2023-05-03, but DFS books were added to the platform at a
  later, undocumented date); the probe must establish per-season coverage
  empirically. Empty responses cost nothing.
- Pinnacle does not offer NHL goalie saves in any region (NHL player props
  are US-bookmakers-only; the EU prop set for NHL excludes saves). There
  is no sharp anchor to buy. Settled; do not re-check.
- `betonlineag` is in the plain `us` region alongside draftkings, fanduel,
  betmgm, and bovada -- no region multiplier for the standard book set.
- The "every group of 10 named bookmakers bills as 1 region" rule is
  documented on the standard odds endpoint, and the historical event-odds
  endpoint states it takes the same parameters, so the rule should carry
  over; it is not restated with historical cost examples, so the probe
  should still confirm actual billing from the responses' usage headers
  before any full pass relies on it. If it holds, one pass carries the us
  majors plus underdog and prizepicks at no extra cost.
- `includeMultipliers=true` returns DFS multipliers per outcome where
  available; the vendor warns DFS pricing can be indicative and
  user-dependent. Historical support: verify in the probe.
- Dense intraday snapshot passes remain ruled out (roughly 131,000 credits
  per season at hourly granularity).

Revised probe (about 1,000-1,500 credits, run first): 18-24 games
stratified across 2023-24, 2024-25, and 2025-26; markets
`player_total_saves`, `player_total_saves_alternate`,
`player_shots_on_goal` (and its alternate); named bookmakers = us majors
plus `betonlineag`, `underdog`, `prizepicks`; `includeMultipliers=true`;
anchored at the bettime convention used by the existing snapshot archive.
Gates: (a) per-season us_dfs coverage depth, (b) the billing rule on
historical endpoints read from usage headers, (c) SOG listed-skater
breadth per the 5.2 gate, (d) alternate-lines coverage per 5.2a, (e)
multiplier presence.

Revised purchase table (updated with probe actuals; final remainder mix still
requires user authorization):

| Item | Maximum credits |
|---|---:|
| Probe (completed 2026-07-13) | 800 actual |
| Player SOG, 2023-24 + 2024-25, one bettime-convention snapshot | 26,230 |
| 2024-25 saves pass at the bettime anchor, named-bookmaker set including DFS books if billing confirms (else `us` region only) | 13,110 |
| Flexible remainder: targeted over-only alternate-saves pass or another probe-informed follow-up (2023-24 DFS is unavailable) | up to 12,125 |
| **Total available from the documented 52,265 starting balance** | **52,265** |

Rationale links: the SOG purchase feeds the cross-market coherence model
(section 10 NEXT WAVE, W1) and targets the diagnosed shots-volume weakness
with the one mechanism that reproducibly helped (market information; the
market-state block passed its accuracy primary on two origins). The
2024-25 saves pass is triple-purpose regardless of outcome: it enables a
P2-style selection-over-blind-baseline test at roughly double Experiment
8's sample -- using the FROZEN Origin B market-state model, whose test
season is 2024-25 and whose training never touched it (never Origin C:
its training pool extends into February 2025 and would leak) -- gives the
exploratory BetOnline convergence lead (HISTORICAL_DATA_ANALYSIS.md
section 8) a second season, and completes a three-season bettime archive
before the 2026-27 shadow season. The DFS history census tests venue-level
staleness at the actual products bet. The live tracker analyses found
90.1%-95.2% agreement with sportsbook consensus, depending on the sample,
comparator, and deduplication rules; W2 must reconcile those definitions
before treating either rate as the prior.

All 2023-24 and 2024-25 outcomes remain viewed (section 6.1): every
positive result from these purchases is development evidence grading into
a 2026-27 shadow candidate, never immediate proof of edge.

**Probe execution actuals (Codex-authored, 2026-07-13; independently
verified).** `scripts/probe_opening_markets.py` sampled 24 games (eight per
season), at the registered bettime anchor, in one request per event containing
all four markets and nine named books. The dedicated append-only raw cache is
`data/raw/betting_lines/probes/w1_market_coverage/`; the independent rebuild is
`scripts/audit_w1_probe.py`. All 24 calls returned HTTP 200. The probe spent
800 credits, not the 960 conservative maximum, leaving 51,465. Response-header
arithmetic reconciles exactly: every call cost 10 credits times the number of
distinct markets actually returned, and nine named books billed as one
region-equivalent on all 24 calls.

| Gate | Verified result | Decision |
|---|---|---|
| Historical billing | `x-requests-last` matched `10 * returned markets` on 24/24 calls; usage deltas and remaining balance reconcile | **PASS** |
| DFS depth | No Underdog or PrizePicks data in 2023-24. PrizePicks standard saves existed in 7/8 events in both 2024-25 and 2025-26; Underdog returned no saves market in either season | **PARTIAL:** W2 can buy PrizePicks history from 2024-25 onward, not 2023-24 or historical Underdog saves |
| Standard SOG breadth | At least two usable books on 8/8 events in every season. Player-events with at least one paired O/U line: 98.53%, 100%, and 99.69%. Season-team resolution was 347/347; standard books listed a median 12-15 skaters per game and covered roughly 47%-61% of actual combined team SOG at the book-season median | **PASS:** broad enough for W1 development, with the coverage adjustment still load-bearing |
| Alternate lines | Alternate saves: none in 2023-24; BetOnline 7/8 in 2024-25 and 8/8 in 2025-26, with thinner FanDuel/Fanatics/PrizePicks coverage. All 455 alternate-saves outcomes were over-only | **PARTIAL:** useful as one-sided ladder information, not the clean two-sided probability curve originally hoped for |
| Multipliers | The field was present on all 8,825 outcomes; 222 were non-null, exclusively Underdog 2025-26 SOG. No historical saves multiplier was returned | **LIMITED:** implementation works, but it does not unlock historical DFS saves payouts |

Schema caveats are binding for later builders: deduplicate exact outcomes
(47 duplicate FanDuel SOG outcomes in the 2023-24 sample); define SOG
completeness at the player-event level because some feeds mix a paired central
line with one-sided milestones; and never treat over-only alternate saves as a
de-vigged distribution without an explicit one-sided model. Two cached events'
commence times differed from the returned odds event by about 5-8 minutes; IDs
and request signatures matched, and the effective anchors remained 25-38
minutes before the returned commence times, so this does not change a gate but
should remain visible in the season-scale timing audit.

**Purchase consequence:** the two-season sportsbook SOG pass clears its data
coverage gate, and the named-book billing assumption for the 2024-25 saves pass
is confirmed. Neither purchase has been run; both still require the user's
explicit allocation decision. The tentative 2023-24 DFS remainder is closed
because neither DFS book appears there. W2 must be narrowed to PrizePicks saves
for 2024-25/2025-26 plus prospective Underdog collection. Any remainder spent
on alternate saves should be a targeted partial pass designed for over-only
ladder quotes, not the previously imagined two-sided curve. At maximum core
cost, the exact post-probe balance leaves 12,125 credits for that decision.

**Core purchases authorized (2026-07-13).** The user authorized both core
passes. Execution uses a new dedicated script,
`scripts/purchase_core_bettime_passes.py`, following the probe's append-only
signature-cached record pattern (records under
`data/raw/betting_lines/passes/core_bettime_202607/`, never the canonical
cache naming): 2024-25 is fetched as one combined call per event
(`player_total_saves,player_shots_on_goal`, 20 credits maximum per event, one
consistent snapshot), 2023-24 as SOG-only (10 maximum), nine named books at
the bettime anchor, `includeMultipliers=true`, credit floor 11,500. Two
remainder ideas were rejected before execution: the ~11,000-credit
alternate-ladder purchase (over-only quotes need a one-sided vig model that
does not exist yet; pilot 1-2k first) and a 2024-25 closing pass (redundant --
the archive already holds 14,954 closing rows across 1,288 events for
2024-25, verified directly against `saves_lines_snapshots.parquet`; only
bettime is missing there, so CLV and the W6 second-season pairing unlock as
soon as this purchase lands). The remainder allocation is deferred until
purchase actuals are known.

**Core purchase actuals (2026-07-14, executed and independently audited).**
Both passes completed. Total spend 38,570 credits (60 smoke + 25,390
combined-2024-25 + 13,120 sog-2023-24); balance 51,465 -> 12,895, reconciled
exactly: `sum(x-requests-last)` across all 2,626 records equals the claimed
spend, every 200 response satisfies `x-requests-last == 10 * distinct
returned markets`, zero signature or parameter violations, zero API-key
leakage. Audit deliverables are persisted (`scripts/audit_core_bettime_passes.py`,
`data/raw/betting_lines/passes/core_bettime_202607/audit_summary.json`).
Headline coverage: 2024-25 has SOG on 1,301 events and saves on 1,244
(betonlineag 1,050, prizepicks 1,139, underdog 0 saves); 1,233 saves events
intersect the existing 2024-25 closing archive (the CLV/W6-usable set).
2023-24 has SOG on 1,312 events with two-plus paired books essentially
everywhere, covering 100% of the existing bettime-saves events there. Two
HTTP 404s (both free) correspond to postponed/dead games whose replacement
ids were purchased, so no game is missing.

**Core-pass ingestion complete and independently reconciled (2026-07-14).**
The binding ingestion contract is now registered in
`PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 14.5 and implemented by
`scripts/build_core_bettime_pass_snapshots.py`. The builder produced
`data/processed/core_bettime_202607_snapshots.parquet`: 413,758 rows across
23 columns (2023-24 SOG 214,252 rows / 1,310 events; 2024-25 SOG 182,686 /
1,301; 2024-25 saves 16,820 / 1,244). An independent raw-record-to-parquet
reconciliation matched all 413,758 outcome keys exactly. Three events were
excluded because the true puck-drop gap was under the registered 10-minute
floor: BUF@NJD 2024-10-05 (-24.8 minutes), BOS@PHI 2024-01-27 (5.0), and
PHI@OTT 2023-10-14 (9.2). After those exclusions, 5,282 byte-identical
outcome copies were dropped. The purchase audit's earlier 5,296 count used
a key that omitted price: on a price-aware key the all-200-record total is
5,293 exact copies, while the remaining three are conflicting-price groups.
Those three FanDuel groups (six rows) were excluded entirely, not tie-broken.
No Fanatics rows, null lines, null prices, invalid sides, or sub-10-minute
rows survived; saves-row goalie matching is 98.93%. Eleven event ids overlap
the old 21-event 2024-25 bettime fragment and must be deduplicated at join
time. Experiments 11-13 -- frozen-Origin-B P2, W1 coherence, and W2 DFS
census -- are now preregistered in sections 14-16 and data-ready. No model
experiment had touched the new price-level data at that ingestion checkpoint;
the Experiment 11 result immediately below is the first touch. The unallocated
remainder is still 12,895 credits.

**Experiment 11 frozen-Origin-B P2 re-test -- PASS (2026-07-14,
Codex-verified).** After the closing wiring gate reproduced exactly, the
new-pass-only BetOnline universe held 1,719 paired quotes and 473 frozen
UNDER selections. Selected ROI was `+12.29%` versus `+2.63%` for blind UNDER
on the same universe: delta `+9.66` points, goalie-night cluster CI95
`[+2.49, +16.72]`. Train-fitted dispersion also passed (delta `+9.47`, CI95
`[+2.35, +16.52]`). All-books outcome selection agreed. The important
counterweight is venue CLV: BetOnline full-policy CLV net of drift was only
`+0.0167` probability points, CI95 `[-0.0627, +0.0979]`, while all-books CLV
was positive. Per the preregistered consequence this promotes the frozen
UNDER mechanism only to a 2026-27 shadow candidate; it does not establish a
durable edge from a viewed 2024-25 season. Full contract, source clarification,
verification, and artifacts are in preregistration section 14.11.

**Experiment 13 W2 DFS census -- NULL (2026-07-14, Codex-verified).** The
2024-25 development season contained 443 PrizePicks deviations from 1,868
comparable goalie-nights. Its 420 gradeable non-push candidates hit 50.48%;
the even-money outcome-grade CI95 was `[-8.57%, +10.48%]`. The 2025-26
confirmation archive supplied only six deviations (five wins, CI95
`[0.00%, +100.00%]`), which did not clear the strict lower-bound bar and
cannot rescue the null development result. Per the locked consequence,
historical DFS staleness is closed for this cycle rather than promoted to a
model-EV filter. This is not PrizePicks ROI. Full result: preregistration
section 16.9 and historical analysis section 9.7.

**Experiment 12 W1 cross-market coherence -- NO OFFICIAL VERDICT,
PROMISING SHADOW EVIDENCE (2026-07-14, Codex-verified).** The 2023-24 recipe
was frozen before 2024-25. Its original touch failed before performance was
computed; its registered recovery completed the calculation but crashed
post-touch while writing metadata, consuming the only recovery. Independent
reconstruction found no arithmetic defect: OVER beat blind OVER by `+7.17`
points, CI95 `[+2.73, +11.64]`, but still lost `-8.34%`; UNDER returned
`+11.12%` and beat blind UNDER by `+9.28` points, CI95
`[+2.49, +16.02]`. Global Brier/log-loss were worse than the market and CLV
was positive but tiny. Preserve the exact recipe for 2026-27 shadowing; do
not call this an official PASS or proven edge. Full record: preregistration
section 15.11 and historical analysis section 9.8.

## 6. Experimental protocol

### 6.1 Development versus confirmation

The 2023-24 and 2024-25 outcomes have now been viewed in the rolling-origin
experiment. New modeling against them is development, even when the new input
data has not been seen before.

The honest interpretation hierarchy is:

1. Improvement on 2023-24 development data: hypothesis support only.
2. Improvement on 2024-25 after locking on 2023-24: stronger chronological
   evidence, but still conditioned on the broader project history.
3. Frozen 2026-27 shadow performance and CLV: final confirmation.

The existing December 2025-April 2026 fold is worn. It may be reported as a
secondary diagnostic but cannot be promoted back into a clean test fold.

### 6.2 No-credit experiment order

Before any full purchase, run these pre-registered ablations using existing
data:

1. No-pace distributional control.
2. Season-normalized pace and explicit attempt-to-SOG funnel.
3. Validation-fitted dispersion.
4. First-30-days shots-level correction, frozen for the rest of the season.
5. Existing game-total and moneyline features.
6. Exposure-state mixture.
7. Cross-line outlier pricing (Component G): develop and lock on 2023-24,
   single test touch on the existing 2024-25 closing pass.

Do not add all components at once. Each component must justify its inclusion
through distributional metrics before it can enter the combined model.

### 6.3 Primary metrics

Fair-distribution model:

- Paired Brier delta versus the de-vigged market.
- Full-distribution log score.
- Probability integral transform diagnostics.
- OVER and UNDER calibration separately.
- Lower-tail calibration for short starts.
- Shots-against MAE and mean bias.
- Exposure Brier and log loss.

Movement model:

- Mean probability CLV.
- Mean same-book price CLV.
- Movement-direction AUC and Brier.
- Line-movement accuracy.
- Coverage at books the user can bet.
- Goalie-night cluster bootstrap confidence intervals.

ROI remains secondary. A positive ROI with negative CLV or poor calibration
does not pass.

### 6.4 Statistical rules

- All splits are chronological.
- All feature construction is prior-only at prediction time.
- Model and threshold selection occur on development/validation data only.
- Each designated test season is touched once after the policy is locked.
- Confidence intervals cluster by goalie-night.
- Book rows from the same goalie-night are not independent observations.
- Report side, book, line, month, and time-to-game breakdowns as diagnostics,
  not as permission to select a pocket after seeing test results.
- Report missing-market and unmatched-player coverage prominently.

## 7. Decision gates

### Gate A: no-credit architecture

Proceed to paid SOG integration only if the season-aware shot funnel plus
exposure model:

- Removes the persistent positive shots bias on both historical origins.
- Improves Brier versus the no-pace control on both origins.
- Produces better lower-tail calibration.
- Uses validation-fitted dispersion without extreme edge inflation.

It does not need to beat the market yet, but it must fix the demonstrated
mechanism rather than merely move ROI.

**Verdict (Claude, 2026-07-13, six parallel Sonnet sub-agents executing
`docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md`): Gate A FAILS.** Full detail
in that document's section 10 (Results). Summary: the season-aware funnel
(Experiment 2) fixes bias on at most one origin at a time -- season-
normalization roughly halves Origin A's bias but barely moves Origin B;
the attempt-to-SOG funnel does the reverse -- and both variants make
lower-tail calibration worse than the plain no-pace control on both
origins. The exposure mixture (Experiment 6) fails against the correct
primary baseline (no-pace control with validation-fitted dispersion
already applied) because its exposure classifier has no real
discrimination (AUC 0.52-0.55, log loss/Brier statistically tied with a
constant base-rate guess). **Correction (2026-07-13 dual audit, Codex +
Claude): validation-fitted dispersion (Experiment 3) was originally
recorded here as the round's one unconditional pass; that was wrong
against its own registered bar and is corrected to FAIL.** The alpha
values themselves reproduce exactly across three agents, and the
lower-tail improvement is real on both origins, but the registered
PRIMARY metric (summed central 50/80 coverage deviation) WORSENS on
Origin A (3.36 -> 7.80; train-fitted A was already centrally
near-nominal) while improving on Origin B (8.43 -> 4.22). Single-alpha
NB2 cannot fit the middle and the lower tail at once; the next
dispersion treatment must be pre-registered before reuse (cross-fitted
residuals, or an explicit heavier-lower-tail shape such as a fixed
pooled early-exit mixture component, which Experiment 6 showed nearly
perfects the marginal tail). See
`docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md` section 10.1 for the full
correction record.
**Consequence: no SOG probe or purchase (sections 5.2/5.4) and no 2024-25
opening saves purchase (section 5.3) is authorized by this round's
evidence.** Market game-state features (Component C, Experiment 5) showed
a real, CI-excluding-zero improvement on Origin B only -- not a Gate A
requirement, but worth retaining as a candidate input once a real
architecture exists to attach it to. Per the pre-registration's own rule
for Experiment 1, this is recorded as a clean negative for the two
proposed fixes, not a refutation of the underlying section 2 diagnosis
(which step 0 and Experiment 1 both independently confirmed) -- re-
diagnosis, not abandonment, was the indicated next move.

**Follow-up (Codex, 2026-07-13): the fixed-offset lead was tested and
FAILED.** Experiment 9 locked the deterministic attempt-to-SOG rate as a
Poisson log offset, restricted XGBoost to the 104-column no-pace residual
feature set, and bundled the fixed-weight early-exit mixture. It failed
shots bias and Brier versus control on both origins; Origin B also
inflated the fixed-policy bet rate to 59.3%. The mixture genuinely
improved marginal lower-tail error and full-distribution log score on both
origins, but that shape gain did not translate to posted-line probability
accuracy. Gate A therefore remains failed and the fixed-offset lead is now
closed, not pending. Full verified numbers are in preregistration section
12.8.

### Gate B: data coverage

Proceed to season-scale SOG purchasing only if the probe meets the coverage
rules in section 5.2 and the aggregate player projection correlates sensibly
with actual team SOG.

### Gate C: movement-model evidence

The 2024-25 chronological movement test passes only if:

- Mean probability CLV is positive with a goalie-night cluster CI above zero
  AFTER subtracting the unconditional drift baseline for the same window and
  bet mix (section 3.4 item 2).
- Same-book price CLV agrees in direction.
- Performance is not concentrated in one book or a handful of goalie-nights.
- OVER and UNDER splits are reported; a result carried entirely by one side
  is a hypothesis about that side, not a general movement edge.
- Results remain useful at the user's real betting window and venues
  (section 1a): a single early-evening decision, executable at BetOnline,
  Underdog, or PrizePicks. Value that exists only intraday or only at
  inaccessible books does not pass this gate, however real it is.

### Gate D: bankroll use

Nothing in this plan authorizes meaningful stake scaling. The strongest result
this plan can license before next season is:

- Frozen opening-season shadow run.
- Token live stakes at books and timestamps represented in the evidence.
- Weekly review by CLV first and P/L second.

Meaningful scaling requires positive 2026-27 live CLV plus acceptable
calibration and operational coverage.

## 8. Expected implementation artifacts

Names are proposed and may be adjusted to existing repo conventions during
implementation:

```text
scripts/probe_opening_markets.py
scripts/build_opening_saves_snapshots.py
scripts/build_player_sog_market_features.py
scripts/build_exposure_features.py
scripts/experiment_exposure_shot_funnel.py
scripts/experiment_opening_movement.py
scripts/experiment_cross_line_pricing.py
scripts/check_venue_value.py
data/processed/opening_saves_snapshots.parquet
data/processed/player_sog_market_features.parquet
data/processed/exposure_features.parquet
models/trained/experiment_exposure_shot_funnel_*/
models/trained/experiment_opening_movement_*/
models/trained/experiment_cross_line_pricing_*/
```

The alternate-lines probe (5.2a) reuses `probe_opening_markets.py` with a
different market key rather than adding a separate script.
`check_venue_value.py` is the deployed venue-relative filter from
section 1a / Component G, intended to be wired into the daily
fetch-and-predict workflow before opening night so the shadow run logs its
verdicts from day one.

Raw Odds API responses remain append-only, cache-first, and backed up under the
contract in `OFFSEASON_OPTIMIZATION_PLAN.md` section 3.15.

## 9. Risks and failure interpretations

### Player SOG props are too sparse

Disposition: stop after the probe. Do not extrapolate a team projection from a
few star-player lines. Continue with the existing game-market and hockey-data
components.

### Exposure risk is not predictable

Disposition: retain a pooled calibrated early-replacement mixture if it
improves lower-tail scoring. Do not claim individualized pull/injury skill.

### Season correction improves Brier but not CLV

Disposition: the model is becoming more honest but has not found a trading
edge. Continue shadow-only.

### Movement is real but too late to execute

Disposition: evaluate the earliest anchor with adequate coverage. If the move
occurs before the user can bet, it is not an operational edge.

### One direction appears profitable

Disposition: require the effect to survive both chronological seasons and
cluster uncertainty. The known macro decline in saves can make blind UNDER
look skilled for one season.

### The combined model still trails the market

Disposition: accept the negative. Preserve remaining credits and use the
2026-27 shadow program to test the frozen production-era anomaly. More data
does not guarantee a beatable market.

## 10. Recommended execution sequence (merged 2026-07-10)

0. **DONE (2026-07-13).** Independently verified the section 2 diagnostics
   by reloading the origin artifacts. All six claims CONFIRMED (one
   downgrade: the Origin B "control still trailed market" sub-claim is
   statistically marginal). See section 2's verification note.
1. **DONE (2026-07-13).** Wrote `docs/PREREGISTRATION_NO_CREDIT_ABLATIONS.md`
   -- pre-registered pass/fail bars for all seven section 6.2 experiments,
   including Component G.
2. **DONE (2026-07-13) -- FAILED.** Season-normalized shot conversion
   fixes bias on at most one origin at a time. Validation-fitted
   dispersion (bundled into this step) was originally recorded as PASSED
   and adopted as a standing default; **corrected 2026-07-13 (dual
   audit): it FAILS its registered central-coverage bar on Origin A**
   and is a tails-vs-middle tradeoff, not a default -- see the Gate A
   verdict correction above. Two implementation deviations also recorded
   (funnel exposure stage built as starter-share instead of the
   registered per-60-times-projected-minutes construction; 4 duplicated
   z-score features) -- verdict unaffected.
3. **DONE (2026-07-13) -- FAILED.** Exposure mixture does not beat the
   correct primary baseline (no-pace control + validation-fitted
   dispersion) on lower-tail coverage; its exposure classifier has no
   real discrimination (AUC 0.52-0.55). See Gate A verdict above.
4. **DONE (2026-07-13) -- PARTIAL.** Joined the existing moneyline/total
   features (Component C). Origin B showed a real Brier improvement;
   Origin A was structurally uninformative (zero training-window market
   coverage). Not a Gate A requirement; not currently blocking anything.
5. **DONE (2026-07-13) -- LOCK FAILED, insufficient sample.** Ran the
   Component G cross-line pricing development phase on 2023-24. No
   threshold reached the minimum sample for a valid lock, and BetOnline
   (the deployment-relevant book) has zero 2023-24 coverage, so the
   2024-25 confirmatory touch does NOT proceed this round. See section
   4.7's development-phase result above.
6. **CURRENT NEXT ROUND (updated 2026-07-13 after the independently
   verified 6a readout).** All items below are zero-credit. Step 6a is
   complete; its promotion/purchase gate was not met. Purchases remain
   blocked, and step 6b is now the first implementation priority.

   6a. **DONE -- Origin C market-state replication: P1 PASS; P2
   INSUFFICIENT SAMPLE.** The run and independent Codex audit are recorded
   in preregistration section 11.12. The Origin B wiring gate reproduced
   bit-exactly. On Origin C, market-state features again beat the no-pace
   control on closing paired Brier: `-0.003111`, CI95
   `[-0.005039, -0.001192]`, so the incremental accuracy benefit is
   reproducible. The executable BetOnline selection primary produced only
   85 qualifying UNDERs, below its registered 100-bet floor; selected ROI
   was `-11.40%` versus `-5.24%` blind UNDER, delta `-6.16` points,
   CI95 `[-25.36, +13.25]`. The all-books secondary was also null and
   the side result flipped to OVER. The model tied the closing market and
   was worse than the bettime market. Positive drift-adjusted CLV
   (`+0.12` probability points all books; `+0.19` BetOnline) is real
   but too small to establish a vig-clearing edge. **Consequence: no
   promotion, no purchase, and no claim that the Origin B UNDER-selection
   mechanism replicated.**

   The frozen design that produced this result was:
   - Freeze the exact recipe from `experiment_market_state_20260710_213106`:
     104-feature no-pace control + the 7 market feature columns + match
     indicator, same hyperparameter grid, same selection protocol. No
     new features, no reselection against the test fold.
   - Origin C folds per the rolling-origin convention: train 2022-10-07
     through 2024-25 with the final 49 days as validation, test =
     2025-26 (2,624 goalie-games; market-feature coverage ~64% train /
     100% val / 100% test -- better than Origin B's 42% train).
   - Evaluate both passes. The final paired evaluation frames contained
     5,763 bettime quotes and 5,729 closing quotes; the primary BetOnline
     bettime universe contained 1,185 gradeable goalie-nights.
   - PRIMARY metrics: paired Brier vs. the no-pace control (must improve
     with cluster CI excluding zero) and selection-over-blind-UNDER ROI
     delta on the BetOnline bettime cut (the model's UNDER picks at the
     fixed 0.05 EV threshold vs. betting every UNDER quote in the same
     universe). SECONDARY: Brier vs. de-vigged market, OVER/UNDER split,
     bettime-to-close CLV net of the drift baseline, all-books cuts.
   - Honesty constraint: 2025-26 outcomes are "viewed" data (the live
     production UNDER record is known), so a raw UNDER ROI number on
     2025-26 proves nothing by itself -- that is exactly why the
     registered primary is the selection-over-blind delta (which nets
     out the season-wide direction) plus Brier (which is side-neutral).
     The motivating post-hoc observation (Origin B 2024-25 closing:
     market-state UNDER picks +11.18% all-books / +8.66% BetOnline vs.
     blind-UNDER +1.06%/+1.11%, CIs in prereg section 10.1 item 5) was
     sliced after seeing results and authorizes nothing on its own.
   - Dispersion policy, pre-registered per the Experiment 3 correction:
     report BOTH train-fitted and val-fitted dispersion results
     side-by-side (they are cheap to produce together); do not let the
     choice affect the primary metrics (Brier at posted lines is nearly
     dispersion-invariant in the middle; the selection-delta uses the
     same distribution for both arms).
   6b. **DONE -- fixed-offset attempt-to-SOG funnel FAILED.** Binding
   implementation and pass/fail rules were written in preregistration
   section 12 before candidate test predictions. The registered run and
   independent audits are in section 12.8. The candidate missed the
   shots-bias bar on both origins (`+0.78`, `+1.99`) and worsened
   closing Brier versus the no-pace control on both (`+0.00216`,
   `+0.01526`; B reliably worse). It is not a partial mean-model pass.
   The tested construction was the
   registered-but-never-tested construction: deterministic funnel
   projection (attempts -> unblocked -> SOG conversion -> exposure) as a
   FIXED offset/base level, with XGBoost restricted to predicting the
   residual around it -- not raw funnel features as free inputs (that
   variant failed and is dead). Build the exposure stage the way 3.1b
   actually registered it: shots-per-60 scaled by projected exposure
   minutes, reusing Experiment 6's `shots_rate60` machinery and pooled
   TOI bins. Bar: Gate A bullets 1-3 on both origins, pre-registered
   before running.
   6c. **DONE -- fixed-weight early-exit shape improved tails but FAILED
   the combined gate.** The fixed-weight pooled early-exit mixture (weight =
   train-fold early rate, no classifier) + NB2 body as the distribution
   shape, against the corrected Experiment 3 criteria: summed central
   coverage deviation no worse on both origins AND lower-tail gaps
   improved. Experiment 6's metadata already shows this shape had the
   best marginal tail and the best Origin B central coverage of the
   round. In Experiment 9 it improved aggregate lower-tail error and
   negative log score on both origins, and central coverage on B, but
   worsened central coverage on A and could not repair the mean/Brier
   failures. Retain it only as a possible shape layer for a future mean
   architecture; it is not independently promotable.
   6d. **DONE -- Component G executable form is TOO SPARSE.** The repaired
   binding contract was registered in preregistration section 13 before
   any 2025-26 count: the 20-bet minimum is now explicit, the target must
   be outside every tied modal line, and the primary is the accessible
   `betonlineag` cut. The zero-outcome-touch recon found 1,185 paired
   BetOnline goalie-nights but only 31 strictly off-modal quotes; the
   frozen scorer flagged just **one** at the loosest `0.02` threshold and
   zero at every higher threshold. Independent reconstruction reproduced
   every count. Since `1 < 20`, no valid threshold can exist in this
   season even before outcome matching. Component G is closed without
   grading or a confirmatory touch. Revisit only if access to a materially
   different venue or quote product expands the candidate universe; do
   not weaken the corrected contract to manufacture volume.
   6e. **Purchase policy -- gates evaluated, NOT MET.** Step 6a required
   both primaries to pass. P1 passed, but P2 was insufficient and
   directionally unfavorable, so the market-anchored model is not
   promoted and the 2024-25 bettime-pass purchase remains unauthorized.
   This is not evidence that the market-state feature gain was an Origin
   B artifact -- P1 replicated it -- but it is evidence that better
   accuracy relative to the internal control has not translated into an
   executable venue edge. Experiment 9 then failed the 6b/6c Gate-A
   architecture bars on both shots level and Brier. Reconsider purchases
   only after a new preregistered architecture clears its own gate or a
   future untouched bettime season supplies a valid replication target.

   **NEXT WAVE (2026-07-13, decided after the 6a-6e readout; supersedes
   the sequencing of steps 7-12 below).** The user redirected the
   remaining research effort at historical data, with approximately
   50,000 credits expiring 2026-07-31. Acquisition specifics and verified
   vendor facts are in section 5.7. Wave experiments, each requiring its
   own binding preregistration before any candidate run:

   W1. **Cross-market coherence model -- HISTORICAL TOUCH CLOSED; NO OFFICIAL
   VERDICT, SHADOW CANDIDATE (2026-07-14).** Use
   the hockey identity E[saves] ~= E[opponent team SOG] - E[opponent
   goals]: aggregate opponent skater SOG lines into a coverage-adjusted
   team shots projection, estimate opponent goals from moneyline/total
   information, produce an implied saves distribution, and bet only where
   the saves market is materially incoherent with it. Develop on 2023-24,
   single touch on 2024-25. Known modeling hazards to bind in the
   preregistration: SOG props cover only listed skaters (the coverage
   adjustment is load-bearing), prop lines are medians not means,
   empty-net and backup-relief goals break the identity in the tails, and
   book-level SOG coverage breadth is a probe gate. The frozen 2024-25
   calculation met both selection-over-blind numerical bars, with profitable
   UNDERs but losing OVERs, yet the registered recovery failed after the
   touch and cannot receive an official verdict. Global proper scores were
   worse than the market. Preserve the recipe for prospective shadowing; do
   not rerun the historical fold.
   W2. **DFS venue-history census -- COMPLETED, NULL (2026-07-14).**
   Underdog/PrizePicks saves lines
   versus same-timestamp sportsbook consensus and versus outcomes, across
   every season the us_dfs archive reaches. Honest prior: two
   tracker-based analyses agree DFS lines overwhelmingly track consensus
   but disagree on the exact rate (95.2% of 248 goalie-nights in the
   2026-07-07 venue analysis, OFFSEASON_OPTIMIZATION_PLAN.md section
   4.3, versus 90.1% of 294 rows in the 2026-07-13 recon; different
   windows, comparators, units, and dedup rules -- the W2
   preregistration must reconcile the two before adopting either as its
   prior). The registered census reconciled Underdog at 95.16%, found a
   2024-25 PrizePicks deviation sample that graded at coin-flip with a wide
   clustered CI, and had only six 2025-26 confirmation deviations. Close
   historical DFS staleness for this cycle; do not promote it as an edge.
   W3 (zero credit). **Market-microstructure feature block.** Juice skew
   (same sign in both seasons at r of about 0.03, but exploratory:
   unclustered inference from unpersisted scripts, see
   HISTORICAL_DATA_ANALYSIS.md section 8's statistical-standard caveat)
   plus related bettime-observable price-shape features, tested as model
   inputs against the no-pace control under Gate-A-style bars.
   W4 (zero credit). **Rink scorer-effect adjustment.** Official saves
   settle on officially recorded SOG, and published rink-to-rink
   recording bias is persistent; build a prior-season, heavily shrunk
   venue adjustment and test out of sample. Likely failure mode: books
   already price it.
   W5 (zero credit; data already on disk). **Score-state elasticity.**
   `data/raw/play_by_play/` already holds 5,248 event-level play-by-play
   files covering 2022-23 through 2025-26 (a sampled file has 323 events
   under `plays` with `situationCode`, timestamps, and `rosterSpots`;
   the archive is documented in MODEL_TRAINING_GUIDE.md and is consumed
   by `shot_quality_features.py` in the dead pipeline). Parse that
   archive -- do not re-fetch anything -- to estimate team-specific
   score-state shot generation and suppression elasticities, and combine
   with pregame moneyline/total win-probability inputs. (Correction
   2026-07-13, caught by user review: an earlier version of this entry
   wrongly declared no play-by-play exists on disk after checking only
   `data/raw/boxscores/`.)
   W6. **BetOnline convergence-filter policy -- CLOSED, NULL
   (2026-07-14).** The exploratory lead that BetOnline's bettime
   deviations from consensus revert by closing (r=-0.147, nominal
   p=2e-10 on n=1,851 correlated quote rows, unclustered, 2025-26 only)
   was registered and re-derived under clustered, goalie-night-unit
   definitions on both available seasons. 2025-26 (discovery season,
   in-sample): r=-0.05019, CI95 [-0.10550, +0.00543], n=931
   goalie-nights. 2024-25 (already-viewed development season): r=
   -0.05829, CI95 [-0.12429, +0.00488], n=1,380 goalie-nights. Both
   CIs include zero, so neither phase passed; the sign was consistent
   but the lead does not clear this project's statistical bar on
   either season. No shadow-candidate registration follows. Full
   result: preregistration section 17.9; historical analysis section
   9.9.

   Steps 7-9 below (opening anchor and the opening-to-close movement
   model) are deprioritized, not deleted: the zero-credit recon found
   bettime-to-closing steam resolves to noise after deduplication, so an
   opening purchase now needs a specific surviving hypothesis to justify
   it against the W1/W2 purchases. Step 10's condition is superseded by
   section 5.7 (the SOG purchase proceeds on its own probe gate; the
   movement model it was conditioned on is deprioritized). Step 12
   stands. Steps 13-14 (the shadow season) remain the terminal arbiter
   for anything this wave produces.
7. Freeze the opening anchor and full-purchase rules.
8. Purchase the 2024-25 opening saves pass if its probe gate passes.
9. Build and test the opening-to-close movement model (drift-baseline and
   side-split reporting per 4.6/Gate C).
10. Purchase two seasons of player SOG only if its probe gate passes AND the
    movement model shows evidence worth feeding.
11. Build the synthetic cross-market fair saves distribution.
12. Deprioritize the venue-relative Component G filter: the corrected
    BetOnline form produced only one candidate in a season. If retained
    for shadow telemetry, log it as a dormant diagnostic only; do not
    present it as a candidate betting policy.
13. Lock the final candidate before the 2026-27 season.
14. Run the candidate in shadow mode with token stakes and live CLV
    tracking. This season is explicitly a MEASUREMENT season: its job is to
    settle whether the live 2025-26 record and the frozen policy's CLV were
    skill, using the tracking infrastructure already built (roadmap item 6),
    at token cost.

## 11. Final planning posture

The project should not search for a winner by maximizing historical ROI over
more feature combinations. The credible path is to model the mechanism the
current system misses and target the part of the market that may actually be
beatable:

```text
season-aware workload
+ goalie exposure
+ cross-market information
+ book-specific timing
-> stale opening goalie saves quotes
```

This is a high-upside research direction, not a promise that an edge exists.
Its value is that every phase can produce a clear answer while protecting the
remaining data budget and the integrity of the next live test.

Everything above must also fit the bettor described in section 1a: one
automated pipeline, one early-evening decision, three accessible venues.
A strategy that only works for someone watching lines all day at six
sportsbooks is, for this project, indistinguishable from no strategy.
