# Model Issues and Proposed Fixes

## Issue 1: Betting Line Has No Meaningful Impact on Predictions

### Problem

The XGBoost classifier produces the same `prob_over` for a given goalie regardless of the betting line. For example, on 2026-01-31:

| Goalie | Book | Line | P(Over) |
|--------|------|------|---------|
| Bobrovsky | BetOnline | 21.5 | 51.2% |
| Bobrovsky | Underdog | 20.5 | 51.2% |
| Vladar | Underdog | 22.5 | 54.6% |
| Vladar | PrizePicks | 21.5 | 54.6% |
| Vladar | BetOnline | 21.5 | 54.6% |

A 1-unit line difference should produce a meaningfully different probability. Going over 20.5 is strictly easier than going over 21.5. The model cannot distinguish them.

### Root Cause

1. **Correlated features**: Bookmakers set lines based on the same recent performance the model uses (rolling averages). So `betting_line` is highly correlated with `saves_rolling_5`, `saves_rolling_10`, etc. The model learns little from `betting_line` that it doesn't already know.

2. **Coarse tree structure**: With `max_depth=4` and `min_child_weight=15`, XGBoost creates broad splits. Tree splits on `betting_line` land at thresholds (e.g., ">23.5") that don't separate nearby values like 20.5 vs 21.5 for the same goalie. All other features are identical for the same goalie on the same day, so the prediction is identical.

3. **Single training sample per goalie per game**: The training data has one row per goalie per game with one betting line value. The model never sees the same goalie-game with two different lines, so it cannot learn how line changes affect the probability in isolation.

### Impact

- EV comparisons across books with different lines are unreliable. The model treats a 20.5 line the same as a 21.5 line, so the EV difference between books is driven entirely by odds, ignoring line advantage.
- Multi-book line shopping (the main reason to track multiple sportsbooks) provides no value through the model.

---

## Issue 2: Systematic Directional Bias (OVER for Low Lines, UNDER for High Lines)

### Problem

The model systematically recommends:
- **OVER** for goalies with low betting lines (20.5, 21.5, 22.5)
- **UNDER** for goalies with high betting lines (25.5, 26.5, 27.5)

This pattern appears consistently across dates regardless of goalie, opponent, or game context.

### Root Cause

Since the model effectively ignores the betting line value, it's really predicting "will this goalie's saves regress toward the league mean?" based on rolling stats:

- **Low-line goalies** have low recent averages. The model detects that their rolling stats are below league average and predicts slight upward reversion -> P(over) slightly > 0.5 -> OVER.
- **High-line goalies** have high recent averages. The model detects above-average rolling stats and predicts slight downward reversion -> P(over) slightly < 0.5 -> UNDER.

This is regression-to-the-mean detection, not line-specific analysis. The model is answering "will this goalie do better or worse than recently?" rather than "will this goalie go over this specific number?"

### Impact

- Recommendations cluster around the same direction for all lines on the same goalie, even when different lines should produce different recommendations.
- The model has a structural blind spot: it cannot identify when a book offers a line that's particularly favorable vs. unfavorable for the same goalie.

---

## Issue 3: Predicted Saves Is Not a Real Prediction

### Problem

The `predicted_saves` column is calculated as:

```python
estimated_offset = (prob_over - 0.5) * 5
predicted_saves = round(betting_line + estimated_offset, 1)
```

This is a cosmetic reverse-engineering of the probability back to a saves number. It's not an independent estimate. The same goalie on the same day shows different predicted saves depending on which book's line is used:

| Goalie | Book | Line | Pred Saves | P(Over) |
|--------|------|------|------------|---------|
| Bobrovsky | BetOnline | 21.5 | 21.6 | 51.2% |
| Bobrovsky | Underdog | 20.5 | 20.6 | 51.2% |

A real prediction would show the same expected saves regardless of which book's line is referenced.

### Impact

- The "predicted saves" column in the tracker is misleading. Users may think the model predicts different save totals depending on the book, which makes no sense.
- No actual saves estimate exists that could be compared across lines.

---

## Proposed Fixes

### Approach A: Two-Stage Model (Regression + Probability Derivation)

**Concept**: Predict expected saves with a regression model, then derive P(over) mathematically for any line.

**How it works**:

1. **Stage 1 - Regression model**: Train an XGBoost regressor to predict actual saves. Uses the same 89 features (everything except `betting_line`). Outputs a single number: expected saves for this goalie in this game.

2. **Stage 2 - Distribution fitting**: Estimate the prediction uncertainty (standard deviation) from the model's residuals on validation data, possibly varying by goalie tier or confidence level.

3. **Stage 3 - Line-specific probability**: For any given betting line, calculate:
   ```
   P(over) = P(saves > line) = 1 - CDF(line, mean=predicted_saves, std=estimated_std)
   ```
   Using a normal distribution (or a better-fitted distribution like negative binomial).

**Advantages**:
- One prediction per goalie per game, independent of book/line.
- P(over) naturally decreases as the line increases (by definition).
- Line shopping becomes meaningful: P(over 20.5) > P(over 21.5) automatically.
- `predicted_saves` becomes a real, meaningful number.

**Disadvantages**:
- Requires choosing a distribution family (normal, Poisson, negative binomial).
- The distribution assumption may not perfectly fit all goalies.
- Two sources of error: the mean prediction and the variance estimate.

**Implementation sketch**:
```python
class TwoStagePredictor:
    def __init__(self):
        self.regression_model = ...  # XGBoost regressor
        self.residual_std = ...      # Estimated from validation set

    def predict(self, features, betting_line, line_over_odds, line_under_odds):
        predicted_saves = self.regression_model.predict(features)
        prob_over = 1 - norm.cdf(betting_line, loc=predicted_saves, scale=self.residual_std)
        # ... EV calculation using prob_over and odds
```

---

### Approach B: Line-Relative Feature Engineering

**Concept**: Keep the classifier architecture, but engineer features that explicitly capture the relationship between the line and the goalie's stats.

**New features**:
```python
# How far the line is from the goalie's recent average
line_vs_rolling_3 = betting_line - saves_rolling_3
line_vs_rolling_5 = betting_line - saves_rolling_5
line_vs_rolling_10 = betting_line - saves_rolling_10

# Normalized version (in standard deviations)
line_z_score_3 = (betting_line - saves_rolling_3) / saves_rolling_std_3
line_z_score_5 = (betting_line - saves_rolling_5) / saves_rolling_std_5

# Historical over rate at similar lines
# (requires bucketing lines and computing hit rates)
```

**Advantages**:
- Minimal architecture change; stays within the existing classifier framework.
- The model can learn "line is 2 saves above recent average -> less likely to go over" as a direct feature.
- Compatible with existing training pipeline.

**Disadvantages**:
- Still a classifier that predicts one P(over) per row. If the new features work, different lines should produce different probabilities, but this depends on the model learning the relationship.
- Doesn't solve the fundamental issue of one training sample per goalie per game per line.

---

### Approach C: Data Augmentation with Multiple Lines Per Game

**Concept**: For each goalie-game in the training data, create multiple training rows at different betting lines. Since we know the actual saves outcome, we can label each line correctly.

**How it works**:

For a game where a goalie made 24 saves and the real line was 23.5:
```
Original row:  line=23.5, over_hit=1 (24 > 23.5)

Augmented rows:
  line=21.5, over_hit=1 (24 > 21.5)
  line=22.5, over_hit=1 (24 > 22.5)
  line=23.5, over_hit=1 (24 > 23.5)  <- original
  line=24.5, over_hit=0 (24 < 24.5)
  line=25.5, over_hit=0 (24 < 25.5)
```

**Advantages**:
- The model sees the same goalie-game features with different lines and different outcomes. This forces it to learn that the line matters.
- Dramatic increase in training data volume.
- Compatible with the existing classifier architecture.

**Disadvantages**:
- Augmented rows are not independent samples (same underlying game). Must account for this in validation to avoid data leakage (all augmented rows from the same game must be in the same split).
- Need to decide what range of lines to generate and whether to weight them (real market lines vs. synthetic ones).
- Odds are only available for the actual market line, not the augmented ones. EV-based evaluation during training would need adjustment.

---

### Approach D: Hybrid (Recommended)

Combine approaches A and C for the strongest result:

1. **Train a regression model** (Approach A) to predict expected saves. This gives a meaningful `predicted_saves` number and a residual distribution.

2. **Train a line-aware classifier** (Approach C) with augmented data and line-relative features (Approach B). This classifier can serve as the EV-based betting signal.

3. **Use the regression model's distribution** as a sanity check on the classifier's probabilities. If the classifier says P(over 20.5) = 51% but the regression model predicts 25 saves with std=4, the regression model's P(over 20.5) = 87%, revealing a large discrepancy.

4. **Ensemble the two** for final predictions - e.g., weighted average of classifier P(over) and regression-derived P(over).

---

### Approach E: Quick Fix (Minimal Change)

If a full retrain isn't feasible right now, a post-hoc adjustment can partially address the issue:

**Concept**: After the classifier produces `prob_over`, adjust it based on where the line sits relative to the goalie's rolling average.

```python
# Get base probability from classifier (line-insensitive)
base_prob = model.predict(features)

# Adjust based on line position relative to goalie's recent average
line_diff = betting_line - saves_rolling_5
# Each save of line difference shifts probability by ~3-5%
adjustment = -line_diff * 0.04  # Negative because higher line = lower P(over)
adjusted_prob = np.clip(base_prob + adjustment, 0.05, 0.95)
```

**Advantages**:
- No retraining needed. Can be deployed immediately.
- Different lines for the same goalie will produce different probabilities.

**Disadvantages**:
- The adjustment factor (0.04 per save) is a heuristic, not learned.
- Doesn't address the underlying model limitation.
- May degrade calibration if the factor is wrong.

---

## Summary

| Approach | Effort | Line Sensitivity | Real Pred Saves | Retraining Required |
|----------|--------|-----------------|-----------------|---------------------|
| A: Two-Stage Regression | Medium | Yes (by design) | Yes | Yes |
| B: Line-Relative Features | Low | Partial | No | Yes |
| C: Data Augmentation | Medium | Yes (learned) | No | Yes |
| D: Hybrid (A+B+C) | High | Yes | Yes | Yes |
| E: Quick Post-Hoc Fix | Low | Partial | No | No |

**Recommendation**: Start with **Approach A** (two-stage regression). It directly solves all three issues, produces a real `predicted_saves` number, and makes line-specific probability a mathematical consequence rather than something the model must learn. The regression model likely already exists in the codebase (`src/models/trainer.py`), so much of the infrastructure is already there.
