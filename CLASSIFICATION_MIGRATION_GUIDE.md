# Classification Model Migration Guide

## Executive Summary

This document provides a comprehensive roadmap for transitioning from a **regression-based model** (predicting save count) to a **classification-based model** (predicting OVER/UNDER outcomes) for NHL goalie save betting.

**Current State:** Predict saves (e.g., 27.3) → Compare to line → Recommend OVER/UNDER
**Target State:** Predict OVER/UNDER probability directly → Recommend based on probability

---

## Table of Contents

1. [Why This Change Matters](#why-this-change-matters)
2. [Data Availability & Coverage](#data-availability--coverage)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Detailed Code Changes](#detailed-code-changes)
6. [Testing & Validation](#testing--validation)
7. [Deployment Strategy](#deployment-strategy)
8. [Rollback Plan](#rollback-plan)

---

## Why This Change Matters

### Key Advantages of Classification:

1. **Directly Optimizes for Betting Outcomes**
   - Regression optimizes: `min (predicted_saves - actual_saves)²`
   - Classification optimizes: `max P(correct OVER/UNDER prediction)`
   - **Result:** Better betting accuracy even with similar RMSE

2. **Learns Line-Specific Patterns**
   - Model learns: "When line is 28.5 and team defense is weak → OVER"
   - Can identify when Vegas misprices lines
   - Exploits market inefficiencies

3. **Better Probability Calibration**
   - Get explicit confidence: "72% chance of OVER"
   - Enables sophisticated bankroll management (Kelly Criterion)
   - Can filter low-confidence bets

4. **Errors Only Matter at the Margin**
   - Predicting 26 vs 28 saves: SAME outcome if line is 25.5
   - Classification doesn't penalize irrelevant errors
   - Focus is on getting the direction right

### Expected Improvements:

- **Betting Accuracy:** 5-10% improvement expected
- **ROI:** Higher expected value per bet
- **Bankroll Management:** Better risk-adjusted returns

---

## Data Availability & Coverage

### Available Data:
- **Seasons:** 2023-24 and 2024-25 (via API)
- **Missing:** 2022-23 season (no betting lines)

### Coverage Strategy:

**Option 1: Use Only 2 Seasons (Recommended)**
- Train on 2023-24 + 2024-25 with lines (~5,000 games)
- Keep 2022-23 as regression-only or discard
- **Pros:** All data has ground truth (actual lines)
- **Cons:** Less training data

**Option 2: Mixed Approach**
- Use all 3 seasons for regression
- Use 2 seasons for classification
- Ensemble predictions
- **Pros:** More data utilization
- **Cons:** More complex pipeline

**Recommended:** Option 1 - Train classification model on 2023-24 + 2024-25 only.

### Data Split Strategy:

```
2023-24 Season: Training (70%) + Validation (15%)
2024-25 Season: Test Set (15%)
```

**Rationale:**
- Time-based split prevents future data leakage
- 2024-25 as test set simulates real production scenario
- Can retrain periodically as new data arrives

---

## Architecture Overview

### Current Pipeline (Regression):

```
Raw Data → Feature Engineering → XGBoost Regressor
    ↓
Predicted Saves → Compare to Line → OVER/UNDER Recommendation
```

### New Pipeline (Classification):

```
Raw Data + Betting Lines → Feature Engineering + Line Feature
    ↓
XGBoost Classifier → P(OVER) → Threshold (e.g., >0.55) → Recommendation
```

### Hybrid Pipeline (Recommended for Transition):

```
Raw Data + Betting Lines → Feature Engineering
    ↓                              ↓
Regression Model              Classification Model
    ↓                              ↓
Predicted Saves          P(OVER) with confidence
    ↓                              ↓
    └──────────┬──────────┘
               ↓
      Ensemble / A-B Test
               ↓
      Final Recommendation
```

---

## Implementation Phases

### Phase 0: Prerequisites (Before Starting)
- [ ] API access confirmed for 2023-24 and 2024-25
- [ ] Sample data fetched and validated
- [ ] Data format understood (JSON/CSV structure)
- [ ] Game/goalie matching strategy tested

### Phase 1: Data Integration (Week 1)
**Goal:** Fetch and merge betting lines with existing dataset

**Tasks:**
1. Create betting lines data collector
2. Fetch historical lines for 2023-24 and 2024-25
3. Build matching logic (game_id + goalie_name)
4. Merge with `training_data.parquet`
5. Create binary target variable (`over_hit`)
6. Validate data quality

**Deliverables:**
- `data/raw/betting_lines/` directory with raw API responses
- `data/processed/training_data_with_lines.parquet` (enhanced dataset)
- Data quality report

### Phase 2: Feature Engineering (Week 2)
**Goal:** Prepare features for classification model

**Tasks:**
1. Add betting line as a feature
2. Engineer line-related features (line margin, line vs historical avg, etc.)
3. Update feature engineering pipeline
4. Regenerate processed dataset
5. Update data splits (exclude 2022-23 from classification)

**Deliverables:**
- Updated `create_features.py`
- New feature set with line-based features
- Updated documentation

### Phase 3: Model Development (Week 2-3)
**Goal:** Build and tune classification model

**Tasks:**
1. Create `ClassifierTrainer` class (parallel to regression)
2. Train XGBoost classifier
3. Hyperparameter tuning (optimize for log loss)
4. Calibration (Platt scaling if needed)
5. Feature importance analysis

**Deliverables:**
- `src/models/classifier_trainer.py`
- Trained classification model
- Hyperparameter tuning results
- Model evaluation report

### Phase 4: Evaluation & Comparison (Week 3)
**Goal:** Compare classification vs regression on betting performance

**Tasks:**
1. Backtest both models on test set (2024-25)
2. Calculate betting metrics (accuracy, ROI, Sharpe ratio)
3. Statistical significance testing
4. Generate comparison report

**Deliverables:**
- A/B test results
- Betting simulation results
- Decision on which model to use

### Phase 5: Production Integration (Week 4)
**Goal:** Deploy chosen model to production

**Tasks:**
1. Update prediction pipeline
2. Update API/interface for realtime predictions
3. Add confidence thresholds
4. Update documentation
5. Monitor initial performance

**Deliverables:**
- Updated `scripts/predict_games.py`
- Production-ready classification model
- Deployment documentation

---

## Detailed Code Changes

### 1. Data Collection: Betting Lines Fetcher

**New File:** `src/data/betting_lines_collector.py`

```python
"""
Fetch historical betting lines from API and store locally
"""
import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class BettingLinesCollector:
    """Collect historical goalie save O/U lines from betting API"""

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize collector

        Args:
            api_key: API authentication key
            base_url: Base URL for betting API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def fetch_lines_by_date_range(
        self,
        start_date: str,
        end_date: str,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Fetch goalie save lines for date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save raw responses

        Returns:
            DataFrame with betting lines
        """
        logger.info(f"Fetching lines from {start_date} to {end_date}")

        all_lines = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')

            # Fetch lines for this date
            lines = self._fetch_lines_for_date(date_str)

            if lines:
                # Save raw response
                self._save_raw_response(lines, date_str, output_dir)

                # Parse and accumulate
                parsed = self._parse_response(lines, date_str)
                all_lines.extend(parsed)

            # Rate limiting
            time.sleep(0.5)

            current_date += timedelta(days=1)

            if len(all_lines) % 100 == 0:
                logger.info(f"Progress: {len(all_lines)} lines collected")

        df = pd.DataFrame(all_lines)
        logger.info(f"Total lines collected: {len(df)}")

        return df

    def _fetch_lines_for_date(self, date: str) -> Optional[Dict]:
        """Fetch lines for a specific date from API"""
        # TODO: Implement based on specific API
        # This is a template - actual implementation depends on API structure

        endpoint = f"{self.base_url}/nhl/props"
        params = {
            'api_key': self.api_key,
            'date': date,
            'market': 'goalie_saves',
            'type': 'over_under'
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {date}: {e}")
            return None

    def _parse_response(self, response: Dict, date: str) -> List[Dict]:
        """Parse API response into standardized format"""
        # TODO: Implement based on API response structure

        lines = []

        # Example parsing (adjust based on actual API):
        for game in response.get('games', []):
            game_id = game.get('nhl_game_id')  # If API provides this
            home_team = game.get('home_team')
            away_team = game.get('away_team')

            for prop in game.get('player_props', []):
                if prop.get('market') == 'goalie_saves':
                    lines.append({
                        'game_id': game_id,
                        'game_date': date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'goalie_name': prop['player_name'],
                        'team': prop.get('team'),
                        'save_line': float(prop['line']),
                        'over_odds': prop.get('over_odds'),
                        'under_odds': prop.get('under_odds'),
                        'sportsbook': prop.get('sportsbook', 'Unknown'),
                        'timestamp': prop.get('timestamp')
                    })

        return lines

    def _save_raw_response(self, response: Dict, date: str, output_dir: Path):
        """Save raw API response for audit trail"""
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = output_dir / f"lines_{date}.json"

        import json
        with open(filename, 'w') as f:
            json.dump(response, f, indent=2)
```

**New Script:** `scripts/collect_betting_lines.py`

```python
"""
Collect historical betting lines for all games
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.betting_lines_collector import BettingLinesCollector
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load API credentials from config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['betting_api']['api_key']
    base_url = config['betting_api']['base_url']

    collector = BettingLinesCollector(api_key, base_url)

    # Collect for 2023-24 season
    logger.info("Collecting 2023-24 season lines...")
    lines_2023 = collector.fetch_lines_by_date_range(
        start_date='2023-10-10',
        end_date='2024-04-18',
        output_dir=Path('data/raw/betting_lines/2023-24')
    )
    lines_2023.to_parquet('data/raw/betting_lines/lines_2023-24.parquet')

    # Collect for 2024-25 season
    logger.info("Collecting 2024-25 season lines...")
    lines_2024 = collector.fetch_lines_by_date_range(
        start_date='2024-10-08',
        end_date='2025-04-17',
        output_dir=Path('data/raw/betting_lines/2024-25')
    )
    lines_2024.to_parquet('data/raw/betting_lines/lines_2024-25.parquet')

    logger.info("✅ Betting lines collection complete!")

if __name__ == "__main__":
    main()
```

### 2. Data Integration: Merge Lines with Games

**New Script:** `scripts/merge_betting_lines.py`

```python
"""
Merge betting lines with existing training data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_goalie_name_mapping() -> dict:
    """
    Create mapping from goalie_id to name
    Read from raw boxscores to build this mapping
    """
    # TODO: Build from boxscore data
    # This is needed if API uses names but we have IDs
    return {}

def fuzzy_match_goalie(
    api_name: str,
    candidate_names: list,
    threshold: int = 85
) -> str:
    """
    Fuzzy match goalie name from API to our dataset

    Args:
        api_name: Name from betting API
        candidate_names: List of known names for this game
        threshold: Minimum similarity score (0-100)

    Returns:
        Best matching name or None
    """
    result = process.extractOne(
        api_name,
        candidate_names,
        scorer=fuzz.ratio
    )

    if result and result[1] >= threshold:
        return result[0]

    logger.warning(f"No match found for '{api_name}' in {candidate_names}")
    return None

def merge_lines_with_games(
    games_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    name_mapping: dict
) -> pd.DataFrame:
    """
    Merge betting lines with game data

    Args:
        games_df: Game-level data with goalie performances
        lines_df: Betting lines from API
        name_mapping: goalie_id → name mapping

    Returns:
        Merged DataFrame with betting_line column
    """
    logger.info(f"Merging {len(games_df)} games with {len(lines_df)} betting lines")

    # Strategy 1: Match on game_id (if API provides it)
    if 'game_id' in lines_df.columns and lines_df['game_id'].notna().any():
        logger.info("Using game_id matching strategy")

        # Group lines by game
        lines_by_game = lines_df.groupby('game_id')

        merged_rows = []
        for idx, game_row in games_df.iterrows():
            game_id = game_row['game_id']

            if game_id in lines_by_game.groups:
                game_lines = lines_by_game.get_group(game_id)

                # Match goalie by name
                goalie_name = name_mapping.get(game_row['goalie_id'])

                # Find line for this goalie
                matched_line = None
                for _, line_row in game_lines.iterrows():
                    if fuzzy_match_goalie(line_row['goalie_name'], [goalie_name]):
                        matched_line = line_row
                        break

                if matched_line is not None:
                    game_row['betting_line'] = matched_line['save_line']
                    game_row['line_source'] = matched_line['sportsbook']
                    game_row['line_timestamp'] = matched_line['timestamp']
                else:
                    game_row['betting_line'] = np.nan
            else:
                game_row['betting_line'] = np.nan

            merged_rows.append(game_row)

        result_df = pd.DataFrame(merged_rows)

    else:
        # Strategy 2: Match on date + teams
        logger.info("Using date + teams matching strategy")

        # Create composite key
        lines_df['match_key'] = (
            lines_df['game_date'].astype(str) + '_' +
            lines_df['home_team'] + '_' +
            lines_df['away_team']
        )

        games_df['match_key'] = (
            games_df['game_date'].astype(str) + '_' +
            games_df.apply(
                lambda r: r['team_abbrev'] if r['is_home'] else r['opponent_team'],
                axis=1
            ) + '_' +
            games_df.apply(
                lambda r: r['opponent_team'] if r['is_home'] else r['team_abbrev'],
                axis=1
            )
        )

        result_df = games_df.merge(
            lines_df[['match_key', 'goalie_name', 'save_line', 'sportsbook', 'timestamp']],
            on='match_key',
            how='left'
        )

        result_df.rename(columns={
            'save_line': 'betting_line',
            'sportsbook': 'line_source',
            'timestamp': 'line_timestamp'
        }, inplace=True)

    # Create derived fields
    result_df['over_hit'] = (result_df['saves'] > result_df['betting_line']).astype(int)
    result_df['line_margin'] = result_df['saves'] - result_df['betting_line']

    # Report coverage
    coverage = result_df['betting_line'].notna().sum() / len(result_df) * 100
    logger.info(f"Betting line coverage: {coverage:.1f}%")

    return result_df

def main():
    # Load existing training data
    games_df = pd.read_parquet('data/processed/training_data.parquet')
    logger.info(f"Loaded {len(games_df)} games")

    # Load betting lines
    lines_2023 = pd.read_parquet('data/raw/betting_lines/lines_2023-24.parquet')
    lines_2024 = pd.read_parquet('data/raw/betting_lines/lines_2024-25.parquet')
    lines_df = pd.concat([lines_2023, lines_2024], ignore_index=True)
    logger.info(f"Loaded {len(lines_df)} betting lines")

    # Filter games to only 2023-24 and 2024-25
    games_df = games_df[games_df['season'].isin(['20232024', '20242025'])]
    logger.info(f"Filtered to {len(games_df)} games from 2023-24 and 2024-25")

    # Load name mapping
    name_mapping = load_goalie_name_mapping()

    # Merge
    merged_df = merge_lines_with_games(games_df, lines_df, name_mapping)

    # Save
    output_path = Path('data/processed/training_data_with_lines.parquet')
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved merged data to {output_path}")

    # Quality report
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    print(f"Total games: {len(merged_df)}")
    print(f"Games with betting lines: {merged_df['betting_line'].notna().sum()}")
    print(f"Coverage: {merged_df['betting_line'].notna().sum() / len(merged_df) * 100:.1f}%")
    print(f"\nLine distribution:")
    print(merged_df['betting_line'].describe())
    print(f"\nOVER hit rate: {merged_df['over_hit'].mean() * 100:.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()
```

### 3. Feature Engineering: Add Line-Based Features

**Update:** `src/features/feature_engineering.py`

Add new function:

```python
def add_betting_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features derived from betting line

    Args:
        df: DataFrame with betting_line column

    Returns:
        DataFrame with additional line-based features
    """
    logger.info("Adding betting line-based features...")

    # Line relative to recent performance
    df['line_vs_avg_saves_5'] = df['betting_line'] - df['saves_rolling_5']
    df['line_vs_avg_saves_10'] = df['betting_line'] - df['saves_rolling_10']

    # Line difficulty (higher line = harder to hit OVER)
    df['line_percentile'] = df.groupby('season')['betting_line'].rank(pct=True)

    # Line vs opponent average shots
    df['line_vs_opp_shots_5'] = df['betting_line'] - df['opp_offense_team_shots_rolling_5']

    # Is line suspiciously high/low?
    df['line_zscore'] = (
        (df['betting_line'] - df['saves_rolling_10']) /
        df['saves_rolling_10'].std()
    )

    # Categorical: Line bucket
    df['line_bucket'] = pd.cut(
        df['betting_line'],
        bins=[0, 20, 23, 26, 29, 100],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    logger.info(f"Added {5} betting line features")

    return df
```

Update `calculate_all_features` to call this:

```python
# After Step 4.10 in existing pipeline
if 'betting_line' in df.columns:
    df = add_betting_line_features(df)
```

### 4. Model Training: Classification Trainer

**New File:** `src/models/classifier_trainer.py`

```python
"""
Binary classification trainer for OVER/UNDER prediction
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    log_loss, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class GoalieClassifierTrainer:
    """Train binary classifier for goalie save OVER/UNDER prediction"""

    def __init__(self, config: dict):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.threshold = 0.5  # Probability threshold for OVER prediction

    def prepare_classification_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Prepare data for classification
        Filter to only games with betting lines

        Args:
            df: Full dataset
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing classification data...")

        # Filter to games with betting lines
        df_with_lines = df[df['betting_line'].notna()].copy()
        logger.info(f"Filtered to {len(df_with_lines)} games with betting lines")

        # Sort by date for time-based split
        df_with_lines = df_with_lines.sort_values('game_date').reset_index(drop=True)

        # Define feature columns
        exclude_cols = [
            'saves', 'over_hit', 'line_margin',  # Targets
            'game_id', 'goalie_id', 'goalie_name', 'game_date', 'season',
            'team_abbrev', 'opponent_team', 'decision', 'toi',
            'line_source', 'line_timestamp'  # Metadata
        ]

        feature_cols = [col for col in df_with_lines.columns if col not in exclude_cols]
        self.feature_names = feature_cols

        # Create features and target
        X = df_with_lines[feature_cols]
        y = df_with_lines['over_hit']  # Binary: 1 if OVER, 0 if UNDER

        # Time-based split
        n = len(df_with_lines)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]

        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Val: {len(X_val)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_cols)}")

        # Class distribution
        logger.info(f"Train OVER rate: {y_train.mean():.1%}")
        logger.info(f"Val OVER rate: {y_val.mean():.1%}")
        logger.info(f"Test OVER rate: {y_test.mean():.1%}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **xgb_params
    ) -> XGBClassifier:
        """
        Train XGBoost binary classifier

        Args:
            X_train: Training features
            y_train: Training targets (0/1)
            X_val: Validation features
            y_val: Validation targets
            **xgb_params: XGBoost parameters

        Returns:
            Trained classifier
        """
        logger.info("Training XGBoost classifier...")

        # Default parameters optimized for classification
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 600,
            'max_depth': 4,
            'learning_rate': 0.012,
            'subsample': 0.9,
            'colsample_bytree': 1.0,
            'min_child_weight': 7,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 2.0,
            'random_state': self.config.get('model', {}).get('random_state', 42),
            'n_jobs': -1,
            'verbosity': 1
        }

        # Override with provided parameters
        default_params.update(xgb_params)

        logger.info(f"XGBoost parameters: {default_params}")

        # Initialize classifier
        self.model = XGBClassifier(**default_params)

        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        logger.info("✅ Classifier training complete")

        return self.model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance

        Args:
            X: Features
            y: True labels
            set_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        # Predict probabilities and classes
        y_proba = self.model.predict_proba(X)[:, 1]  # Probability of OVER
        y_pred = (y_proba >= self.threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'log_loss': log_loss(y, y_proba)
        }

        logger.info(f"\n{set_name} Set Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Log Loss:  {metrics['log_loss']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

        return metrics

    def save_model(
        self,
        model_path: Path,
        metadata: Optional[Dict] = None
    ):
        """Save trained classifier"""
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")

        # Save feature names
        feature_path = model_path.parent / f"{model_path.stem}_features.json"
        import json
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Features saved to {feature_path}")

        # Save metadata
        if metadata:
            meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {meta_path}")
```

**New Script:** `scripts/train_classifier.py`

```python
"""
Train binary classification model for OVER/UNDER prediction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.classifier_trainer import GoalieClassifierTrainer
import pandas as pd
import yaml
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("NHL Goalie OVER/UNDER Classifier Training")
    logger.info("="*70)

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data with betting lines
    data_path = 'data/processed/training_data_with_lines.parquet'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Initialize trainer
    trainer = GoalieClassifierTrainer(config)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_classification_data(df)

    # Train
    logger.info("\nStarting classifier training...")
    model = trainer.train_classifier(X_train, y_train, X_val, y_val)

    # Evaluate
    logger.info("\n" + "="*70)
    logger.info("Model Evaluation")
    logger.info("="*70)

    train_metrics = trainer.evaluate(X_train, y_train, "Train")
    val_metrics = trainer.evaluate(X_val, y_val, "Validation")
    test_metrics = trainer.evaluate(X_test, y_test, "Test")

    # Save model
    logger.info("\n" + "="*70)
    logger.info("Saving Model")
    logger.info("="*70)

    model_path = Path('models/trained/xgboost_goalie_classifier.pkl')
    metadata = {
        'train_date': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': test_metrics['accuracy'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_log_loss': test_metrics['log_loss'],
        'features': trainer.feature_names,
        'model_type': 'xgboost_classifier'
    }

    trainer.save_model(model_path, metadata)

    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
    logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
```

### 5. Prediction Pipeline: Classification Predictions

**Update:** `src/models/predictor.py`

Add new class:

```python
class GoalieClassifierPredictor:
    """Make OVER/UNDER predictions using classification model"""

    def __init__(self, model_path: Path):
        """Load trained classifier"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load feature names
        feature_path = model_path.parent / f"{model_path.stem}_features.json"
        import json
        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)

    def predict_with_line(
        self,
        features: pd.DataFrame,
        betting_line: float,
        confidence_threshold: float = 0.55
    ) -> Dict:
        """
        Predict OVER/UNDER with confidence

        Args:
            features: Game features (must include betting_line)
            betting_line: The over/under line
            confidence_threshold: Minimum probability to recommend bet

        Returns:
            Prediction dict with recommendation and confidence
        """
        # Ensure betting_line is in features
        if 'betting_line' not in features.columns:
            features['betting_line'] = betting_line

        # Predict probability of OVER
        proba_over = self.model.predict_proba(features[self.feature_names])[0, 1]
        proba_under = 1 - proba_over

        # Determine recommendation
        if proba_over >= confidence_threshold:
            recommendation = 'OVER'
            confidence = proba_over
        elif proba_under >= confidence_threshold:
            recommendation = 'UNDER'
            confidence = proba_under
        else:
            recommendation = 'NO BET'
            confidence = max(proba_over, proba_under)

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'proba_over': proba_over,
            'proba_under': proba_under,
            'betting_line': betting_line,
            'confidence_threshold': confidence_threshold
        }
```

**Update:** `scripts/predict_games.py`

Add option to use classifier:

```python
parser.add_argument(
    '--use-classifier',
    action='store_true',
    help='Use classification model instead of regression'
)

# In prediction logic:
if args.use_classifier:
    classifier = GoalieClassifierPredictor('models/trained/xgboost_goalie_classifier.pkl')

    for goalie in [home_goalie, away_goalie]:
        result = classifier.predict_with_line(
            features=goalie_features,
            betting_line=betting_line,
            confidence_threshold=0.55
        )

        print(f"\n{goalie['name']}:")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  P(OVER): {result['proba_over']:.1%}")
        print(f"  P(UNDER): {result['proba_under']:.1%}")
```

### 6. Evaluation: A/B Testing Script

**New Script:** `scripts/compare_models.py`

```python
"""
Compare regression vs classification on betting performance
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """Load both models"""
    with open('models/trained/xgboost_goalie_model.pkl', 'rb') as f:
        regression_model = pickle.load(f)

    with open('models/trained/xgboost_goalie_classifier.pkl', 'rb') as f:
        classifier_model = pickle.load(f)

    return regression_model, classifier_model

def simulate_betting_performance(
    predictions: pd.Series,
    actuals: pd.Series,
    betting_lines: pd.Series,
    stake: float = 100.0,
    odds: float = -110  # Standard betting odds
) -> Dict:
    """
    Simulate betting performance

    Returns:
        Dict with ROI, win rate, total profit, etc.
    """
    # Convert odds to decimal
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)

    correct = (predictions == (actuals > betting_lines)).sum()
    total = len(predictions)
    win_rate = correct / total

    # Calculate profit
    profit_per_win = stake * (decimal_odds - 1)
    profit_per_loss = -stake

    wins = correct
    losses = total - correct
    total_profit = (wins * profit_per_win) + (losses * profit_per_loss)
    roi = (total_profit / (stake * total)) * 100

    return {
        'win_rate': win_rate,
        'total_bets': total,
        'wins': wins,
        'losses': losses,
        'total_profit': total_profit,
        'roi': roi,
        'avg_profit_per_bet': total_profit / total
    }

def main():
    # Load test data
    df = pd.read_parquet('data/processed/training_data_with_lines.parquet')
    df = df[df['betting_line'].notna()].sort_values('game_date')

    # Use 2024-25 season as test set
    test_df = df[df['season'] == '20242025']
    logger.info(f"Test set: {len(test_df)} games")

    # Load models
    reg_model, clf_model = load_models()

    # Prepare features
    # (Similar to training - exclude targets, metadata)

    # Regression predictions
    reg_preds = reg_model.predict(X_test)
    reg_recommendations = reg_preds > test_df['betting_line']

    # Classification predictions
    clf_probs = clf_model.predict_proba(X_test)[:, 1]
    clf_recommendations = clf_probs > 0.5

    # Evaluate both
    print("\n" + "="*70)
    print("BETTING PERFORMANCE COMPARISON")
    print("="*70)

    print("\nREGRESSION MODEL:")
    reg_perf = simulate_betting_performance(
        reg_recommendations,
        test_df['saves'],
        test_df['betting_line']
    )
    for k, v in reg_perf.items():
        print(f"  {k}: {v}")

    print("\nCLASSIFICATION MODEL:")
    clf_perf = simulate_betting_performance(
        clf_recommendations,
        test_df['saves'],
        test_df['betting_line']
    )
    for k, v in clf_perf.items():
        print(f"  {k}: {v}")

    # Statistical significance test
    from scipy.stats import chi2_contingency

    contingency = np.array([
        [reg_perf['wins'], reg_perf['losses']],
        [clf_perf['wins'], clf_perf['losses']]
    ])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✅ Difference is statistically significant (p < 0.05)")
    else:
        print("❌ Difference is NOT statistically significant")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if clf_perf['roi'] > reg_perf['roi']:
        print("✅ Use CLASSIFICATION MODEL")
        print(f"   ROI improvement: {clf_perf['roi'] - reg_perf['roi']:.2f}%")
    else:
        print("✅ Keep REGRESSION MODEL")
        print(f"   Regression ROI advantage: {reg_perf['roi'] - clf_perf['roi']:.2f}%")

if __name__ == "__main__":
    main()
```

---

## Testing & Validation

### Unit Tests

**New File:** `tests/test_classifier.py`

```python
"""Unit tests for classification model"""
import pytest
import pandas as pd
import numpy as np
from src.models.classifier_trainer import GoalieClassifierTrainer

def test_classifier_training():
    """Test that classifier trains without errors"""
    # Create dummy data
    df = pd.DataFrame({
        'game_date': pd.date_range('2023-10-01', periods=100),
        'betting_line': np.random.uniform(20, 30, 100),
        'saves': np.random.randint(15, 35, 100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    df['over_hit'] = (df['saves'] > df['betting_line']).astype(int)

    trainer = GoalieClassifierTrainer({'model': {'random_state': 42}})
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_classification_data(df)

    model = trainer.train_classifier(X_train, y_train)

    assert model is not None
    assert hasattr(model, 'predict_proba')

def test_prediction_format():
    """Test prediction output format"""
    # Load model and make prediction
    # Verify output has required fields
    pass
```

### Integration Tests

1. **End-to-end pipeline test**
   - Fetch sample lines from API
   - Merge with game data
   - Train classifier
   - Make predictions
   - Validate output format

2. **Data quality tests**
   - Check line coverage %
   - Verify no data leakage
   - Validate feature distributions

### Performance Benchmarks

Track these metrics:

- **Betting Accuracy:** % of correct OVER/UNDER predictions
- **ROI:** Return on investment (simulated betting)
- **Sharpe Ratio:** Risk-adjusted returns
- **Log Loss:** Probability calibration quality
- **Coverage:** % of games with confident predictions

---

## Deployment Strategy

### Staged Rollout

**Phase 1: Shadow Mode (Week 1)**
- Run classification model in parallel with regression
- Log predictions from both
- No user-facing changes
- Monitor performance

**Phase 2: A/B Testing (Week 2-3)**
- 50% of predictions from classifier
- 50% from regression
- Track comparative performance
- Gather feedback

**Phase 3: Full Deployment (Week 4)**
- Switch to best-performing model
- Keep alternative as backup
- Continue monitoring

### Monitoring

Track daily:
- Prediction volume
- Average confidence scores
- Actual vs predicted OVER rate
- Model drift indicators

Set alerts for:
- Accuracy drop >5%
- Coverage drop >10%
- API failures

---

## Rollback Plan

### Trigger Conditions:

Rollback to regression if:
1. Betting accuracy drops >7% vs regression baseline
2. API becomes unreliable (>20% failed line fetches)
3. Critical bug discovered

### Rollback Process:

1. Switch prediction endpoint back to regression model
2. Disable betting line fetching
3. Notify users of temporary regression use
4. Debug and fix classification issues
5. Re-test before re-deploying

### Backup Strategy:

- Keep regression model trained and updated
- Maintain both codepaths
- Easy toggle via config flag

---

## Success Criteria

### Minimum Viable Product:

- [ ] Betting lines integrated for 2023-24 and 2024-25
- [ ] Classification model trained with >75% line coverage
- [ ] Test accuracy >52% (better than random)
- [ ] Prediction pipeline functional

### Success Metrics:

- [ ] Betting accuracy >55% on test set
- [ ] ROI >0% on simulated betting
- [ ] Outperforms regression on ROI by >2%
- [ ] Log loss <0.68

### Stretch Goals:

- [ ] Betting accuracy >60%
- [ ] ROI >5% on test set
- [ ] Sharpe ratio >1.0
- [ ] Production uptime >99%

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Data Integration | Lines fetched, merged, validated |
| 2 | Model Development | Classifier trained, features engineered |
| 3 | Evaluation | A/B test complete, winner selected |
| 4 | Deployment | Production rollout, monitoring active |

**Total Estimated Time:** 4 weeks (part-time)

---

## Configuration Changes

### Update `config/config.yaml`

Add betting API section:

```yaml
betting_api:
  api_key: "YOUR_API_KEY"
  base_url: "https://api.example.com/v1"
  rate_limit: 5  # requests per second

classification:
  confidence_threshold: 0.55  # Minimum probability to recommend bet
  min_line_coverage: 0.75  # Minimum % of games with lines
  use_classifier: true  # Toggle between regression and classification
```

---

## Appendix: File Structure Changes

### New Files Created:

```
saves-model-v3/
├── src/
│   ├── data/
│   │   └── betting_lines_collector.py          [NEW]
│   └── models/
│       ├── classifier_trainer.py               [NEW]
│       └── classifier_predictor.py             [NEW]
├── scripts/
│   ├── collect_betting_lines.py                [NEW]
│   ├── merge_betting_lines.py                  [NEW]
│   ├── train_classifier.py                     [NEW]
│   └── compare_models.py                       [NEW]
├── data/
│   └── raw/
│       └── betting_lines/                      [NEW]
│           ├── 2023-24/
│           └── 2024-25/
├── models/
│   └── trained/
│       ├── xgboost_goalie_classifier.pkl       [NEW]
│       ├── xgboost_goalie_classifier_features.json [NEW]
│       └── xgboost_goalie_classifier_metadata.json [NEW]
├── tests/
│   └── test_classifier.py                      [NEW]
└── CLASSIFICATION_MIGRATION_GUIDE.md           [NEW]
```

### Modified Files:

```
saves-model-v3/
├── src/
│   └── features/
│       └── feature_engineering.py              [MODIFIED: add line features]
├── scripts/
│   └── predict_games.py                        [MODIFIED: add --use-classifier flag]
└── config/
    └── config.yaml                             [MODIFIED: add betting_api section]
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-02
**Status:** Ready for Implementation

