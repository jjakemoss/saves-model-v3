"""Line-relative feature computation, used by the live training pipeline.

Added Feb 1, 2026 to fix a structural bug where the model's predictions were
nearly insensitive to the actual betting line (see docs/MODEL_TRAINING_GUIDE.md
§2 Phase 3). Imported directly by scripts/build_multibook_training_data.py.

This file used to also contain a full feature-engineering orchestrator
(shot quality, Corsi/Fenwick, matchup, interaction features) that was
superseded by scripts/create_clean_features.py back in January 2026 and
never consumed by anything downstream. That dead code was removed during a
repo cleanup -- see git history if you need it back.
"""

import numpy as np


def compute_line_relative_features(df):
    """
    Compute line-relative features from betting_line and rolling averages.

    These features explicitly encode the gap between the betting line and
    the goalie's recent performance, helping the model learn line sensitivity.

    Args:
        df: DataFrame with betting_line and saves_rolling_* columns

    Returns:
        DataFrame with 6 new columns added:
        - line_vs_rolling_3/5/10: raw difference (betting_line - rolling avg)
        - line_z_score_3/5/10: standardized difference (in std devs)
    """
    for window in [3, 5, 10]:
        rolling_col = f'saves_rolling_{window}'
        std_col = f'saves_rolling_std_{window}'

        if rolling_col in df.columns:
            df[f'line_vs_rolling_{window}'] = df['betting_line'] - df[rolling_col]
        else:
            df[f'line_vs_rolling_{window}'] = 0.0

        if rolling_col in df.columns and std_col in df.columns:
            df[f'line_z_score_{window}'] = np.where(
                df[std_col] > 0.01,
                (df['betting_line'] - df[rolling_col]) / df[std_col],
                0.0
            )
        else:
            df[f'line_z_score_{window}'] = 0.0

    return df
