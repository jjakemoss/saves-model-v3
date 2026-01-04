"""
Excel manager for betting tracker operations
"""
import openpyxl
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


class BettingTracker:
    """Manage betting tracker Excel file"""

    def __init__(self, tracker_file='betting_tracker.xlsx'):
        """
        Initialize tracker manager

        Args:
            tracker_file: Path to betting tracker Excel file
        """
        self.tracker_file = Path(tracker_file)

        if not self.tracker_file.exists():
            raise FileNotFoundError(
                f"Betting tracker not found: {tracker_file}\n"
                f"Run: python scripts/init_betting_tracker.py"
            )

    def append_games(self, games_df):
        """
        Append new game rows to Bets sheet

        Args:
            games_df: pd.DataFrame with game info (partial columns)
        """
        # Load workbook
        wb = openpyxl.load_workbook(self.tracker_file)
        ws = wb['Bets']

        # Find next empty row
        next_row = ws.max_row + 1

        # Append each game
        for _, game in games_df.iterrows():
            row_data = [
                game.get('game_date'),
                game.get('game_id'),
                game.get('goalie_name'),
                game.get('goalie_id'),
                game.get('team_abbrev'),
                game.get('opponent_team'),
                game.get('is_home'),
                '',  # betting_line (user fills)
                '',  # predicted_saves
                '',  # prob_over
                '',  # confidence_pct
                '',  # confidence_bucket
                '',  # recommendation
                '',  # bet_amount
                'NONE',  # bet_selection
                '',  # actual_saves
                '',  # result
                '',  # profit_loss
                '',  # notes
            ]

            for col_idx, value in enumerate(row_data, 1):
                ws.cell(row=next_row, column=col_idx, value=value)

            next_row += 1

        # Save
        wb.save(self.tracker_file)
        print(f"[OK] Appended {len(games_df)} games to {self.tracker_file}")

    def update_predictions(self, predictions_df):
        """
        Update prediction columns for games

        Args:
            predictions_df: pd.DataFrame with game_id and prediction columns
        """
        # Load existing data
        df = pd.read_excel(self.tracker_file, sheet_name='Bets')

        # Update predictions by game_id and goalie_id
        for _, pred in predictions_df.iterrows():
            game_id = pred['game_id']
            goalie_id = pred['goalie_id']
            mask = (df['game_id'] == game_id) & (df['goalie_id'].isna() | (df['goalie_id'] == goalie_id))

            if mask.any():
                # Update goalie_id if it was looked up
                if pd.notna(goalie_id):
                    df.loc[mask, 'goalie_id'] = goalie_id

                df.loc[mask, 'predicted_saves'] = pred.get('predicted_saves')
                df.loc[mask, 'prob_over'] = pred.get('prob_over')
                df.loc[mask, 'confidence_pct'] = pred.get('confidence_pct')
                df.loc[mask, 'confidence_bucket'] = pred.get('confidence_bucket')
                df.loc[mask, 'recommendation'] = pred.get('recommendation')

        # Write back to Excel
        with pd.ExcelWriter(self.tracker_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Bets', index=False)

        print(f"[OK] Updated {len(predictions_df)} predictions")

    def update_results(self, results_df):
        """
        Update actual saves and results for completed games

        Args:
            results_df: pd.DataFrame with game_id, actual_saves, result, profit_loss
        """
        # Load existing data
        df = pd.read_excel(self.tracker_file, sheet_name='Bets')

        # Update results by game_id
        for _, result in results_df.iterrows():
            game_id = result['game_id']
            goalie_id = result['goalie_id']
            mask = (df['game_id'] == game_id) & (df['goalie_id'] == goalie_id)

            if mask.any():
                df.loc[mask, 'actual_saves'] = result.get('actual_saves')
                df.loc[mask, 'result'] = result.get('result')
                df.loc[mask, 'profit_loss'] = result.get('profit_loss')

        # Write back
        with pd.ExcelWriter(self.tracker_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Bets', index=False)

        print(f"[OK] Updated results for {len(results_df)} games")

    def get_todays_games(self, date=None):
        """
        Get games for a specific date from tracker

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today

        Returns:
            pd.DataFrame: Games for that date
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        df = pd.read_excel(self.tracker_file, sheet_name='Bets')

        # Filter by date
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
        todays_games = df[df['game_date'] == date]

        return todays_games

    def get_pending_predictions(self, date=None):
        """
        Get games that need predictions (have goalie name and betting line but no prediction)

        Args:
            date: Date string. If None, uses today

        Returns:
            pd.DataFrame: Games needing predictions
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        df = pd.read_excel(self.tracker_file, sheet_name='Bets')
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # Filter: today's date, has goalie name, has betting line, no prediction yet
        mask = (
            (df['game_date'] == date) &
            (df['goalie_name'].notna()) &
            (df['goalie_name'] != '') &
            (df['betting_line'].notna()) &
            (df['betting_line'] != '') &
            ((df['predicted_saves'].isna()) | (df['predicted_saves'] == ''))
        )

        return df[mask]

    def get_pending_results(self, date=None):
        """
        Get games needing results update

        Args:
            date: Date string. If None, uses yesterday

        Returns:
            pd.DataFrame: Games needing results
        """
        if date is None:
            yesterday = datetime.now() - pd.Timedelta(days=1)
            date = yesterday.strftime('%Y-%m-%d')

        df = pd.read_excel(self.tracker_file, sheet_name='Bets')
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # Filter: specific date, no results yet
        mask = (
            (df['game_date'] == date) &
            ((df['actual_saves'].isna()) | (df['actual_saves'] == ''))
        )

        return df[mask]

    def backup_to_csv(self, backup_dir='data/betting_history'):
        """
        Save backup of Bets sheet to CSV

        Args:
            backup_dir: Directory for backups
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        # Read Bets sheet
        df = pd.read_excel(self.tracker_file, sheet_name='Bets')

        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = backup_path / f'betting_tracker_{timestamp}.csv'
        df.to_csv(csv_file, index=False)

        print(f"[OK] Backup saved: {csv_file}")

        return csv_file
