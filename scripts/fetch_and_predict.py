"""
Automated script to fetch betting lines and generate predictions.

This script:
1. Fetches today's NHL schedule
2. Fetches goalie saves betting lines from Underdog Fantasy
3. Matches lines to NHL games and looks up goalie IDs
4. Appends new rows to the tracker (skipping duplicates)
5. Generates predictions for all pending rows
6. Displays EV opportunities

Usage:
    python scripts/fetch_and_predict.py [--date YYYY-MM-DD]
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from betting import (
    NHLBettingData,
    BettingTracker,
    BettingFeatureCalculator,
    BettingPredictor,
    UnderdogFetcher,
    PrizePicksFetcher,
    TheOddsAPIFetcher,
    extract_last_name,
)


def fetch_and_predict(date=None, tracker_file='betting_tracker.xlsx', verbose=False):
    """
    Fetch betting lines and generate predictions.

    Args:
        date: Date string (YYYY-MM-DD). If None, uses today
        tracker_file: Path to betting tracker Excel file
        verbose: If True, display all rows with predictions at the end

    Returns:
        int: Number of EV opportunities found
    """
    # Get date
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*70}")
    print(f"FETCH AND PREDICT - {date}")
    print(f"{'='*70}")

    # Initialize clients
    nhl_data = NHLBettingData()
    underdog = UnderdogFetcher()
    # prizepicks = PrizePicksFetcher()
    sportsbook = TheOddsAPIFetcher()

    # Check if tracker exists, create if needed
    tracker_path = Path(tracker_file)
    if not tracker_path.exists():
        print(f"\n[INFO] Tracker not found, creating new one...")
        # Import and run init
        from init_betting_tracker import create_betting_tracker
        create_betting_tracker()

    tracker = BettingTracker(tracker_file)
    feature_calc = BettingFeatureCalculator()
    predictor = BettingPredictor()

    # Step 1: Fetch NHL schedule
    print(f"\n[1/6] Fetching NHL schedule for {date}...")
    nhl_games = nhl_data.get_todays_games(date)

    if not nhl_games:
        print(f"[OK] No NHL games scheduled for {date}")
        return 0

    print(f"  Found {len(nhl_games)} games")
    for game in nhl_games:
        print(f"    {game['away_team']} @ {game['home_team']} (ID: {game['game_id']})")

    # Step 2: Fetch betting lines from all sources
    print(f"\n[2/6] Fetching betting lines...")

    # Fetch Underdog lines
    print(f"  Fetching Underdog goalie saves lines...")
    underdog_lines = underdog.get_goalie_saves()
    if underdog_lines:
        print(f"    Found {len(underdog_lines)} Underdog lines")
    else:
        print(f"    [WARNING] No Underdog lines found")

    # Fetch PrizePicks lines
    # print(f"  Fetching PrizePicks goalie saves lines...")
    # prizepicks_lines = prizepicks.get_goalie_saves()
    # if prizepicks_lines:
    #     print(f"    Found {len(prizepicks_lines)} PrizePicks lines")
    # else:
    #     print(f"    [WARNING] No PrizePicks lines found (API may be blocked)")

    # Fetch BetMGM/Caesars lines (via The-Odds-API with caching)
    print(f"  Fetching BetMGM/Caesars goalie saves lines...")
    sportsbook_lines = sportsbook.get_goalie_saves(date)
    if sportsbook_lines:
        print(f"    Found {len(sportsbook_lines)} BetMGM/Caesars lines")
    else:
        print(f"    [WARNING] No BetMGM/Caesars lines found")

    # Combine all lines
    all_lines = underdog_lines + sportsbook_lines

    if not all_lines:
        print(f"\n[WARNING] No betting lines found from any source")
        return 0

    print(f"  Total lines: {len(all_lines)}")

    # Step 3: Match lines to NHL games
    print(f"\n[3/6] Matching lines to NHL games...")
    matched_lines = []
    unmatched = []

    # Cache goalie team lookups to avoid repeated API calls
    goalie_team_cache = {}

    for line in all_lines:
        player_name = line['player_name']
        last_name = extract_last_name(player_name)

        # Try to find this goalie using stats leaders (more efficient than checking each game)
        goalie_id = nhl_data.get_goalie_id_by_name(last_name)

        if not goalie_id:
            unmatched.append(player_name)
            continue

        # Get goalie's team from their recent games
        if goalie_id not in goalie_team_cache:
            goalie_team_cache[goalie_id] = _get_goalie_team(nhl_data, goalie_id)

        goalie_team = goalie_team_cache[goalie_id]

        if not goalie_team:
            unmatched.append(player_name)
            continue

        # Find the game involving this goalie's team
        found = False
        for game in nhl_games:
            game_id = game['game_id']

            if goalie_team == game['home_team']:
                matched_lines.append({
                    'game_date': date,
                    'game_id': game_id,
                    'book': line['book'],
                    'goalie_name': last_name,
                    'goalie_id': goalie_id,
                    'team_abbrev': game['home_team'],
                    'opponent_team': game['away_team'],
                    'is_home': 1,
                    'betting_line': line['line'],
                    'line_over': line['line_over'],
                    'line_under': line['line_under'],
                })
                found = True
                break
            elif goalie_team == game['away_team']:
                matched_lines.append({
                    'game_date': date,
                    'game_id': game_id,
                    'book': line['book'],
                    'goalie_name': last_name,
                    'goalie_id': goalie_id,
                    'team_abbrev': game['away_team'],
                    'opponent_team': game['home_team'],
                    'is_home': 0,
                    'betting_line': line['line'],
                    'line_over': line['line_over'],
                    'line_under': line['line_under'],
                })
                found = True
                break

        if not found:
            unmatched.append(f"{player_name} ({goalie_team} not playing today)")

    print(f"  Matched: {len(matched_lines)} lines")
    if unmatched:
        print(f"  Unmatched: {len(unmatched)} - {', '.join(unmatched[:5])}{'...' if len(unmatched) > 5 else ''}")

    if not matched_lines:
        print(f"\n[WARNING] No lines could be matched to today's games")
        return 0

    # Steps 4+5: Predict all matched lines, deduplicate by predictions, append new rows
    print(f"\n[4/6] Predicting all lines and checking for changes...")

    existing_df = tracker.get_todays_games(date)

    new_lines = []
    predictions_list = []
    ev_opportunities = []
    skipped_identical = 0

    for line in matched_lines:
        game_id = line['game_id']
        goalie_id = line['goalie_id']
        goalie_name = line['goalie_name']
        team = line['team_abbrev']
        opponent = line['opponent_team']
        is_home = line['is_home']
        betting_line = line['betting_line']
        line_over_odds = line['line_over']
        line_under_odds = line['line_under']
        book = line['book']

        try:
            recent_games = nhl_data.get_goalie_recent_games(
                goalie_id,
                season='20252026',
                n_games=15
            )

            features_df = feature_calc.prepare_prediction_features(
                goalie_id=goalie_id,
                team=team,
                opponent=opponent,
                is_home=is_home,
                game_date=date,
                recent_games=recent_games,
                betting_line=betting_line,
                nhl_fetcher=nhl_data
            )

            prediction = predictor.predict(
                features_df,
                betting_line=betting_line,
                line_over_odds=line_over_odds if pd.notna(line_over_odds) else None,
                line_under_odds=line_under_odds if pd.notna(line_under_odds) else None
            )

        except Exception as e:
            print(f"    [ERROR] {goalie_name}: {e}")
            continue

        predicted_saves = prediction.get('predicted_saves')
        prob_over = prediction.get('prob_over')
        recommendation = prediction.get('recommendation')
        ev = prediction.get('recommended_ev')

        # Collect EV opportunity regardless of whether this is a new row
        if ev is not None and ev >= 0.12:
            ev_opportunities.append({
                'goalie_name': goalie_name,
                'team': team,
                'opponent': opponent,
                'book': book,
                'recommendation': recommendation,
                'line': betting_line,
                'odds': line_over_odds if recommendation == 'OVER' else line_under_odds,
                'ev': ev,
                'prob': prob_over if recommendation == 'OVER' else (1 - prob_over),
            })

        # Check if an identical row already exists (same odds AND same predictions)
        is_duplicate = False
        if not existing_df.empty:
            mask = (
                (existing_df['game_id'] == game_id) &
                (existing_df['goalie_name'].fillna('').astype(str).str.lower() == goalie_name.lower()) &
                (existing_df['betting_line'] == betting_line) &
                (existing_df['line_over'] == line_over_odds) &
                (existing_df['line_under'] == line_under_odds)
            )
            if 'book' in existing_df.columns:
                mask = mask & (existing_df['book'] == book)

            matching = existing_df[mask]
            if not matching.empty:
                def _round_pred(v, decimals):
                    try:
                        return round(float(v), decimals)
                    except (TypeError, ValueError):
                        return None

                for _, existing_row in matching.iterrows():
                    if (_round_pred(existing_row.get('predicted_saves'), 1) == _round_pred(predicted_saves, 1) and
                            _round_pred(existing_row.get('prob_over'), 3) == _round_pred(prob_over, 3) and
                            existing_row.get('recommendation') == recommendation and
                            _round_pred(existing_row.get('ev'), 3) == _round_pred(ev, 3)):
                        is_duplicate = True
                        break

        if is_duplicate:
            skipped_identical += 1
            continue

        new_lines.append(line)

        pred_entry = dict(prediction)
        pred_entry.update({
            'game_id': game_id,
            'goalie_id': goalie_id,
            'goalie_name': goalie_name,
            'game_date': date,
            'book': book,
            'betting_line': betting_line,
            'line_over': line_over_odds,
            'line_under': line_under_odds,
        })
        predictions_list.append(pred_entry)

    if new_lines:
        new_df = pd.DataFrame(new_lines)
        tracker.append_games(new_df)
        print(f"  Added {len(new_lines)} new rows")

        shown = 0
        for line in new_lines:
            over_str = f"{line['line_over']:+d}" if line['line_over'] else 'N/A'
            under_str = f"{line['line_under']:+d}" if line['line_under'] else 'N/A'
            print(f"    {line['goalie_name']} ({line['team_abbrev']} vs {line['opponent_team']}) - "
                  f"Line: {line['betting_line']}, Over: {over_str}, Under: {under_str}")
            shown += 1
            if shown >= 20 and len(new_lines) > 25:
                print(f"    ... and {len(new_lines) - shown} more rows")
                break
    else:
        print(f"  No new rows to add (all lines identical to existing predictions)")

    if skipped_identical:
        print(f"  Skipped {skipped_identical} line(s) with unchanged predictions")

    # Step 5: Write predictions for new rows
    print(f"\n[5/6] Writing predictions...")

    if predictions_list:
        predictions_df = pd.DataFrame(predictions_list)
        tracker.update_predictions(predictions_df)
        print(f"  Updated {len(predictions_list)} predictions")
    else:
        print(f"  No new predictions to write")

    # Step 6: Display results
    print(f"\n[6/6] Summary")
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Lines fetched: {len(all_lines)} (Underdog: {len(underdog_lines)}, BetMGM/Caesars: {len(sportsbook_lines)})")
    print(f"Lines matched: {len(matched_lines)}")
    print(f"New rows added: {len(new_lines)} (skipped {skipped_identical} identical)")
    print(f"Predictions written: {len(predictions_list)}")

    if ev_opportunities:
        print(f"\nEV OPPORTUNITIES (>= 12%):")
        print("-" * 70)
        ev_opportunities.sort(key=lambda x: x['ev'], reverse=True)
        for opp in ev_opportunities:
            odds_str = f"{int(opp['odds']):+d}" if opp['odds'] else 'N/A'
            print(f"  {opp['goalie_name']} ({opp['team']} vs {opp['opponent']}) @ {opp['book']}")
            print(f"    {opp['recommendation']} {opp['line']} @ {odds_str}")
            print(f"    EV: {opp['ev']:+.1%}, Model Prob: {opp['prob']:.1%}")
    else:
        print(f"\nNo EV opportunities >= 12% found")

    print(f"\n{'='*70}")

    # Verbose output: show all rows for this date
    if verbose:
        print(f"\nALL LINES FOR {date}:")
        print("-" * 70)
        all_rows = tracker.get_todays_games(date)
        if not all_rows.empty:
            # Identify the current (latest) row index for each goalie+book combo
            # Rows are in insertion order, so last occurrence = current line
            current_idx = set(
                all_rows.groupby(['goalie_name', 'book']).apply(lambda g: g.index[-1], include_groups=False).values
            )

            # Sort by EV descending
            all_rows = all_rows.sort_values('ev', ascending=False, na_position='last')
            for idx, row in all_rows.iterrows():
                goalie = row.get('goalie_name', 'Unknown')
                team = row.get('team_abbrev', '')
                opponent = row.get('opponent_team', '')
                book = row.get('book', 'Unknown')
                line = row.get('betting_line')
                line_over = row.get('line_over')
                line_under = row.get('line_under')
                pred_saves = row.get('predicted_saves')
                prob_over = row.get('prob_over')
                rec = row.get('recommendation', '')
                ev = row.get('ev')
                outdated = idx not in current_idx

                line_str = f"{line:.1f}" if pd.notna(line) else "N/A"
                over_str = f"{int(line_over):+d}" if pd.notna(line_over) else "N/A"
                under_str = f"{int(line_under):+d}" if pd.notna(line_under) else "N/A"
                pred_str = f"{pred_saves:.1f}" if pd.notna(pred_saves) else "N/A"
                prob_str = f"{prob_over:.1%}" if pd.notna(prob_over) else "N/A"
                ev_str = f"{ev:+.1%}" if pd.notna(ev) else "N/A"
                stale_tag = " [OUTDATED]" if outdated else ""

                print(f"  {goalie:12} ({team:3} vs {opponent:3}) @ {book:10} | Line: {line_str:5} (O:{over_str:5}/U:{under_str:5}) | "
                      f"Pred: {pred_str:5} | P(Over): {prob_str:6} | {rec:6} | EV: {ev_str}{stale_tag}")
        else:
            print("  No data found")
        print(f"{'='*70}")

    return len(ev_opportunities)


def _get_goalie_team(nhl_data, goalie_id):
    """
    Get the team abbreviation for a goalie by looking up their recent games.

    Args:
        nhl_data: NHLBettingData instance
        goalie_id: NHL player ID

    Returns:
        str: Team abbreviation or None
    """
    try:
        recent_games = nhl_data.get_goalie_recent_games(goalie_id, n_games=1)
        if recent_games:
            game = recent_games[0]
            # teamAbbrev is a direct string, not a nested dict
            return game.get('teamAbbrev')
    except Exception:
        pass
    return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fetch betting lines and generate predictions'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to fetch lines for (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='betting_tracker.xlsx',
        help='Path to betting tracker file. Default: betting_tracker.xlsx'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Display all rows with predictions for the date'
    )

    args = parser.parse_args()

    try:
        fetch_and_predict(date=args.date, tracker_file=args.tracker, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
