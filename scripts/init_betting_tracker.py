"""
Initialize betting tracker Excel file with proper structure

Creates betting_tracker.xlsx with:
- Bets sheet (main tracking)
- Summary sheet (performance metrics)
- Settings sheet (configuration)
"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from pathlib import Path
from datetime import datetime

def create_betting_tracker():
    """Create new betting tracker Excel file"""

    wb = openpyxl.Workbook()

    # Remove default sheet
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])

    # Create Bets sheet
    bets_sheet = wb.create_sheet('Bets', 0)

    # Define columns
    headers = [
        'game_date', 'game_id', 'goalie_name', 'goalie_id', 'team_abbrev',
        'opponent_team', 'is_home', 'betting_line', 'predicted_saves',
        'prob_over', 'confidence_pct', 'confidence_bucket', 'recommendation',
        'bet_amount', 'bet_selection', 'actual_saves', 'result',
        'profit_loss', 'notes'
    ]

    # Header styling
    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    header_font = Font(bold=True, color='FFFFFF')
    header_alignment = Alignment(horizontal='center', vertical='center')

    # Write headers
    for col_idx, header in enumerate(headers, 1):
        cell = bets_sheet.cell(row=1, column=col_idx)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment

    # Set column widths
    column_widths = {
        'A': 12,  # game_date
        'B': 12,  # game_id
        'C': 18,  # goalie_name
        'D': 12,  # goalie_id
        'E': 12,  # team_abbrev
        'F': 15,  # opponent_team
        'G': 10,  # is_home
        'H': 13,  # betting_line
        'I': 15,  # predicted_saves
        'J': 12,  # prob_over
        'K': 14,  # confidence_pct
        'L': 16,  # confidence_bucket
        'M': 15,  # recommendation
        'N': 12,  # bet_amount
        'O': 14,  # bet_selection
        'P': 13,  # actual_saves
        'Q': 10,  # result
        'R': 12,  # profit_loss
        'S': 30,  # notes
    }

    for col, width in column_widths.items():
        bets_sheet.column_dimensions[col].width = width

    # Freeze top row
    bets_sheet.freeze_panes = 'A2'

    # Create Summary sheet
    summary_sheet = wb.create_sheet('Summary', 1)

    # Summary headers
    summary_sheet['A1'] = 'BETTING PERFORMANCE SUMMARY'
    summary_sheet['A1'].font = Font(bold=True, size=14)
    summary_sheet.merge_cells('A1:D1')

    summary_sheet['A3'] = 'Generated:'
    summary_sheet['B3'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Overall Performance section
    summary_sheet['A5'] = 'OVERALL PERFORMANCE'
    summary_sheet['A5'].font = Font(bold=True, size=12)

    overall_metrics = [
        ('Total Bets', '=COUNTIF(Bets!O:O,"OVER")+COUNTIF(Bets!O:O,"UNDER")'),
        ('Wins', '=COUNTIF(Bets!Q:Q,"WIN")'),
        ('Losses', '=COUNTIF(Bets!Q:Q,"LOSS")'),
        ('Pushes', '=COUNTIF(Bets!Q:Q,"PUSH")'),
        ('Win Rate', '=IF(B6>0,B7/B6,0)'),
        ('Total Profit/Loss', '=SUM(Bets!R:R)'),
        ('ROI %', '=IF(B6>0,B11/(B6*110)*100,0)'),
    ]

    for idx, (label, formula) in enumerate(overall_metrics, 6):
        summary_sheet[f'A{idx}'] = label
        summary_sheet[f'B{idx}'] = formula
        if label in ['Win Rate', 'ROI %']:
            summary_sheet[f'B{idx}'].number_format = '0.0%' if label == 'Win Rate' else '0.00'

    # Performance by Confidence section
    summary_sheet['A14'] = 'PERFORMANCE BY CONFIDENCE LEVEL'
    summary_sheet['A14'].font = Font(bold=True, size=12)

    conf_headers = ['Confidence', 'Bets', 'Wins', 'Win Rate', 'ROI %']
    for col_idx, header in enumerate(conf_headers, 1):
        cell = summary_sheet.cell(row=15, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)

    confidence_levels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']
    for idx, conf_level in enumerate(confidence_levels, 16):
        summary_sheet[f'A{idx}'] = conf_level
        # Formulas will be added by dashboard script

    # Create Settings sheet
    settings_sheet = wb.create_sheet('Settings', 2)

    settings_sheet['A1'] = 'BETTING TRACKER SETTINGS'
    settings_sheet['A1'].font = Font(bold=True, size=14)

    settings = [
        ('Default Unit Size', 1.0),
        ('Min Confidence to Bet (%)', 55),
        ('High Confidence Threshold (%)', 65),
        ('', ''),
        ('Odds Format', '-110'),
        ('Win Payout Multiplier', 0.909),  # 100/110
        ('', ''),
        ('Model Path', 'models/classifier_model.json'),
        ('Feature Count', 89),
    ]

    settings_sheet['A3'] = 'Setting'
    settings_sheet['B3'] = 'Value'
    settings_sheet['A3'].font = Font(bold=True)
    settings_sheet['B3'].font = Font(bold=True)

    for idx, (setting, value) in enumerate(settings, 4):
        settings_sheet[f'A{idx}'] = setting
        settings_sheet[f'B{idx}'] = value

    settings_sheet.column_dimensions['A'].width = 30
    settings_sheet.column_dimensions['B'].width = 20

    # Save workbook
    output_path = Path('betting_tracker.xlsx')
    wb.save(output_path)

    print(f'[OK] Created betting tracker: {output_path.absolute()}')
    print('\nSheets created:')
    print('  1. Bets - Main tracking sheet')
    print('  2. Summary - Performance metrics')
    print('  3. Settings - Configuration')
    print('\nNext steps:')
    print('  1. Run: python scripts/populate_daily_games.py')
    print('  2. Enter betting lines in Excel')
    print('  3. Run: python scripts/generate_predictions.py')

if __name__ == '__main__':
    create_betting_tracker()
