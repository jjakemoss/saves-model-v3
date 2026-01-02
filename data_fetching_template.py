import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "your_key_here"
BASE_URL = "https://api.the-odds-api.com/v4"

def fetch_nhl_games_for_date(date_str):
    """Fetch NHL games for a specific date"""
    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events"
    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T18:00:00Z"  # 6 PM UTC (1 PM EST)
    }
    response = requests.get(url, params=params)
    return response.json()

def fetch_goalie_props(event_id, date_str):
    """Fetch goalie saves props for a specific game"""
    url = f"{BASE_URL}/historical/sports/icehockey_nhl/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'date': f"{date_str}T23:00:00Z",  # Close to game time
        'regions': 'us',
        'markets': 'player_total_saves'
    }
    response = requests.get(url, params=params)
    return response.json()

# Iterate through your game dates
dates = pd.date_range('2023-10-10', '2024-04-18', freq='D')
all_lines = []

for date in dates:
    date_str = date.strftime('%Y-%m-%d')
    games = fetch_nhl_games_for_date(date_str)
    
    for game in games:
        props = fetch_goalie_props(game['id'], date_str)
        all_lines.append(props)
    
    # Respect rate limits
    time.sleep(0.5)

# Save raw data
with open('data/raw/betting_lines/odds_api_historical.json', 'w') as f:
    json.dump(all_lines, f)