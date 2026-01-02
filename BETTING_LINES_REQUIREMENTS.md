# Historical Betting Lines Integration Requirements

## Executive Summary

This document outlines the requirements for obtaining historical goalie save over/under betting lines to enable classification-based modeling for NHL goalie save predictions.

---

## Current Dataset Overview

### Data Coverage
- **Time Period:** 3 NHL seasons
  - 2022-2023 (Season ID: 20222023)
  - 2023-2024 (Season ID: 20232024)
  - 2024-2025 (Season ID: 20242025)
- **Total Games:** 3,936 regular season games
- **After Filtering:** 7,465 goalie performances (starters only, complete games)
- **Unique Goalies:** 170

### Game Identification Fields
Each game in our dataset can be uniquely identified by:

1. **Primary Identifier:**
   - `game_id` (int) - NHL's unique game identifier
   - Format: `2023020204` (SSSSTTGGGG where SSSS=season, TT=game type, GGGG=game number)
   - Example: `2023020204` = 2023-24 season, regular season (02), game 204

2. **Secondary Identifiers:**
   - `game_date` (datetime) - Format: `YYYY-MM-DD` (e.g., `2023-11-10`)
   - `team_abbrev` (string) - 3-letter team code (e.g., `TOR`, `BOS`)
   - `opponent_team` (string) - 3-letter opponent code
   - `is_home` (boolean) - Whether the goalie's team was home

3. **Goalie Identifiers:**
   - `goalie_id` (int) - NHL's unique player ID (e.g., `8478402`)
   - `goalie_name` (string) - Full name (available but not in processed data)

### Sample Data Structure
```json
{
  "game_id": 2023020204,
  "game_date": "2023-11-10",
  "season": "20232024",
  "team_abbrev": "TOR",
  "opponent_team": "BOS",
  "is_home": true,
  "goalie_id": 8478402,
  "saves": 28,
  "shots_against": 31,
  "toi_seconds": 3600
}
```

---

## Required Betting Line Data

### Must-Have Fields

For each goalie performance, we need:

```json
{
  "game_identifier": "2023020204 OR date + teams",
  "goalie_identifier": "name OR player_id",
  "save_line": 25.5,
  "line_timestamp": "2023-11-10T18:00:00Z",
  "sportsbook": "DraftKings"
}
```

#### Field Descriptions:

1. **Game Matching Fields** (at least ONE of):
   - `game_id` - NHL game ID (ideal, most reliable)
   - `game_date` + `home_team` + `away_team` - Alternative matching
   - `event_id` - Sportsbook's event identifier (if mappable)

2. **Goalie Matching Fields** (at least ONE of):
   - `player_name` - Full name (e.g., "Ilya Samsonov")
     - ⚠️ May need fuzzy matching for variations
   - `player_id` - NHL player ID (ideal if available)
   - ⚠️ **Critical:** Must be able to identify WHICH goalie in the game

3. **Line Data** (REQUIRED):
   - `save_line` (float) - The over/under line (e.g., 25.5, 28.5)
   - Must be the **closing line** or line closest to game start
   - Avoid opening lines (too far from game time, less accurate)

4. **Metadata** (nice to have):
   - `line_timestamp` - When line was set/last updated
   - `sportsbook` - Which book provided the line
   - `over_odds` / `under_odds` - American odds (e.g., -110, +105)

### Data Format Preferences

**Ideal Format:**
```csv
game_id,game_date,home_team,away_team,goalie_name,save_line,odds_timestamp,sportsbook
2023020204,2023-11-10,TOR,BOS,Ilya Samsonov,25.5,2023-11-10T18:45:00Z,DraftKings
2023020204,2023-11-10,TOR,BOS,Linus Ullmark,28.5,2023-11-10T18:45:00Z,DraftKings
```

**Acceptable Formats:**
- CSV, JSON, Parquet, SQLite
- API endpoint returning JSON/XML

---

## API/Data Source Requirements

### Functional Requirements

1. **Historical Coverage:**
   - Must cover at least: **October 2022 - April 2025**
   - Games needed: ~7,465 goalie performances across 3,936 games
   - Bonus: Earlier seasons for additional training data

2. **Sport/League:**
   - **NHL** (National Hockey League)
   - **Regular Season** games (game type = 2)
   - **North American markets** (US/Canada sportsbooks)

3. **Prop Type:**
   - **Goalie saves over/under** (also called "goalie total saves")
   - NOT team total saves
   - NOT general game props

4. **Query Capabilities** (at least ONE of):
   - Query by game ID or game identifier
   - Query by date range
   - Query by team
   - Bulk download/export for date range

5. **Data Completeness:**
   - Need lines for **both goalies** in each game (home & away)
   - Acceptable to have missing data for some games
   - Target: >80% coverage of our 7,465 performances

### Technical Requirements

1. **Access Method:**
   - REST API (preferred)
   - Bulk data download
   - Database access
   - Web scraping (last resort)

2. **Authentication:**
   - API key, OAuth, or free access
   - Budget: Open to paid solutions if reasonable (<$100/month)

3. **Rate Limits:**
   - Must allow fetching ~4,000 games worth of data
   - Acceptable: throttled requests over several hours/days

4. **Data Quality:**
   - Lines should be from reputable sportsbooks (DraftKings, FanDuel, BetMGM, etc.)
   - Prefer closing lines over opening lines
   - Consistent formatting/structure

---

## Known Candidate APIs

### Options to Investigate:

1. **The Odds API** (https://the-odds-api.com/)
   - Pros: Well-documented, NHL support
   - Cons: Historical data may be limited, paid tiers
   - **Check:** NHL goalie prop historical availability

2. **OddsJam API**
   - Pros: Focuses on player props
   - **Check:** Historical NHL goalie saves data

3. **Sportsbook Review (SBR)**
   - Pros: Long historical archive
   - Cons: May require scraping
   - **Check:** NHL goalie prop archives

4. **PropSwap API**
   - Pros: Player prop specialization
   - **Check:** Historical data availability

5. **Pinnacle Sports API**
   - Pros: Sharp lines, API access
   - **Check:** Historical NHL props

6. **Action Network**
   - **Check:** Historical NHL prop data access

### Evaluation Criteria:

For each API, determine:
- ✅ Has NHL goalie save O/U props?
- ✅ Has historical data back to Oct 2022?
- ✅ Provides game/goalie identifiers we can match?
- ✅ Reasonable cost/access terms?
- ✅ Bulk export or efficient querying?

---

## Data Matching Strategy

### Matching Process:

1. **Primary Match** (if API provides NHL game IDs):
   ```python
   df_lines = pd.merge(
       df_games,
       df_betting_lines,
       on='game_id',
       how='left'
   )
   ```

2. **Secondary Match** (if using date + teams):
   ```python
   df_lines = pd.merge(
       df_games,
       df_betting_lines,
       left_on=['game_date', 'team_abbrev', 'opponent_team'],
       right_on=['game_date', 'home_team', 'away_team'],
       how='left'
   )
   ```

3. **Goalie Name Matching** (if needed):
   - Use fuzzy matching (e.g., `fuzzywuzzy`, `RapidFuzz`)
   - Handle variations: "Ilya Samsonov" vs "I. Samsonov"
   - Build name→ID mapping table

### Quality Checks:

After matching, validate:
- ✅ Lines are reasonable (typically 15.5 - 35.5 saves)
- ✅ Both goalies in game have lines (or at least starter)
- ✅ No duplicate lines per goalie per game
- ✅ Line timestamp is before game start time
- ✅ Line is for the correct goalie (not backup)

---

## Expected Output

### Target Schema:

After integration, each row in training data should have:

```python
{
    # Existing fields
    'game_id': 2023020204,
    'goalie_id': 8478402,
    'saves': 28,
    'game_date': '2023-11-10',

    # New fields
    'betting_line': 25.5,           # Over/under line
    'line_source': 'DraftKings',    # Sportsbook
    'line_timestamp': '2023-11-10T18:45:00Z',
    'over_hit': True,               # Did goalie go OVER? (28 > 25.5)
    'line_margin': 2.5,             # How far over/under? (28 - 25.5)
}
```

### Target Coverage:

- **Minimum Acceptable:** 60% of games have betting lines (4,479 / 7,465)
- **Good Coverage:** 80%+ (5,972+ / 7,465)
- **Ideal Coverage:** 90%+ (6,719+ / 7,465)

Missing lines are acceptable if:
- Random (not biased toward certain teams/goalies)
- Can exclude from classification training (still use for regression)

---

## Implementation Plan (Post-API Selection)

### Phase 1: Data Collection
1. Test API with sample queries
2. Develop data fetching script
3. Download historical lines for 3 seasons
4. Store in `/data/raw/betting_lines/`

### Phase 2: Data Integration
1. Build matching logic (game + goalie identification)
2. Merge with existing `training_data.parquet`
3. Create `over_hit` binary target variable
4. Validate data quality

### Phase 3: Model Development
1. Train binary classifier (XGBoost or similar)
2. Features: existing 191 + betting line
3. Target: `over_hit` (0/1)
4. Optimize for log loss / betting accuracy

### Phase 4: Evaluation
1. A/B test: regression vs classification
2. Metric: Betting ROI on test set
3. Choose winner for production

---

## Questions for API Vendors

When evaluating APIs, ask:

1. **Coverage:**
   - "Do you have NHL goalie save O/U props?"
   - "How far back does historical data go?"
   - "What % of NHL games have goalie props?"

2. **Data Structure:**
   - "How are games identified?" (Show sample response)
   - "How are players identified?" (Name? ID? Both?)
   - "Do you provide closing lines or just snapshots?"

3. **Access:**
   - "Can I query by date range?"
   - "What's the rate limit?"
   - "Is there a bulk export option?"

4. **Cost:**
   - "What's the pricing for historical data access?"
   - "Is there a free tier for testing?"

5. **Format:**
   - "What format is data provided in?" (JSON/CSV/etc.)
   - "Can you provide sample historical response?"

---

## Contact for Questions

For technical questions about our existing data structure or integration requirements, reference:
- Dataset: `s:\Documents\GitHub\saves-model-v3\data\processed\training_data.parquet`
- Raw game data: `s:\Documents\GitHub\saves-model-v3\data\raw\boxscores\`
- Data dictionary: See feature engineering pipeline in `src/features/feature_engineering.py`

---

## Appendix: Sample Game Data

### Sample Games Needing Lines:

```
Game ID: 2022020001, Date: 2022-10-07, TOR @ MTL
Game ID: 2022020002, Date: 2022-10-07, NSH @ SJS
Game ID: 2023020001, Date: 2023-10-10, BOS @ CHI
Game ID: 2024020001, Date: 2024-10-08, FLA @ BOS
```

### Team Abbreviations (30 teams):
ANA, ARI, BOS, BUF, CAR, CBJ, CGY, CHI, COL, DAL, DET, EDM, FLA, LAK, MIN, MTL, NJD, NSH, NYI, NYR, OTT, PHI, PIT, SEA, SJS, STL, TBL, TOR, VAN, VGK, WPG, WSH

### Date Range by Season:
- **2022-23:** Oct 7, 2022 → Apr 13, 2023
- **2023-24:** Oct 10, 2023 → Apr 18, 2024
- **2024-25:** Oct 8, 2024 → Apr 17, 2025

---

**Last Updated:** 2026-01-02
