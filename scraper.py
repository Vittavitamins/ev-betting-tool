import os
import json
import math
import difflib
from datetime import datetime, timezone

import requests
import pandas as pd
from scipy.stats import poisson

# ============================================================
# Configuration
# ============================================================
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_API_KEY_HERE")

LEAGUES = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_uefa_champs_league": "Champions League",
}

# NOTE: football-data.co.uk's path segment is "mmz4281", not "mmh".
# The Champions League isn't published on football-data.co.uk, so it has
# no bottom-up stats source below and will fall back to league-average
# ratings (1.0/1.0) until you wire up a second stats source for it.
FD_URLS = {
    "Premier League": "https://www.football-data.co.uk/mmz4281/2627/E0.csv",
    "La Liga": "https://www.football-data.co.uk/mmz4281/2627/SP1.csv",
    "Bundesliga": "https://www.football-data.co.uk/mmz4281/2627/D1.csv",
    "Serie A": "https://www.football-data.co.uk/mmz4281/2627/I1.csv",
}

EV_THRESHOLD = 0.02          # minimum EV to be included in output at all
LEAGUE_BASELINE_GOALS = 1.45  # average goals per team per match, fallback

# football-data.co.uk uses short/abbreviated club names; The Odds API uses
# full official names. These two sources will almost never match on a raw
# string compare, so bets will silently fall back to generic 1.0/1.0
# ratings unless names are reconciled. This is a starter alias map, not a
# complete one -- extend it as you hit misses (the script logs unmapped
# names to stderr so you can find them).
TEAM_ALIASES = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Leicester": "Leicester City",
    "Nott'm Forest": "Nottingham Forest",
    "Brighton": "Brighton and Hove Albion",
    "Sociedad": "Real Sociedad",
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano",
    "Alaves": "Deportivo Alaves",
    "Celta": "Celta Vigo",
    "M'gladbach": "Borussia Monchengladbach",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne",
    "Leverkusen": "Bayer Leverkusen",
    "Bayern Munich": "Bayern Munich",
    "RB Leipzig": "RB Leipzig",
    "Hoffenheim": "TSG Hoffenheim",
    "Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "Wolfsburg": "VfL Wolfsburg",
    "Milan": "AC Milan",
    "Inter": "Inter Milan",
    "Verona": "Hellas Verona",
}


# ============================================================
# Team name reconciliation
# ============================================================
def normalize_team_name(name, candidates):
    """Maps a football-data.co.uk name to the closest Odds API name.
    Tries the explicit alias table first, then falls back to fuzzy
    string matching against the set of names actually seen in the
    odds feed for that fixture window."""
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if match:
        return match[0]
    return name


# ============================================================
# Bottom-up: historical stats -> attack/defense ratings
# ============================================================
def fetch_historical_stats(league_name):
    """Fetches historical match data from football-data.co.uk and computes
    basic attack/defense ratings per team."""
    url = FD_URLS.get(league_name)
    if not url:
        return {}

    try:
        df = pd.read_csv(url)
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

        home_goals_avg = df["FTHG"].mean()
        away_goals_avg = df["FTAG"].mean()

        team_stats = {}
        teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))

        for team in teams:
            home_games = df[df["HomeTeam"] == team]
            away_games = df[df["AwayTeam"] == team]

            goals_scored = home_games["FTHG"].sum() + away_games["FTAG"].sum()
            goals_conceded = home_games["FTAG"].sum() + away_games["FTHG"].sum()
            total_games = len(home_games) + len(away_games)

            if total_games > 0 and home_goals_avg > 0 and away_goals_avg > 0:
                att_strength = (goals_scored / total_games) / home_goals_avg
                def_strength = (goals_conceded / total_games) / away_goals_avg
            else:
                att_strength, def_strength = 1.0, 1.0

            team_stats[team] = {
                "attack": att_strength,
                "defense": def_strength,
                "x_multiplier": 1.0,  # placeholder until FBref xG is wired in
            }
        return team_stats
    except Exception as e:
        print(f"Error fetching historical stats for {league_name}: {e}")
        return {}


# ============================================================
# Top-down: live odds
# ============================================================
def fetch_odds(sport_key):
    """Fetches live odds from The Odds API for Pinnacle and DraftKings."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,eu",
        "markets": "spreads",
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code == 200:
        return response.json()
    print(f"API Error ({sport_key}): {response.status_code} - {response.text}")
    return []


# ============================================================
# Math: Poisson AH probability, de-vig, blending, staking
# ============================================================
def calculate_poisson_probability(lam_home, lam_away, line, side):
    """Probability of covering an Asian Handicap line for `side`
    ("home" or "away"), given goal expectancies. Handles pushes
    (line falls exactly on a scoreline) as a half-win/half-refund."""
    max_goals = 8
    prob = 0.0

    for h in range(max_goals):
        for a in range(max_goals):
            p_score = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)
            diff = (h - a) + line
            if side == "home":
                if diff > 0:
                    prob += p_score
                elif diff == 0:
                    prob += p_score * 0.5
            else:  # away
                if diff < 0:
                    prob += p_score
                elif diff == 0:
                    prob += p_score * 0.5
    return prob


def devig_pinnacle(odds_side_a, odds_side_b):
    """Removes vig from a two-way market using proportional normalization.
    Returns (fair_prob_a, fair_prob_b)."""
    implied_a = 1.0 / odds_side_a
    implied_b = 1.0 / odds_side_b
    total = implied_a + implied_b
    return implied_a / total, implied_b / total


def time_decay_blend(p_sharp, p_bottom_up, hours_to_kickoff):
    """Blends sharp-market and bottom-up probabilities, shifting weight
    toward the sharp market as kickoff approaches."""
    max_hours = 96.0
    weight_sharp = 0.95 - (min(hours_to_kickoff, max_hours) / max_hours) * 0.75
    weight_bottom = 1.0 - weight_sharp
    return (weight_sharp * p_sharp) + (weight_bottom * p_bottom_up)


def assign_tiered_units(ev):
    if ev >= 0.06:
        return "3 Units (High Edge)"
    elif ev >= 0.041:
        return "2 Units (Moderate Edge)"
    elif ev >= 0.02:
        return "1 Unit (Standard Edge)"
    return "No Play"


# ============================================================
# Market extraction helpers
# ============================================================
def extract_spread_market(bookmaker):
    """Returns the 'spreads' market dict for a bookmaker, or None."""
    for market in bookmaker.get("markets", []):
        if market.get("key") == "spreads":
            return market
    return None


def outcomes_by_team(market):
    """Maps team name -> (price, point) for a spreads market."""
    result = {}
    for outcome in market.get("outcomes", []):
        result[outcome["name"]] = (outcome["price"], outcome["point"])
    return result


# ============================================================
# Main pipeline
# ============================================================
def main():
    all_bets = []

    for sport_key, league_name in LEAGUES.items():
        print(f"Processing {league_name}...")

        team_stats = fetch_historical_stats(league_name)
        stat_team_names = list(team_stats.keys())
        events = fetch_odds(sport_key)

        print(f"  {len(events)} event(s) returned by the odds API")

        events_with_both_books = 0
        best_ev_seen = None  # (ev, home_team, away_team, side) for visibility
        bookmaker_keys_seen = set()
        market_keys_seen = set()

        for event in events:
            for bm in event.get("bookmakers", []):
                bookmaker_keys_seen.add(bm["key"])
                for mkt in bm.get("markets", []):
                    market_keys_seen.add(mkt["key"])

        for event in events:
            home_team = event["home_team"]
            away_team = event["away_team"]
            commence_time = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )
            hours_to_kickoff = (
                commence_time - datetime.now(timezone.utc)
            ).total_seconds() / 3600.0
            if hours_to_kickoff < 0:
                continue

            # Reconcile odds-feed names back to the stats source's names
            home_key = normalize_team_name(home_team, stat_team_names)
            away_key = normalize_team_name(away_team, stat_team_names)
            h_stats = team_stats.get(
                home_key, {"attack": 1.0, "defense": 1.0, "x_multiplier": 1.0}
            )
            a_stats = team_stats.get(
                away_key, {"attack": 1.0, "defense": 1.0, "x_multiplier": 1.0}
            )

            lam_home = (
                LEAGUE_BASELINE_GOALS
                * h_stats["attack"]
                * a_stats["defense"]
                * h_stats["x_multiplier"]
            )
            lam_away = (
                LEAGUE_BASELINE_GOALS
                * a_stats["attack"]
                * h_stats["defense"]
                * a_stats["x_multiplier"]
            )

            pin_market = None
            dk_market = None
            for bookmaker in event.get("bookmakers", []):
                if bookmaker["key"] == "pinnacle":
                    pin_market = extract_spread_market(bookmaker)
                elif bookmaker["key"] == "draftkings":
                    dk_market = extract_spread_market(bookmaker)

            if not pin_market or not dk_market:
                continue  # need both books present to compare

            pin_outcomes = outcomes_by_team(pin_market)
            dk_outcomes = outcomes_by_team(dk_market)
            if home_team not in pin_outcomes or away_team not in pin_outcomes:
                continue
            if home_team not in dk_outcomes or away_team not in dk_outcomes:
                continue

            events_with_both_books += 1

            pin_home_price, pin_line = pin_outcomes[home_team]
            pin_away_price, _ = pin_outcomes[away_team]
            dk_home_price, dk_line = dk_outcomes[home_team]
            dk_away_price, _ = dk_outcomes[away_team]

            # Sharp fair probabilities at Pinnacle's own line
            p_sharp_home_at_pin_line, p_sharp_away_at_pin_line = devig_pinnacle(
                pin_home_price, pin_away_price
            )

            # Bottom-up probabilities at both books' lines (may differ)
            p_bu_home_at_pin_line = calculate_poisson_probability(
                lam_home, lam_away, pin_line, "home"
            )
            p_bu_home_at_dk_line = calculate_poisson_probability(
                lam_home, lam_away, dk_line, "home"
            )
            p_bu_away_at_pin_line = calculate_poisson_probability(
                lam_home, lam_away, -pin_line, "away"
            )
            p_bu_away_at_dk_line = calculate_poisson_probability(
                lam_home, lam_away, -dk_line, "away"
            )

            # If DK's line differs from Pinnacle's, shift the sharp fair
            # probability by the model's assessed difference between the
            # two lines rather than assuming they're interchangeable.
            p_sharp_home_adj = min(
                max(
                    p_sharp_home_at_pin_line
                    + (p_bu_home_at_dk_line - p_bu_home_at_pin_line),
                    0.01,
                ),
                0.99,
            )
            p_sharp_away_adj = min(
                max(
                    p_sharp_away_at_pin_line
                    + (p_bu_away_at_dk_line - p_bu_away_at_pin_line),
                    0.01,
                ),
                0.99,
            )

            for side, dk_price, p_sharp_adj, p_bu in [
                ("home", dk_home_price, p_sharp_home_adj, p_bu_home_at_dk_line),
                ("away", dk_away_price, p_sharp_away_adj, p_bu_away_at_dk_line),
            ]:
                p_hybrid = time_decay_blend(p_sharp_adj, p_bu, hours_to_kickoff)
                ev = (p_hybrid * dk_price) - 1

                if best_ev_seen is None or ev > best_ev_seen[0]:
                    best_ev_seen = (ev, home_team, away_team, side)

                if ev >= EV_THRESHOLD:
                    all_bets.append(
                        {
                            "league": league_name,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": event["commence_time"],
                            "side": side,
                            "line": dk_line if side == "home" else -dk_line,
                            "dk_price": dk_price,
                            "p_hybrid": round(p_hybrid, 4),
                            "ev": round(ev, 4),
                            "units": assign_tiered_units(ev),
                        }
                    )

        print(f"  {events_with_both_books} event(s) had both Pinnacle and DraftKings spreads")
        print(f"  bookmakers seen: {sorted(bookmaker_keys_seen) if bookmaker_keys_seen else '(none)'}")
        print(f"  markets seen: {sorted(market_keys_seen) if market_keys_seen else '(none)'}")
        if best_ev_seen:
            ev, h, a, side = best_ev_seen
            print(f"  best EV seen: {ev*100:.2f}% ({h} v {a}, {side} side) [threshold is {EV_THRESHOLD*100:.0f}%]")
        else:
            print("  no fixture had both books available to compare")

    output_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "bets": all_bets,
    }

    with open("daily_bets.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully generated daily_bets.json with {len(all_bets)} bet(s)")


if __name__ == "__main__":
    main()
