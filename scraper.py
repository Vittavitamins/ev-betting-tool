import os
import json
import math
import time
import difflib
from datetime import datetime, timezone

import requests
import pandas as pd
from scipy.stats import poisson

# ============================================================
# Configuration
# ============================================================
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_API_KEY_HERE")
SHARPAPI_KEY = os.environ.get("SHARPAPI_KEY", "YOUR_SHARPAPI_KEY_HERE")

LEAGUES = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_uefa_champs_league": "Champions League",
}

# SharpAPI uses its own league slugs, separate from The Odds API's sport
# keys above. Champions League isn't included here since SharpAPI's slug
# for it wasn't confirmed during testing -- add it once you've verified
# the exact slug via GET /api/v1/leagues.
SHARPAPI_LEAGUES = {
    "Premier League": "england_-_premier_league",
    "La Liga": "spain_-_la_liga",
    "Bundesliga": "germany_-_bundesliga",
    "Serie A": "italy_-_serie_a",
}

SHARPAPI_RATE_LIMIT_SLEEP = 5.5  # seconds between calls; free tier is 12 req/min

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
EV_SANITY_CEILING = 0.15     # EVs above this almost always mean a data/matching
                              # problem, not a real edge -- flagged and excluded
                              # rather than trusted at face value
LEAGUE_BASELINE_GOALS = 1.45  # average goals per team per match, fallback

# Preference order for the "soft" side of the comparison. The script tries
# each of these in order per fixture and uses the first one that's actually
# posting a spreads line for that game -- DraftKings does post soccer
# Asian handicap lines, but not for every fixture this far out, so this
# keeps the tool producing output on regulated US books rather than
# requiring DK specifically on every single game.
TARGET_BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers"]

# football-data.co.uk uses short/abbreviated club names; that's the space
# every incoming name needs to reconcile into, since it's what
# fetch_historical_stats() keys team_stats by. Different odds providers use
# different naming (The Odds API tends to use full official names; SharpAPI
# uses its own short forms that don't always match football-data's). Every
# alias below maps a known alternate spelling *to* football-data's short
# form -- never the other way -- so it works the same regardless of which
# provider's name comes in. This is a starter map, not a complete one --
# extend it as you hit misses (the script logs unmapped names so you can
# find them).
TEAM_ALIASES = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Real Sociedad": "Sociedad",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Betis": "Betis",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves",
    "Celta Vigo": "Celta",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "1. FC Koln": "FC Koln",
    "Bayer Leverkusen": "Leverkusen",
    "TSG Hoffenheim": "Hoffenheim",
    "VfL Wolfsburg": "Wolfsburg",
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "Hellas Verona": "Verona",
}


# ============================================================
# Team name reconciliation
# ============================================================
def normalize_team_name(name, candidates):
    """Maps an incoming team name (from any odds provider) to football-data
    .co.uk's canonical short-form name, since that's what team_stats is
    keyed by. Tries the explicit alias table first, then falls back to
    fuzzy string matching against football-data's own names."""
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
    """Fetches live odds from The Odds API. Used only for Pinnacle now --
    the soft-book (DraftKings/FanDuel) side comes from SharpAPI instead,
    since The Odds API doesn't return spreads from those books for soccer."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "spreads",
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code == 200:
        return response.json()
    print(f"API Error ({sport_key}): {response.status_code} - {response.text}")
    return []


def fetch_sharpapi_point_spreads(league_slug):
    """Fetches all point_spread rows for a league from SharpAPI (free tier:
    DraftKings + FanDuel only), paginating through results and respecting
    the free-tier rate limit."""
    url = "https://api.sharpapi.io/api/v1/odds"
    headers = {"X-API-Key": SHARPAPI_KEY}
    all_rows = []
    offset = 0
    limit = 50

    while True:
        params = {
            "sport": "soccer",
            "league": league_slug,
            "market": "point_spread",
            "limit": limit,
            "offset": offset,
        }
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            print(f"SharpAPI Error ({league_slug}): {response.status_code} - {response.text}")
            break

        payload = response.json()
        all_rows.extend(payload.get("data", []))

        pagination = payload.get("pagination", {})
        if not pagination.get("has_more"):
            break
        offset = pagination.get("next_offset", offset + limit)
        time.sleep(SHARPAPI_RATE_LIMIT_SLEEP)

    return all_rows


def build_sharpapi_lookup(rows, stat_team_names):
    """Groups raw SharpAPI point_spread rows into one entry per fixture,
    keyed by (normalized_home_team, normalized_away_team) using the same
    name-normalization used for the bottom-up stats, so fixtures line up
    with events from The Odds API regardless of each source's own naming.
    Only uses the main line, and prefers DraftKings over FanDuel per
    TARGET_BOOKS order when both are present for a fixture."""
    events = {}
    for row in rows:
        if not row.get("is_main_line"):
            continue
        event_id = row["event_id"]
        events.setdefault(event_id, {"home_team": row["home_team"], "away_team": row["away_team"],
                                      "event_start_time": row["event_start_time"], "rows": []})
        events[event_id]["rows"].append(row)

    lookup = {}
    for event in events.values():
        # Pick the preferred book among whichever posted the main line
        books_present = {r["sportsbook"] for r in event["rows"]}
        chosen_book = next((b for b in TARGET_BOOKS if b in books_present), None)
        if not chosen_book:
            continue

        home_row = next(
            (r for r in event["rows"] if r["sportsbook"] == chosen_book and r["team_side"] == "home"),
            None,
        )
        away_row = next(
            (r for r in event["rows"] if r["sportsbook"] == chosen_book and r["team_side"] == "away"),
            None,
        )
        if not home_row or not away_row:
            continue

        home_key = normalize_team_name(event["home_team"], stat_team_names)
        away_key = normalize_team_name(event["away_team"], stat_team_names)

        lookup[(home_key, away_key)] = {
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "event_start_time": event["event_start_time"],
            "home_price": home_row["odds_decimal"],
            "home_line": home_row["line"],
            "away_price": away_row["odds_decimal"],
            "away_line": away_row["line"],
            "book": chosen_book,
        }
    return lookup


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
    toward the sharp market as kickoff approaches. Leans heavily on the
    sharp market even far from kickoff (70% at 96+ hours out) since the
    bottom-up model's current-season-only ratings are noisy early in a
    season with only a handful of games played -- shifts to 95% sharp /
    5% bottom-up right at kickoff, same as before."""
    max_hours = 96.0
    weight_sharp = 0.95 - (min(hours_to_kickoff, max_hours) / max_hours) * 0.25
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

        sharpapi_league_slug = SHARPAPI_LEAGUES.get(league_name)
        if sharpapi_league_slug:
            sharpapi_rows = fetch_sharpapi_point_spreads(sharpapi_league_slug)
            sharpapi_lookup = build_sharpapi_lookup(sharpapi_rows, stat_team_names)
            time.sleep(SHARPAPI_RATE_LIMIT_SLEEP)  # stay under free-tier rate limit
        else:
            sharpapi_lookup = {}
            print(f"  no SharpAPI league slug configured for {league_name}, skipping soft-book side")

        print(f"  {len(sharpapi_lookup)} fixture(s) with a DraftKings/FanDuel main line from SharpAPI")

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
            for bookmaker in event.get("bookmakers", []):
                if bookmaker["key"] == "pinnacle":
                    pin_market = extract_spread_market(bookmaker)
                    break

            # Match this fixture to a SharpAPI quote via the same normalized
            # team-name keys used for the bottom-up stats lookup above.
            soft_quote = sharpapi_lookup.get((home_key, away_key))

            if not pin_market or not soft_quote:
                continue  # need Pinnacle plus a SharpAPI soft-book quote to compare

            pin_outcomes = outcomes_by_team(pin_market)
            if home_team not in pin_outcomes or away_team not in pin_outcomes:
                continue

            events_with_both_books += 1

            pin_home_price, pin_line = pin_outcomes[home_team]
            pin_away_price, _ = pin_outcomes[away_team]
            target_book_key = soft_quote["book"]
            dk_home_price, dk_line = soft_quote["home_price"], soft_quote["home_line"]
            dk_away_price, dk_away_line = soft_quote["away_price"], soft_quote["away_line"]

            # Skip fixtures where either team has no real rating data. This
            # usually means a cup tie or a team from a different division
            # got tagged under this league by one of the odds feeds -- the
            # bottom-up model would otherwise silently treat an unknown team
            # as league-average, which can produce wildly wrong probabilities
            # when blended against a correctly-priced sharp line.
            if home_key not in team_stats or away_key not in team_stats:
                print(f"  skipping {home_team} v {away_team}: team not in {league_name} stats (likely a cup/cross-division fixture)")
                continue

            # Sharp fair probabilities at Pinnacle's own line
            p_sharp_home_at_pin_line, p_sharp_away_at_pin_line = devig_pinnacle(
                pin_home_price, pin_away_price
            )

            # Bottom-up probabilities at both books' lines (may differ).
            # calculate_poisson_probability() always takes a HOME-oriented
            # line (e.g. -0.5 means home favored by half a goal) for both
            # sides -- it internally handles which side that favors. Passing
            # a negated line for "away" here was the bug: it double-flipped
            # the win condition and made every away-side probability wrong.
            p_bu_home_at_pin_line = calculate_poisson_probability(
                lam_home, lam_away, pin_line, "home"
            )
            p_bu_home_at_dk_line = calculate_poisson_probability(
                lam_home, lam_away, dk_line, "home"
            )
            p_bu_away_at_pin_line = calculate_poisson_probability(
                lam_home, lam_away, pin_line, "away"
            )
            p_bu_away_at_dk_line = calculate_poisson_probability(
                lam_home, lam_away, -dk_away_line, "away"
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

                if ev > EV_SANITY_CEILING:
                    print(f"  flagged and excluded: {home_team} v {away_team} ({side}) showed {ev*100:.1f}% EV -- above the sanity ceiling, likely a data issue, not a real edge")
                    continue

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
                            "line": dk_line if side == "home" else dk_away_line,
                            "book": target_book_key,
                            "price": dk_price,
                            "p_hybrid": round(p_hybrid, 4),
                            "ev": round(ev, 4),
                            "units": assign_tiered_units(ev),
                        }
                    )

        print(f"  {events_with_both_books} event(s) had both Pinnacle and a SharpAPI soft-book quote")
        print(f"  Odds API bookmakers seen: {sorted(bookmaker_keys_seen) if bookmaker_keys_seen else '(none)'}")
        print(f"  Odds API markets seen: {sorted(market_keys_seen) if market_keys_seen else '(none)'}")
        if best_ev_seen:
            ev, h, a, side = best_ev_seen
            print(f"  best EV seen: {ev*100:.2f}% ({h} v {a}, {side} side) [threshold is {EV_THRESHOLD*100:.0f}%]")
        else:
            print("  no fixture had Pinnacle plus a SharpAPI soft-book quote to compare")

    output_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "bets": all_bets,
    }

    with open("daily_bets.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully generated daily_bets.json with {len(all_bets)} bet(s)")


if __name__ == "__main__":
    main()
