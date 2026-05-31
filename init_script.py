### download data for all seasons and save to CSV files


from datetime import date
import os
import time
import random
import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashTeamStats,
    LeagueGameFinder,
    LeagueDashPlayerStats,
 )

output_dir = "/Users/donma/Code/NBA-Analytics/data"
os.makedirs(output_dir, exist_ok=True)

seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2010, 2026)]
seasons[:3], seasons[-1]

def fetch_df(endpoint_cls, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            endpoint = endpoint_cls(timeout=60, **kwargs)
            return endpoint.get_data_frames()[0]
        except Exception as exc:
            last_error = exc
            wait = 2 + (attempt * 4) + random.uniform(0.5, 1.5)
            print(f"retry {attempt + 1}/4 for {endpoint_cls.__name__}: {exc} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise last_error

season_types = ["Regular Season", "Playoffs"]

for season in seasons:
    final_teams_df = None
    final_games_df = None
    final_player_stats_df = None
    for season_type in season_types:
        print(f"Downloading season {season} ({season_type})...")
        teams_df = fetch_df(
            LeagueDashTeamStats,
            season=season,
            season_type_all_star=season_type,
            last_n_games=0,
            measure_type_detailed_defense="Base",
            month=0,
            opponent_team_id=0,
            pace_adjust="N",
            per_mode_detailed="Totals",
            period=0,
            plus_minus="Y",
            rank="N",
        )
        player_stats_df = fetch_df(
            LeagueDashPlayerStats,
            season=season,
            season_type_all_star=season_type,
            last_n_games=0,
            measure_type_detailed_defense="Base",
            month=0,
            opponent_team_id=0,
            pace_adjust="N",
            per_mode_detailed="PerGame",
            period=0,
            plus_minus="Y",
            rank="N",
          
        )
        games_df = fetch_df(
            LeagueGameFinder,
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",
        )

        teams_df["season"] = season
        teams_df["season_type"] = season_type
        teams_df["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")
        player_stats_df["season"] = season
        player_stats_df["season_type"] = season_type
        player_stats_df["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")
        games_df["season"] = season
        games_df["season_type"] = season_type
        games_df["ingestion_date"] = datetime.now().strftime("%Y-%m-%d")
        season_type_slug = season_type.lower().replace(" ", "_")
        if final_teams_df is None:
            final_teams_df = teams_df
        else:
            final_teams_df = pd.concat([final_teams_df, teams_df], ignore_index=True)

        if final_player_stats_df is None:
            final_player_stats_df = player_stats_df
        else:
            final_player_stats_df = pd.concat([final_player_stats_df, player_stats_df], ignore_index=True)
        if final_games_df is None:
            final_games_df = games_df
        else:
            final_games_df = pd.concat([final_games_df, games_df], ignore_index=True)
    filename_today = datetime.today().strftime("%Y_%m_%d.csv")
    teams_path = os.path.join(output_dir, f"teams_{season}_{filename_today}")
    player_stats_path = os.path.join(output_dir, f"player_stats_{season}_{filename_today}")
    games_path = os.path.join(output_dir, f"games_{season}_{filename_today}")

    final_teams_df.to_csv(teams_path, index=False)
    final_player_stats_df.to_csv(player_stats_path, index=False)
    final_games_df.to_csv(games_path, index=False)
