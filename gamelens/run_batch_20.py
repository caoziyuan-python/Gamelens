# -*- coding: utf-8 -*-
"""Run overnight batch analysis for 20 selected games."""
from __future__ import annotations

from datetime import datetime

from insights.engine import run_analysis_pipeline
from utils.cache import get_or_fetch, save_game_result

SELECTED_GAMES = [
    ("Maylee: American Mahjong", "6758261607"),
    ("Match Mahjong: Tile Game", "6742874143"),
    ("Wonder Villa - Tile Match Game", "6740087353"),
    ("Mahjong Solitaire - Master", "1569177946"),
    ("Koi Mahjong: Solitaire Game", "6748625842"),
    ("Tile Home: Triple Match", "6480406974"),
    ("Arcadia Zen Math: Number Games", "6743825175"),
    ("Tile Match -Triple puzzle game", "6448493522"),
    ("Tile Burst - Match Puzzle", "6741840721"),
    ("Mahjong Club - Solitaire Game", "1565374299"),
    ("Match Tile Scenery", "1595779374"),
    ("Mahjong Solitaire Card Games", "6504851426"),
    ("Arcadia Mahjong", "6503700976"),
    ("Triple Match 3D", "1607122287"),
    ("Tile Match 3D : Triple Match", "1578204014"),
    ("Tile Scenery: Match Puzzle", "6736857648"),
    ("Tile Explorer: Tiles Clear!", "6498883328"),
    ("Tile Match Story", "6475805107"),
    ("Jigmatch - Solitaire Puzzle", "6754753312"),
    ("Vita Mahjong", "6468921495"),
]


def main() -> None:
    total = len(SELECTED_GAMES)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Start batch for {total} games")

    failed: list[str] = []
    for idx, (game_name, app_id) in enumerate(SELECTED_GAMES, start=1):
        print("-" * 80)
        print(f"[{idx}/{total}] {game_name} ({app_id})")
        try:
            reviews = get_or_fetch(app_id, game_name)
            print(f"  reviews={len(reviews)}")

            result = run_analysis_pipeline(
                reviews,
                game_name,
                run_level2=False,
                enable_llm=True,
            )
            result["game_name"] = game_name

            save_game_result(
                game_name=game_name,
                result=result,
                reviews=reviews,
                analysis_mode="full",
                run_level2=False,
            )
            print("  status=ok saved=full_l2off")
        except Exception as exc:
            failed.append(game_name)
            print(f"  status=failed error={exc}")

    print("=" * 80)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Done total={total} failed={len(failed)}")
    if failed:
        print("Failed games:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
