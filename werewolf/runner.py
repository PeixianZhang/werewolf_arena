# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import traceback
from typing import List, Tuple
import itertools
import pandas as pd
import os
import datetime

from absl import flags
import tqdm

from werewolf import logging
from werewolf import game
from werewolf.model import Doctor
from werewolf.model import SEER
from werewolf.model import Seer
from werewolf.model import State
from werewolf.model import Villager
from werewolf.model import WEREWOLF
from werewolf.model import Werewolf
from werewolf.config import get_player_names, get_demographic, ANONYMOUS_MODE, ANONYMOUS_MODE

_RUN_GAME = flags.DEFINE_boolean("run", False, "Runs a single game.")
_RESUME = flags.DEFINE_boolean("resume", False, "Resumes games.")
_EVAL = flags.DEFINE_boolean("eval", False, "Collect eval data by running many games.")
_BATCH = flags.DEFINE_boolean("batch", False, "Run multiple games in batch mode.")
_NUM_GAMES = flags.DEFINE_integer(
    "num_games", 2, "Number of games to run used with eval or batch mode."
)
_VILLAGER_MODELS = flags.DEFINE_list(
    "v_models", "", "The model used for villagers values are: gpt4o, gpt4o, gpt4o"
)
_WEREWOLF_MODELS = flags.DEFINE_list(
    "w_models", "", "The model used for werewolves values are: gpt4o, gpt4o, gpt4o"
)
_ARENA = flags.DEFINE_boolean(
    "arena", False, "Only run games using different models for villagers and werewolves"
)
_THREADS = flags.DEFINE_integer("threads", 2, "Number of threads to run.")

DEFAULT_WEREWOLF_MODELS = ["flash", "flash"]
DEFAULT_VILLAGER_MODELS = ["flash", "flash"]
RESUME_DIRECTORIES = []

model_to_id = {
    "pro1.5": "gemini-1.5-pro-preview-0514",
    "flash": "gemini-2.5-flash",
    "pro1": "gemini-pro",
    "gpt5":"gpt-5",
    "gpt4o": "o4-mini",
}


def initialize_players(
    villager_model: str, werewolf_model: str
) -> Tuple[Seer, Doctor, List[Villager], List[Werewolf]]:
    """Assigns roles to players and initializes their game view."""

    player_names = get_player_names()
    random.shuffle(player_names)

    _sn = player_names.pop()
    seer = Seer(name=_sn, model=villager_model, demographic=get_demographic(_sn))
    _dn = player_names.pop()
    doctor = Doctor(name=_dn, model=villager_model, demographic=get_demographic(_dn))
    _w_names = [player_names.pop() for _ in range(2)]
    werewolves = [Werewolf(name=nm, model=werewolf_model, demographic=get_demographic(nm)) for nm in _w_names]
    villagers = [Villager(name=name, model=villager_model, demographic=get_demographic(name)) for name in player_names]

    all_players = [seer, doctor] + werewolves + villagers
    # 匿名模式下不显示 demographic 信息
    if ANONYMOUS_MODE:
        player_introductions = [f"- {p.name}" for p in all_players]
    else:
        player_introductions = [f"- {p.name}: {p.demographic}" for p in all_players]

    # Initialize game view for all players
    current_players_list = [seer.name, doctor.name] + [w.name for w in werewolves] + [v.name for v in villagers]
    for player in all_players:
        if isinstance(player, Werewolf):
            # Get all other werewolves as companions
            other_wolves = [w.name for w in werewolves if w != player]
        else:
            other_wolves = None
        tqdm.tqdm.write(f"{player.name} has role {player.role}")
        player.initialize_game_view(
            current_players=current_players_list,
            round_number=0,
            other_wolf=other_wolves[0] if other_wolves else None,  # Keep for backward compatibility
            player_introductions=player_introductions,
        )
        # Store all werewolf companions if player is a werewolf
        if isinstance(player, Werewolf) and hasattr(player.gamestate, 'other_wolves'):
            player.gamestate.other_wolves = other_wolves

    return seer, doctor, villagers, werewolves


def resume_game(directory: str) -> bool:
    state, logs = logging.load_game(directory)

    # remove the failed round and resume from the beginning of that round.
    last_round = state.rounds[-1]
    if not last_round.success:
        state.rounds.pop()
        logs.pop()
    # Reset the error state
    state.error_message = ""

    # 匿名模式下不显示 demographic 信息
    if ANONYMOUS_MODE:
        player_introductions = [
            f"- {name}"
            for name in state.players
        ]
    else:
        player_introductions = [
            f"- {name}: {state.players[name].demographic}"
            for name in state.players
        ]
    if not state.rounds:
        werewolves = []
        for p in state.players.values():
            p.initialize_game_view(
                round_number=0,
                current_players=list(state.players.keys()),
                player_introductions=player_introductions,
            )
            p.observations = []

            if p.role == WEREWOLF:
                werewolves.append(p)

            if p.role == SEER:
                p.previously_unmasked = {}

        # Set up werewolf companions
        for i, wolf in enumerate(werewolves):
            other_wolves = [w.name for w in werewolves if w != wolf]
            if wolf.gamestate:
                wolf.gamestate.other_wolf = other_wolves[0] if other_wolves else None
                wolf.gamestate.other_wolves = other_wolves
    else:
        # Update the GameView for every active player
        werewolves = []
        for p in state.rounds[-1].players:
            player = state.players.get(p, None)
            if player:
                player.initialize_game_view(
                    round_number=len(state.rounds),
                    current_players=state.rounds[-1].players[:],
                    player_introductions=player_introductions,
                )

                # Remove the observation from the failed round for all active players
                failed_round = len(state.rounds)
                player.observations = [
                    o
                    for o in player.observations
                    if not o.startswith(f"Round {failed_round}")
                ]

                if player.role == WEREWOLF:
                    werewolves.append(player)

                # update the seer's unmasking history
                unmasking_history = {}
                if player.role == SEER:
                    for r in state.rounds:
                        if r.unmasked:
                            unmasked_player = state.players.get(r.unmasked, None)
                            if unmasked_player:
                                unmasking_history[r.unmasked] = unmasked_player.role
                    player.previously_unmasked = unmasking_history

        # Set up werewolf companions
        for i, wolf in enumerate(werewolves):
            other_wolves = [w.name for w in werewolves if w != wolf]
            if wolf.gamestate:
                wolf.gamestate.other_wolf = other_wolves[0] if other_wolves else None
                wolf.gamestate.other_wolves = other_wolves

    gm = game.GameMaster(state, num_threads=_THREADS.value)
    gm.logs = logs
    try:
        gm.run_game()
    except Exception as e:
        state.error_message = traceback.format_exc()
    logging.save_game(state, gm.logs, directory)
    return not state.error_message


def resume_games(directories: list[str]):
    successful_resumes = []
    failed_resumes = []
    invalid_resumes = []
    for i in tqdm.tqdm(range(len(directories)), desc="Games"):
        d = directories[i]
        try:
            success = resume_game(d)
            if success:
                successful_resumes.append(d)
            else:
                failed_resumes.append(d)
        except Exception as e:
            if "not found" in str(e):
                invalid_resumes.append(d)
            print(f"Error encountered during resume: {e}")

    print(
        f"Successful resumes: {successful_resumes}.\nFailed resumes:"
        f" {failed_resumes}\nInvalid resumes(no partial game found):"
        f" {invalid_resumes}"
    )


def run_game(
    werewolf_model: str,
    villager_model: str,
) -> Tuple[str, str]:
    """Runs a single game of Werewolf.

    Returns: (winner, log_dir)
    """
    seer, doctor, villagers, werewolves = initialize_players(
        villager_model, werewolf_model
    )
    session_id = "10"  # You might want to make this unique per game
    state = State(
        villagers=villagers,
        werewolves=werewolves,
        seer=seer,
        doctor=doctor,
        session_id=session_id,
    )

    gamemaster = game.GameMaster(state, num_threads=_THREADS.value)
    winner = None
    try:
        winner = gamemaster.run_game()
    except Exception as e:
        state.error_message = traceback.format_exc()
        print(f"Error encountered during game: {e}")

    log_directory = logging.log_directory()
    logging.save_game(state, gamemaster.logs, log_directory)
    print(f"Game logs saved to: {log_directory}")

    return winner, log_directory


def run() -> None:
    # #region agent log
    import json
    try:
        with open(r'd:\Github_Clone\werewolf_arena\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"location":"runner.py:267","message":"run() entry","data":{"model_to_id_keys":list(model_to_id.keys()),"model_to_id":model_to_id},"runId":"initial","hypothesisId":"A,B,C,D"})+'\n')
    except:
        pass
    # #endregion
    villager_models = _VILLAGER_MODELS.value or DEFAULT_VILLAGER_MODELS
    werewolf_models = _WEREWOLF_MODELS.value or DEFAULT_WEREWOLF_MODELS
    # #region agent log
    try:
        with open(r'd:\Github_Clone\werewolf_arena\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"location":"runner.py:270","message":"Before model_to_id lookup","data":{"villager_models":villager_models,"werewolf_models":werewolf_models,"available_keys":list(model_to_id.keys())},"runId":"initial","hypothesisId":"A,B,C,D"})+'\n')
    except:
        pass
    # #endregion
    v_ids = [model_to_id[m] for m in villager_models]
    w_ids = [model_to_id[m] for m in werewolf_models]
    model_combinations = list(itertools.product(v_ids, w_ids))
    if _RUN_GAME.value:
        villager_model, werewolf_model = model_combinations[0]
        print(f"Villagers: {villager_model} versus Werwolves:  {werewolf_model}")
        run_game(
            werewolf_model=werewolf_model,
            villager_model=villager_model,
        )
    elif _EVAL.value:
        results = []
        for villager_model, werewolf_model in model_combinations:
            # only run games using different models in the arena mode
            if villager_model == werewolf_model and _ARENA.value:
                continue
            print(
                f"Running games with Villagers: {villager_model} and"
                f" Werewolves:{werewolf_model}"
            )
            for _ in tqdm.tqdm(range(_NUM_GAMES.value), desc="Games"):
                winner, log_dir = run_game(
                    werewolf_model=werewolf_model,
                    villager_model=villager_model,
                )
                results.append([villager_model, werewolf_model, winner, log_dir])

        df = pd.DataFrame(
            results, columns=["VillagerModel", "WerewolfModel", "Winner", "Log"]
        )
        print("######## Eval results ########")
        print(df)

        pacific_timezone = datetime.timezone(datetime.timedelta(hours=-8))
        timestamp = datetime.datetime.now(pacific_timezone).strftime("%Y%m%d_%H%M%S")
        csv_file = f"{os.getcwd()}/logs/eval_results_{timestamp}.csv"
        df.to_csv(csv_file)
        print(f"Wrote eval results to {csv_file}")

    elif _BATCH.value:
        # Batch mode: run multiple games with the same model combination
        villager_model, werewolf_model = model_combinations[0]
        num_games = _NUM_GAMES.value
        print(f"Batch mode: Running {num_games} games")
        print(f"Villagers: {villager_model} versus Werewolves: {werewolf_model}")
        
        results = []
        for i in tqdm.tqdm(range(num_games), desc="Batch Games"):
            winner, log_dir = run_game(
                werewolf_model=werewolf_model,
                villager_model=villager_model,
            )
            results.append({
                "game_number": i + 1,
                "villager_model": villager_model,
                "werewolf_model": werewolf_model,
                "winner": winner,
                "log_directory": log_dir
            })
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch Run Summary:")
        print("=" * 60)
        winner_counts = {}
        for result in results:
            winner = result["winner"] or "Unknown"
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        
        for winner, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_games) * 100
            print(f"{winner}: {count} games ({percentage:.1f}%)")
        
        print(f"\nTotal games: {num_games}")
        print("=" * 60)
        
    elif _RESUME.value:
        resume_games(RESUME_DIRECTORIES)
    else:
        # Default behavior: run a single game if no flag is set
        villager_model, werewolf_model = model_combinations[0]
        print(f"Villagers: {villager_model} versus Werewolves: {werewolf_model}")
        run_game(
            werewolf_model=werewolf_model,
            villager_model=villager_model,
        )
