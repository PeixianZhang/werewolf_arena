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

"""Werewolf game."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import random
from typing import List
import json
import os
from datetime import datetime

import tqdm

from werewolf.lm import LmLog, generate
from werewolf.model import Round, RoundLog, State, VoteLog
from werewolf.config import  MAX_DEBATE_TURNS, RUN_SYNTHETIC_VOTES


class GameMaster:

  def __init__(
      self,
      state: State,
      num_threads: int = 1,
  ) -> None:
    """Initialize the Werewolf game.

    Args:
    """
    self.state = state
    self.current_round_num = len(self.state.rounds) if self.state.rounds else 0
    self.num_threads = num_threads
    self.logs: List[RoundLog] = []
    # Store initial player order for determining speaking order
    self.initial_player_order: List[str] = list(state.players.keys())

  @property
  def this_round(self) -> Round:
    return self.state.rounds[self.current_round_num]

  @property
  def this_round_log(self) -> RoundLog:
    return self.logs[self.current_round_num]

  def eliminate(self):
    """Werewolves choose a player to eliminate. Ensures eliminated is always a valid player in this_round.players."""
    werewolves_alive = [
        w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    valid_targets = [
        p for p in self.this_round.players
        if p not in {w.name for w in werewolves_alive}
    ]
    if not valid_targets:
      self.this_round.eliminated = None
      tqdm.tqdm.write("No valid target to eliminate (only werewolves left).")
      return

    wolf = random.choice(werewolves_alive)
    eliminated, log = wolf.eliminate()
    self.this_round_log.eliminate = log

    if eliminated is None or eliminated not in valid_targets:
      eliminated = random.choice(valid_targets)
    self.this_round.eliminated = eliminated
    tqdm.tqdm.write(f"{wolf.name} eliminated {eliminated}")
    
    # Record werewolf target decision in private memory
    for w in werewolves_alive:
      w.memory_manager.add_werewolf_target(eliminated)

  def protect(self):
    """Doctor chooses a player to protect."""
    if self.state.doctor.name not in self.this_round.players:
      return  # Doctor no longer in the game

    protect, log = self.state.doctor.save()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      tqdm.tqdm.write(f"{self.state.doctor.name} protected {protect}")
      # Record doctor protection in private memory
      self.state.doctor.memory_manager.add_doctor_protection(protect)
    else:
      raise ValueError("Protect did not return a valid player.")

  def unmask(self):
    """Seer chooses a player to unmask."""
    if self.state.seer.name not in self.this_round.players:
      return  # Seer no longer in the game

    unmask, log = self.state.seer.unmask()
    self.this_round_log.investigate = log

    if unmask is not None:
      self.this_round.unmasked = unmask
      unmasked_player = self.state.players[unmask]
      self.state.seer.reveal_and_update(unmask, unmasked_player.role)
      # Record seer verification in private memory
      # Seer sees "Werewolf" or "Villager" side, not exact role
      side = "Werewolf" if unmasked_player.role == "Werewolf" else "Villager"
      self.state.seer.memory_manager.add_seer_verification(unmask, side)
    else:
      raise ValueError("Unmask function did not return a valid player.")

  def run_summaries(self):
    """Collect summaries from players after the debate."""

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_summaries = {
          name: executor.submit(self.state.players[name].summarize)
          for name in self.this_round.players
      }

      for player_name, summary_task in player_summaries.items():
        summary, log = summary_task.result()
        tqdm.tqdm.write(f"{player_name} summary: {summary}")
        self.this_round_log.summaries.append((player_name, log))

  def _log_bias_reflection(self, player_name: str, role: str, round_num: int, 
                          turn_idx: int, bias_self_report: dict, reflection_summary: str):
    """Log bias self-report from reflection to game_analysis_log.jsonl file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "game_session": self.state.session_id,
        "round": round_num,
        "turn": turn_idx,
        "player": player_name,
        "role": role,
        "type": "bias_reflection",
        "reflection_summary": reflection_summary,
        "bias_self_report": bias_self_report
    }
    
    log_file = "game_analysis_log.jsonl"
    try:
      with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
      tqdm.tqdm.write(f"Warning: Failed to write bias reflection log: {e}")

  def _get_starting_speaker(self) -> str:
    """Determine the starting speaker for the day phase.
    
    Returns the player after the eliminated player, or the first player
    if no one was eliminated during the night.
    """
    eliminated_player = self.this_round.eliminated
    
    # If no one was eliminated, start from the first player
    if not eliminated_player or eliminated_player not in self.initial_player_order:
      # Find the first player in initial order who is still alive
      for player_name in self.initial_player_order:
        if player_name in self.this_round.players:
          return player_name
      # Fallback: return first player in current players list
      return self.this_round.players[0] if self.this_round.players else None
    
    # Find the eliminated player's position in initial order
    try:
      eliminated_idx = self.initial_player_order.index(eliminated_player)
    except ValueError:
      # Eliminated player not in initial order (shouldn't happen), fallback to first
      return self.this_round.players[0] if self.this_round.players else None
    
    # Find the next player after eliminated player who is still alive
    for i in range(eliminated_idx + 1, len(self.initial_player_order)):
      next_player = self.initial_player_order[i]
      if next_player in self.this_round.players:
        return next_player
    
    # Wrap around: start from beginning if eliminated was last
    for i in range(eliminated_idx):
      next_player = self.initial_player_order[i]
      if next_player in self.this_round.players:
        return next_player
    
    # Fallback: return first player in current players list
    return self.this_round.players[0] if self.this_round.players else None

  def run_day_phase(self):
    """Run the day phase which consists of the debate and voting.
    
    Each player speaks once in order, starting from the player after
    the eliminated player (or first player if no one was eliminated).
    """
    # Add day timestamp to all players' memory
    for name in self.this_round.players:
      player = self.state.players[name]
      if player.memory_manager:
        player.memory_manager.add_day_timestamp(self.current_round_num)
    
    # Determine starting speaker
    starting_speaker = self._get_starting_speaker()
    if not starting_speaker:
      raise ValueError("No valid starting speaker found.")
    
    # Get the order of players starting from starting_speaker
    starting_idx = self.initial_player_order.index(starting_speaker)
    speaking_order = []
    
    # Add players from starting position to end
    for i in range(starting_idx, len(self.initial_player_order)):
      player_name = self.initial_player_order[i]
      if player_name in self.this_round.players:
        speaking_order.append(player_name)
    
    # Wrap around: add players from beginning to starting position
    for i in range(starting_idx):
      player_name = self.initial_player_order[i]
      if player_name in self.this_round.players:
        speaking_order.append(player_name)
    
    # Each player speaks once
    for idx, next_speaker in enumerate(speaking_order):
      player = self.state.players[next_speaker]
      
      # Trigger reflection before speaking
      if player.memory_manager and player.gamestate:
        reflection_result = player.memory_manager.reflect(
            player_role=player.role,
            current_round=self.current_round_num,
            remaining_players=self.this_round.players,
            model=player.model,
            generate_func=generate
        )
        
        # Log bias self-report to game_analysis_log.jsonl
        if reflection_result and reflection_result.get("bias_self_report"):
          bias_report = reflection_result["bias_self_report"]
          self._log_bias_reflection(
              player_name=next_speaker,
              role=player.role,
              round_num=self.current_round_num,
              turn_idx=idx,
              bias_self_report=bias_report,
              reflection_summary=reflection_result.get("reflection_summary", "")
          )
      
      dialogue, log = player.debate()
      if dialogue is None:
        raise ValueError(
            f"{next_speaker} did not return a valid dialouge from debate()."
        )

      self.this_round_log.debate.append((next_speaker, log))
      self.this_round.debate.append([next_speaker, dialogue])
      tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

      # Update game state and memory for all players
      for name in self.this_round.players:
        p = self.state.players[name]
        if p.gamestate:
          p.gamestate.update_debate(next_speaker, dialogue)
          # Update memory manager for all players (they all hear the dialogue)
          p.memory_manager.add_dialogue(next_speaker, dialogue)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")

    # After all players have spoken, run deductions and voting
    self.run_deductions()
    votes, vote_logs = self.run_voting()
    self.this_round.votes.append(votes)
    self.this_round_log.votes.append(vote_logs)

    for player, vote in self.this_round.votes[-1].items():
      tqdm.tqdm.write(f"{player} voted to remove {vote}")

  def run_deductions(self):
    """Before voting: each player reflects on hidden roles of others; log role/reasoning/confidence/evidence."""
    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      deduction_tasks = {
          name: executor.submit(self.state.players[name].reflect_on_roles)
          for name in self.this_round.players
      }
      for player_name, task in deduction_tasks.items():
        try:
          _result, log = task.result()
          self.this_round_log.deductions.append((player_name, log))
          tqdm.tqdm.write(f"{player_name} deduction logged.")
        except Exception as e:
          tqdm.tqdm.write(f"{player_name} deduction failed: {e}")
          self.this_round_log.deductions.append(
              (player_name, self._empty_deduction_log(player_name))
          )

  def _empty_deduction_log(self, player_name: str) -> LmLog:
    return LmLog(prompt="", raw_resp="", result={"deductions": []})

  def run_voting(self):
    """Conduct a vote among players to exile someone."""
    vote_log = []
    votes = {}

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_votes = {
          name: executor.submit(self.state.players[name].vote)
          for name in self.this_round.players
      }

      for player_name, vote_task in player_votes.items():
        vote, log = vote_task.result()
        vote_log.append(VoteLog(player_name, vote, log))

        if vote is not None:
          votes[player_name] = vote
        else:
          self.this_round.votes.append(votes)
          self.this_round_log.votes.append(vote_log)
          raise ValueError(f"{player_name} vote did not return a valid player.")

    return votes, vote_log

  def exile(self):
    """Exile the player who received the most votes."""

    most_voted, vote_count = Counter(
        self.this_round.votes[-1].values()
    ).most_common(1)[0]

    if vote_count > len(self.this_round.players) / 2:
      self.this_round.exiled = most_voted

    # Record vote records in all players' memory (before exile)
    votes_dict = self.this_round.votes[-1]
    for name in self.this_round.players:
      player = self.state.players[name]
      if player.memory_manager:
        player.memory_manager.add_vote_records(votes_dict)

    if self.this_round.exiled is not None:
      exiled_player = self.this_round.exiled
      if exiled_player in self.this_round.players:
        self.this_round.players.remove(exiled_player)
      announcement = (
          f"The majority voted to remove {exiled_player} from the game."
      )
      # Record death report for exiled player
      for name in self.this_round.players:
        player = self.state.players[name]
        if player.memory_manager:
          player.memory_manager.add_death_report(exiled_player, cause="vote")
          player.memory_manager.update_player_status(exiled_player, False)
    else:
      announcement = (
          "A majority vote was not reached, so no one was removed from the"
          " game."
      )

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate and self.this_round.exiled is not None:
        player.gamestate.remove_player(self.this_round.exiled)
      player.add_announcement(announcement)

    tqdm.tqdm.write(announcement)

  def resolve_night_phase(self):
    """Resolve elimination and protection during the night phase."""
    if self.this_round.eliminated != self.this_round.protected:
      eliminated_player = self.this_round.eliminated
      if eliminated_player and eliminated_player in self.this_round.players:
        self.this_round.players.remove(eliminated_player)
      announcement = (
          f"The Werewolves removed {eliminated_player} from the game during the"
          " night."
      )
      # Record death report in all players' memory
      for name in self.this_round.players:
        player = self.state.players[name]
        if player.memory_manager:
          player.memory_manager.add_death_report(eliminated_player, cause="night")
          player.memory_manager.update_player_status(eliminated_player, False)
    else:
      announcement = "No one was removed from the game during the night."
    tqdm.tqdm.write(announcement)

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate and self.this_round.eliminated != self.this_round.protected and self.this_round.eliminated is not None:
        player.gamestate.remove_player(self.this_round.eliminated)
      player.add_announcement(announcement)

  def run_round(self):
    """Run a single round of the game. this_round.players is the single source of truth; only exile() and resolve_night_phase() remove from it."""
    self.state.rounds.append(Round())
    self.logs.append(RoundLog())

    if self.current_round_num == 0:
      self.this_round.players = list(self.state.players.keys())
      # Store initial player order on first round
      self.initial_player_order = list(self.state.players.keys())
      # Initialize all players as alive in belief_matrix
      for player_name in self.this_round.players:
        for player in self.state.players.values():
          if player.memory_manager:
            player.memory_manager.update_player_status(player_name, True)
    else:
      self.this_round.players = list(
          self.state.rounds[self.current_round_num - 1].players
      )
      # Ensure initial_player_order is set (for resume games)
      if not hasattr(self, 'initial_player_order') or not self.initial_player_order:
        # Try to reconstruct from first round if available
        if self.state.rounds and len(self.state.rounds) > 0:
          # Use all players from state (including eliminated ones)
          self.initial_player_order = list(self.state.players.keys())
        else:
          self.initial_player_order = list(self.state.players.keys())

    for action, message in [
        (
            self.eliminate,
            "The Werewolves are picking someone to remove from the game.",
        ),
        (self.protect, "The Doctor is protecting someone."),
        (self.unmask, "The Seer is investigating someone."),
        (self.resolve_night_phase, ""),
        (self.check_for_winner, "Checking for a winner after Night Phase."),
        (self.run_day_phase, "The Players are debating and voting."),
        (self.exile, ""),
        (self.check_for_winner, "Checking for a winner after Day Phase."),
        (self.run_summaries, "The Players are summarizing the debate."),
    ]:
      tqdm.tqdm.write(message)
      action()

      if self.state.winner:
        tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
        self.this_round.success = True
        return

    tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
    self.this_round.success = True

  def get_winner(self) -> str:
    """Determine the winner of the game."""
    active_wolves = set(self.this_round.players) & set(
        w.name for w in self.state.werewolves
    )
    active_villagers = set(self.this_round.players) - active_wolves
    if len(active_wolves) >= len(active_villagers):
      return "Werewolves"
    return "Villagers" if not active_wolves else ""

  def check_for_winner(self):
    """Check if there is a winner and update the state accordingly."""
    self.state.winner = self.get_winner()
    if self.state.winner:
      tqdm.tqdm.write(f"The winner is {self.state.winner}!")

  def run_game(self) -> str:
    """Run the entire Werewolf game and return the winner."""
    while not self.state.winner:
      tqdm.tqdm.write(f"STARTING ROUND: {self.current_round_num}")
      self.run_round()
      for name in self.this_round.players:
        if self.state.players[name].gamestate:
          self.state.players[name].gamestate.round_number = (
              self.current_round_num + 1
          )
          self.state.players[name].gamestate.clear_debate()
          # Clear short-term memory at end of round
          self.state.players[name].memory_manager.clear_short_term()
      self.current_round_num += 1

    tqdm.tqdm.write("Game is complete!")
    return self.state.winner
