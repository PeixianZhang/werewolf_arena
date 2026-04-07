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
import re
from typing import List

import tqdm

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
    self.round_start_players: List[str] = []

  @property
  def this_round(self) -> Round:
    return self.state.rounds[self.current_round_num]

  @property
  def this_round_log(self) -> RoundLog:
    return self.logs[self.current_round_num]

  def _resolve_wolf_decision(
      self,
      werewolves_alive,
      legal_targets,
      wolf_votes,
      wolf_logs,
  ):
    """Resolve wolf target without random tie-breaking."""
    valid_votes = {
        name: target
        for name, target in wolf_votes.items()
        if target in legal_targets
    }
    if len(valid_votes) == 1:
      return next(iter(valid_votes.items())), wolf_logs
    if len(valid_votes) >= 2:
      vote_counter = Counter(valid_votes.values())
      max_votes = max(vote_counter.values())
      top_targets = [
          target for target, count in vote_counter.items() if count == max_votes
      ]
      if len(top_targets) == 1:
        decided_target = top_targets[0]
        decided_wolf = next(
            name for name, target in valid_votes.items() if target == decided_target
        )
        return (decided_wolf, decided_target), wolf_logs

      # One extra decision round among tied targets.
      runoff_votes = {}
      runoff_logs = {}
      for wolf in werewolves_alive:
        runoff_target, runoff_log = wolf.eliminate(candidate_options=top_targets)
        runoff_votes[wolf.name] = runoff_target
        runoff_logs[wolf.name] = runoff_log

      runoff_valid = {
          name: target
          for name, target in runoff_votes.items()
          if target in top_targets
      }
      if runoff_valid:
        runoff_counter = Counter(runoff_valid.values())
        runoff_max = max(runoff_counter.values())
        runoff_top = [
            target
            for target, count in runoff_counter.items()
            if count == runoff_max
        ]
        if len(runoff_top) == 1:
          decided_target = runoff_top[0]
          decided_wolf = next(
              name
              for name, target in runoff_valid.items()
              if target == decided_target
          )
          return (decided_wolf, decided_target), runoff_logs

        # Still tied: deterministic fallback (seat order at round start).
        for name in self.round_start_players:
          if name in runoff_top:
            decided_target = name
            break
        else:
          decided_target = sorted(runoff_top)[0]
        decided_wolf = werewolves_alive[0].name
        return (decided_wolf, decided_target), runoff_logs

    return (None, None), wolf_logs

  def eliminate(self):
    """Werewolves choose a player to eliminate."""
    werewolves_alive = [
        w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    if not werewolves_alive:
      raise ValueError("No alive werewolves available to eliminate.")

    # All alive wolves propose a target; final target is a joint decision.
    wolf_votes = {}
    wolf_logs = {}
    for wolf in werewolves_alive:
      voted_target, log = wolf.eliminate()
      wolf_votes[wolf.name] = voted_target
      wolf_logs[wolf.name] = log

    legal_targets = [
        name
        for name in self.this_round.players
        if name not in [wolf.name for wolf in werewolves_alive]
    ]
    (deciding_wolf_name, eliminated), decision_logs = self._resolve_wolf_decision(
        werewolves_alive=werewolves_alive,
        legal_targets=legal_targets,
        wolf_votes=wolf_votes,
        wolf_logs=wolf_logs,
    )

    if eliminated is None:
      if not legal_targets:
        raise ValueError("Werewolves could not eliminate: no legal targets.")
      eliminated = legal_targets[0]
      deciding_wolf_name = werewolves_alive[0].name
      deciding_display = werewolves_alive[0].get_display_name(deciding_wolf_name)
      eliminated_display = werewolves_alive[0].get_display_name(eliminated)
      tqdm.tqdm.write(
          f"{deciding_display} returned an invalid eliminate target; fallback"
          f" target is {eliminated_display}."
      )

    self.this_round_log.eliminate = decision_logs.get(
        deciding_wolf_name, next(iter(wolf_logs.values()))
    )
    self.this_round.eliminated = eliminated
    deciding_wolf = self.state.players[deciding_wolf_name]
    display_wolf = deciding_wolf.get_display_name(deciding_wolf_name)
    display_eliminated = deciding_wolf.get_display_name(eliminated)
    tqdm.tqdm.write(f"{display_wolf} eliminated {display_eliminated}")
    for wolf in werewolves_alive:
      display_eliminated_obs = wolf.get_display_name(eliminated)
      wolf._add_observation(
          "During the"
          f" night, {'we' if len(werewolves_alive) > 1 else 'I'} decided to"
          f" eliminate {display_eliminated_obs}."
      )

  def protect(self):
    """Doctor chooses a player to protect."""
    if self.state.doctor.name not in self.this_round.players:
      return  # Doctor no longer in the game

    protect, log = self.state.doctor.save()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      display_doctor = self.state.doctor.get_display_name(self.state.doctor.name)
      display_protect = self.state.doctor.get_display_name(protect)
      tqdm.tqdm.write(f"{display_doctor} protected {display_protect}")
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
      self.state.seer.reveal_and_update(unmask, self.state.players[unmask].role)
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
        player = self.state.players[player_name]
        display_name = player.get_display_name(player_name)
        tqdm.tqdm.write(f"{display_name} summary: {summary}")
        self.this_round_log.summaries.append((player_name, log))

  def _speaker_sort_key(self, player_name: str):
    """Sort players by numeric suffix if available, else lexicographically."""
    display_name = self.state.players[player_name].get_display_name(player_name)
    match = re.search(r"(\d+)$", display_name)
    if match:
      return (0, int(match.group(1)))
    return (1, display_name)

  def _get_day_speaking_order(self) -> List[str]:
    """Get deterministic speaking order for the day phase."""
    alive_players = self.this_round.players.copy()
    if not alive_players:
      return []

    # If someone died last night, start from the next seat after that player.
    if (
        self.this_round.eliminated
        and self.this_round.eliminated != self.this_round.protected
        and self.this_round.eliminated in self.round_start_players
    ):
      start_idx = (
          self.round_start_players.index(self.this_round.eliminated) + 1
      ) % len(self.round_start_players)
      alive_set = set(alive_players)
      ordered_players = []
      for i in range(len(self.round_start_players)):
        player = self.round_start_players[
            (start_idx + i) % len(self.round_start_players)
        ]
        if player in alive_set:
          ordered_players.append(player)
      return ordered_players

    # If nobody died, use ascending player index/identifier.
    return sorted(alive_players, key=self._speaker_sort_key)

  def run_day_phase(self):
    """Run the day phase which consists of the debate and voting."""

    speaking_order = self._get_day_speaking_order()
    debate_turns = min(len(speaking_order), MAX_DEBATE_TURNS)

    for idx in range(debate_turns):
      next_speaker = speaking_order[idx]
      if not next_speaker:
        raise ValueError("Day speaking order produced an invalid player.")

      player = self.state.players[next_speaker]
      dialogue, log = player.debate()
      if dialogue is None:
        raise ValueError(
            f"{next_speaker} did not return a valid dialouge from debate()."
        )

      self.this_round_log.debate.append((next_speaker, log))
      self.this_round.debate.append([next_speaker, dialogue])
      display_speaker = player.get_display_name(next_speaker)
      tqdm.tqdm.write(f"{display_speaker} ({player.role}): {dialogue}")

      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.update_debate(next_speaker, dialogue)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")

      if idx == debate_turns - 1 or RUN_SYNTHETIC_VOTES:
        votes, vote_logs = self.run_voting()
        self.this_round.votes.append(votes)
        self.this_round_log.votes.append(vote_logs)

    for player_name, vote in self.this_round.votes[-1].items():
      player_obj = self.state.players[player_name]
      display_player = player_obj.get_display_name(player_name)
      display_vote = player_obj.get_display_name(vote)
      tqdm.tqdm.write(f"{display_player} voted to remove {display_vote}")

  def run_voting(self, candidate_options=None):
    """Conduct a vote among players to exile someone."""
    vote_log = []
    votes = {}
    allowed_candidates = None
    if candidate_options is not None:
      allowed_candidates = [
          c for c in candidate_options if c in self.this_round.players
      ]

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_votes = {
          name: executor.submit(self.state.players[name].vote, allowed_candidates)
          for name in self.this_round.players
      }

      for player_name, vote_task in player_votes.items():
        vote, log = vote_task.result()
        if vote is None:
          if allowed_candidates:
            fallback_options = [
                name for name in allowed_candidates if name != player_name
            ]
          else:
            fallback_options = [
                name for name in self.this_round.players if name != player_name
            ]
          if not fallback_options:
            raise ValueError(
                f"{player_name} could not vote and no fallback targets exist."
            )
          vote = random.choice(fallback_options)
          display_player = self.state.players[player_name].get_display_name(
              player_name
          )
          display_vote = self.state.players[player_name].get_display_name(vote)
          tqdm.tqdm.write(
              f"{display_player} returned an invalid vote; fallback vote is"
              f" {display_vote}."
          )

        vote_log.append(VoteLog(player_name, vote, log))
        votes[player_name] = vote

    return votes, vote_log

  def exile(self):
    """Exile the player who received the most votes."""

    vote_counter = Counter(self.this_round.votes[-1].values())
    most_voted, vote_count = vote_counter.most_common(1)[0]
    top_candidates = [
        player for player, count in vote_counter.items() if count == vote_count
    ]

    # Tie at top votes triggers one runoff among tied candidates.
    if len(top_candidates) > 1:
      tied_display = ", ".join(
          self.state.players[self.this_round.players[0]].get_display_name(c)
          for c in top_candidates
      )
      tqdm.tqdm.write(
          f"Top vote tie between: {tied_display}. Running one runoff vote."
      )
      runoff_votes, runoff_vote_logs = self.run_voting(top_candidates)
      self.this_round.votes.append(runoff_votes)
      self.this_round_log.votes.append(runoff_vote_logs)
      vote_counter = Counter(runoff_votes.values())
      most_voted, vote_count = vote_counter.most_common(1)[0]
      top_candidates = [
          player for player, count in vote_counter.items() if count == vote_count
      ]

    if (
        len(top_candidates) == 1
        and vote_count >= len(self.this_round.players) / 2
    ):
      self.this_round.exiled = most_voted

    if self.this_round.exiled is not None:
      exiled_player = self.this_round.exiled
      self.this_round.players.remove(exiled_player)
      # Get display name from any player (they all have the same mapping)
      if self.this_round.players:
        sample_player = self.state.players[self.this_round.players[0]]
        display_exiled = sample_player.get_display_name(exiled_player)
      else:
        display_exiled = exiled_player
      announcement = (
          f"The majority voted to remove {display_exiled} from the game."
      )
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
    was_night_kill_successful = (
        self.this_round.eliminated is not None
        and self.this_round.eliminated != self.this_round.protected
    )
    if self.this_round.eliminated != self.this_round.protected:
      eliminated_player = self.this_round.eliminated
      self.this_round.players.remove(eliminated_player)
      # Get display name from any player (they all have the same mapping)
      if self.this_round.players:
        sample_player = self.state.players[self.this_round.players[0]]
        display_eliminated = sample_player.get_display_name(eliminated_player)
      else:
        display_eliminated = eliminated_player
      announcement = (
          f"The Werewolves removed {display_eliminated} from the game during the"
          " night."
      )
    else:
      announcement = "No one was removed from the game during the night."
    tqdm.tqdm.write(announcement)

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate and was_night_kill_successful:
        player.gamestate.remove_player(self.this_round.eliminated)
      player.add_announcement(announcement)

  def run_round(self):
    """Run a single round of the game."""
    self.state.rounds.append(Round())
    self.logs.append(RoundLog())

    self.this_round.players = (
        list(self.state.players.keys())
        if self.current_round_num == 0
        else self.state.rounds[self.current_round_num - 1].players.copy()
    )
    self.round_start_players = self.this_round.players.copy()

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
      self.current_round_num += 1

    tqdm.tqdm.write("Game is complete!")
    return self.state.winner
