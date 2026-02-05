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

import tqdm

from werewolf.lm import LmLog
from werewolf.model import Round, RoundLog, State, VoteLog
from werewolf.config import  MAX_DEBATE_TURNS, RUN_SYNTHETIC_VOTES

def get_max_bids(d):
  """Gets all the keys with the highest value in the dictionary."""
  max_value = max(d.values())
  max_keys = [key for key, value in d.items() if value == max_value]
  return max_keys


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
    for w in werewolves_alive:
      w._add_observation(
          "During the"
          f" night, {'we' if len(werewolves_alive) > 1 else 'I'} decided to"
          f" eliminate {eliminated}."
      )

  def protect(self):
    """Doctor chooses a player to protect."""
    if self.state.doctor.name not in self.this_round.players:
      return  # Doctor no longer in the game

    protect, log = self.state.doctor.save()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      tqdm.tqdm.write(f"{self.state.doctor.name} protected {protect}")
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

  def _get_bid(self, player_name):
    """Gets the bid for a specific player."""
    player = self.state.players[player_name]
    bid, log = player.bid()
    if bid is None:
      raise ValueError(
          f"{player_name} did not return a valid bid. Find the raw response"
          " in the `bid` field in the log"
      )
    if bid > 1:
      tqdm.tqdm.write(f"{player_name} bid: {bid}")
    return bid, log

  def get_next_speaker(self):
    """Determine the next speaker based on bids."""
    previous_speaker, previous_dialogue = (
        self.this_round.debate[-1] if self.this_round.debate else (None, None)
    )

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_bids = {
          player_name: executor.submit(self._get_bid, player_name)
          for player_name in self.this_round.players
          if player_name != previous_speaker
      }

      bid_log = []
      bids = {}
      try:
        for player_name, bid_task in player_bids.items():
          bid, log = bid_task.result()
          bids[player_name] = bid
          bid_log.append((player_name, log))
      except TypeError as e:
        print(e)
        raise e

    self.this_round.bids.append(bids)
    self.this_round_log.bid.append(bid_log)

    potential_speakers = get_max_bids(bids)
    # Prioritize mentioned speakers if there's previous dialogue
    if previous_dialogue:
      potential_speakers.extend(
          [name for name in potential_speakers if name in previous_dialogue]
      )

    random.shuffle(potential_speakers)
    return random.choice(potential_speakers)

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

  def run_day_phase(self):
    """Run the day phase which consists of the debate and voting."""

    for idx in range(MAX_DEBATE_TURNS):
      next_speaker = self.get_next_speaker()
      if not next_speaker:
        raise ValueError("get_next_speaker did not return a valid player.")

      player = self.state.players[next_speaker]
      dialogue, log = player.debate()
      if dialogue is None:
        raise ValueError(
            f"{next_speaker} did not return a valid dialouge from debate()."
        )

      self.this_round_log.debate.append((next_speaker, log))
      self.this_round.debate.append([next_speaker, dialogue])
      tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.update_debate(next_speaker, dialogue)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")

      if idx == MAX_DEBATE_TURNS - 1 or RUN_SYNTHETIC_VOTES:
        # 投票前：每个 agent 做角色反思（role/reasoning/confidence/evidence），写入 log
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

    if self.this_round.exiled is not None:
      exiled_player = self.this_round.exiled
      if exiled_player in self.this_round.players:
        self.this_round.players.remove(exiled_player)
      announcement = (
          f"The majority voted to remove {exiled_player} from the game."
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
    if self.this_round.eliminated != self.this_round.protected:
      eliminated_player = self.this_round.eliminated
      if eliminated_player and eliminated_player in self.this_round.players:
        self.this_round.players.remove(eliminated_player)
      announcement = (
          f"The Werewolves removed {eliminated_player} from the game during the"
          " night."
      )
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
    else:
      self.this_round.players = list(
          self.state.rounds[self.current_round_num - 1].players
      )

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
