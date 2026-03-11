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

"""Memory Management for Werewolf Agents."""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json


class MemoryManager:
    """Manages structured memory for Werewolf game agents.
    
    This class maintains three types of memory:
    - short_term: Current round's raw dialogue and system events
    - belief_matrix: Probability inferences for each player's role (with is_alive status)
    - evidence_archive: Key logical contradictions and important evidence
    - private_memory: Role-specific private memories (Seer verifications, Doctor protections, Werewolf targets)
    """

    def __init__(self, player_name: str):
        """Initialize MemoryManager for a specific player.
        
        Args:
            player_name: Name of the player this memory manager belongs to
        """
        self.player_name = player_name
        self.short_term: List[Tuple[str, str]] = []  # (speaker, dialogue) or ("SYSTEM", message) or ("PRIVATE", message)
        self.belief_matrix: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "Werewolf": 0.0, 
                "Seer": 0.0, 
                "Doctor": 0.0, 
                "Villager": 0.0, 
                "is_alive": True  # Track player's alive status
            }
        )
        self.evidence_archive: List[Dict[str, any]] = []  # List of evidence entries
        self.private_memory: List[Tuple[str, str]] = []  # (role_action, description) for role-specific memories

    def add_dialogue(self, speaker: str, dialogue: str):
        """Add a dialogue entry to short-term memory.
        
        Args:
            speaker: Name of the player who spoke
            dialogue: What they said
        """
        self.short_term.append((speaker, dialogue))

    def update_belief(self, player_name: str, role: str, probability: float):
        """Update belief matrix for a player's role probability.
        
        Args:
            player_name: Name of the player
            role: One of ["Werewolf", "Seer", "Doctor", "Villager"]
            probability: Probability value between 0.0 and 1.0
        """
        if role in self.belief_matrix[player_name]:
            self.belief_matrix[player_name][role] = probability
    
    def update_player_status(self, player_name: str, alive_status: bool):
        """Update player's alive status in belief matrix.
        
        Args:
            player_name: Name of the player
            alive_status: True if alive, False if dead
        """
        self.belief_matrix[player_name]["is_alive"] = alive_status

    def add_evidence(self, evidence_type: str, description: str, players_involved: List[str], 
                     round_number: int, weight: float = 1.0):
        """Add evidence to the evidence archive.
        
        Args:
            evidence_type: Type of evidence (e.g., "logical_clash", "contradiction", "claim")
            description: Description of the evidence
            players_involved: List of player names involved
            round_number: Round number when this evidence was observed
            weight: Importance weight (0.0 to 1.0)
        """
        self.evidence_archive.append({
            "type": evidence_type,
            "description": description,
            "players_involved": players_involved,
            "round": round_number,
            "weight": weight,
            "timestamp": len(self.evidence_archive)  # Simple sequential timestamp
        })
    
    def add_system_message(self, message: str):
        """Add a system message to short-term memory.
        
        Args:
            message: System message (e.g., "Current Date - Day X", death reports, vote records)
        """
        self.short_term.append(("SYSTEM", message))
    
    def add_day_timestamp(self, day_number: int):
        """Add day timestamp at the start of each day.
        
        Args:
            day_number: Current day/round number
        """
        self.add_system_message(f"Current Date - Day {day_number}")
    
    def add_death_report(self, player_name: str, cause: str = "night"):
        """Add death report to short-term memory.
        
        Args:
            player_name: Name of the dead player
            cause: "night" for werewolf elimination, "vote" for exile
        """
        if cause == "night":
            self.add_system_message(f"[Nightly Report] Player {player_name} died last night.")
        elif cause == "vote":
            self.add_system_message(f"[Vote Record] Player {player_name} was exiled by majority vote.")
        # Update player status
        self.update_player_status(player_name, False)
    
    def add_vote_records(self, votes: Dict[str, str]):
        """Add vote records for all players to short-term memory.
        
        Args:
            votes: Dictionary mapping voter_name to voted_for_name
        """
        for voter, voted_for in votes.items():
            self.add_system_message(f"[Vote Record] Player {voter} voted for Player {voted_for}.")
    
    def add_private_memory(self, action_type: str, description: str):
        """Add role-specific private memory.
        
        Args:
            action_type: Type of private action ("Seer", "Doctor", "Werewolf")
            description: Description of the private action
        """
        self.private_memory.append((action_type, description))
        # Also add to short_term with PRIVATE prefix
        self.short_term.append(("PRIVATE", f"[{action_type}] {description}"))
    
    def add_seer_verification(self, verified_player: str, side: str):
        """Record Seer's verification result.
        
        Args:
            verified_player: Name of the player verified
            side: "Werewolf" or "Villager" (Seer sees side, not exact role)
        """
        self.add_private_memory("Seer", f"I verified Player {verified_player}, they are {side}.")
    
    def add_doctor_protection(self, protected_player: str):
        """Record Doctor's protection action.
        
        Args:
            protected_player: Name of the player protected
        """
        self.add_private_memory("Doctor", f"I protected Player {protected_player} tonight.")
    
    def add_werewolf_target(self, target_player: str):
        """Record Werewolf team's target decision.
        
        Args:
            target_player: Name of the player targeted
        """
        self.add_private_memory("Werewolf", f"Our team decided to target Player {target_player}.")

    def clear_short_term(self):
        """Clear short-term memory (typically at the end of a round)."""
        self.short_term.clear()

    def get_context(self, current_round: int, max_evidence: int = 5, remaining_players: Optional[List[str]] = None) -> str:
        """Generate a context string with memory summary for LLM prompts.
        
        Args:
            current_round: Current round number
            max_evidence: Maximum number of evidence entries to include
            remaining_players: Optional list of alive players to filter belief matrix
            
        Returns:
            A formatted string containing memory summary
        """
        context_parts = []
        
        # Get alive players list (filter belief_matrix if remaining_players provided)
        alive_players = remaining_players if remaining_players else []
        
        # ALIVE PLAYERS STATUS (置顶展示)
        if alive_players:
            context_parts.append("=== ALIVE PLAYERS ===")
            alive_beliefs = []
            for player_name in alive_players:
                if player_name == self.player_name:
                    continue
                beliefs = self.belief_matrix.get(player_name, {})
                if beliefs:
                    # Filter out is_alive from role probabilities
                    role_beliefs = {k: v for k, v in beliefs.items() if k != "is_alive"}
                    if role_beliefs:
                        sorted_roles = sorted(role_beliefs.items(), key=lambda x: x[1], reverse=True)
                        top_role, top_prob = sorted_roles[0]
                        second_role, second_prob = sorted_roles[1] if len(sorted_roles) > 1 else (sorted_roles[0][0], 0.0)
                        alive_beliefs.append(
                            f"  - {player_name}: Likely {top_role} ({top_prob:.2f}), "
                            f"possibly {second_role} ({second_prob:.2f})"
                        )
            if alive_beliefs:
                context_parts.extend(alive_beliefs)
            else:
                context_parts.append("  (No strong beliefs about alive players yet)")
        
        # SYSTEM EVENTS (死亡报告和投票记录，按时间顺序)
        system_events = []
        for speaker, dialogue in self.short_term:
            if speaker == "SYSTEM":
                system_events.append(f"  - {dialogue}")
        
        if system_events:
            context_parts.append("\n=== SYSTEM EVENTS (Chronological) ===")
            # Show recent system events (last 15)
            context_parts.extend(system_events[-15:])
        
        # PRIVATE MEMORY (角色特权记忆)
        if self.private_memory:
            context_parts.append("\n=== PRIVATE MEMORY (Role-Specific) ===")
            for action_type, description in self.private_memory[-10:]:  # Last 10 private memories
                context_parts.append(f"  - [{action_type}] {description}")
        
        # RECENT DIALOGUE (Current Round)
        dialogue_entries = [(s, d) for s, d in self.short_term if s not in ["SYSTEM", "PRIVATE"]]
        if dialogue_entries:
            context_parts.append("\n=== RECENT DIALOGUE (Current Round) ===")
            for speaker, dialogue in dialogue_entries[-10:]:  # Last 10 dialogues
                context_parts.append(f"  - {speaker}: {dialogue}")
        
        # BELIEF MATRIX (only for alive players)
        if self.belief_matrix and alive_players:
            context_parts.append("\n=== BELIEF MATRIX & SOCIAL PERCEPTION ===")
            for player_name in alive_players:
                if player_name == self.player_name:
                    continue
                beliefs = self.belief_matrix.get(player_name, {})
                if beliefs:
                    # Filter out is_alive
                    role_beliefs = {k: v for k, v in beliefs.items() if k != "is_alive"}
                    if role_beliefs:
                        sorted_roles = sorted(role_beliefs.items(), key=lambda x: x[1], reverse=True)
                        top_role, top_prob = sorted_roles[0]
                        second_role, second_prob = sorted_roles[1] if len(sorted_roles) > 1 else (sorted_roles[0][0], 0.0)
                        
                        if top_prob > 0:
                            context_parts.append(
                                f"  - {player_name}: Likely {top_role} ({top_prob:.2f}), "
                                f"possibly {second_role} ({second_prob:.2f})"
                            )
        
        # Evidence archive summary
        if self.evidence_archive:
            context_parts.append("\n=== KEY EVIDENCE ARCHIVE ===")
            # Sort by weight and recency
            sorted_evidence = sorted(
                self.evidence_archive,
                key=lambda x: (x["weight"], x["round"]),
                reverse=True
            )[:max_evidence]
            for ev in sorted_evidence:
                context_parts.append(
                    f"  - [{ev['type']}] Round {ev['round']}: {ev['description']} "
                    f"(Weight: {ev['weight']:.2f}, Players: {', '.join(ev['players_involved'])})"
                )
        
        return "\n".join(context_parts) if context_parts else "No memory data available."

    def get_belief_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a copy of the current belief matrix."""
        return dict(self.belief_matrix)

    def reflect(self, player_role: str, current_round: int, remaining_players: List[str], 
                model: Optional[str] = None, generate_func=None) -> Optional[Dict[str, Any]]:
        """Reflect on short-term memory and update belief matrix.
        
        Uses LLM to analyze recent dialogue and update beliefs about other players.
        Also captures bias self-report for logging.
        
        Args:
            player_role: The role of the player (e.g., "Werewolf", "Villager")
            current_round: Current round number
            remaining_players: List of remaining player names
            model: Model name for LLM generation
            generate_func: Function to call for LLM generation (from werewolf.lm.generate)
            
        Returns:
            Dictionary containing reflection results including bias self-report, or None if reflection fails
        """
        if not generate_func:
            return None
            
        if not self.short_term:
            # No dialogue to reflect on yet
            return None
        
        # Prepare short-term memory context
        recent_dialogue = "\n".join([
            f"{speaker}: {dialogue}" 
            for speaker, dialogue in self.short_term[-10:]  # Last 10 dialogues
        ])
        
        # Prepare current belief summary (only for alive players)
        belief_summary = []
        for player_name in remaining_players:
            if player_name == self.player_name:
                continue
            beliefs = self.belief_matrix.get(player_name, {})
            if beliefs:
                # Filter out is_alive from role probabilities
                role_beliefs = {k: v for k, v in beliefs.items() if k != "is_alive"}
                if role_beliefs:
                    max_role = max(role_beliefs.items(), key=lambda x: x[1])
                    if max_role[1] > 0:
                        belief_summary.append(f"{player_name}: {max_role[0]} ({max_role[1]:.2f})")
        
        # Import prompt template
        from werewolf.prompts import MEMORY_REFLECTION, MEMORY_REFLECTION_SCHEMA
        
        worldstate = {
            "player_name": self.player_name,
            "role": player_role,
            "round": current_round,
            "recent_dialogue": recent_dialogue,
            "current_beliefs": "\n".join(belief_summary) if belief_summary else "No strong beliefs yet.",
            "remaining_players": ", ".join([p for p in remaining_players if p != self.player_name])
        }
        
        try:
            result, log = generate_func(
                MEMORY_REFLECTION,
                MEMORY_REFLECTION_SCHEMA,
                worldstate,
                model=model,
                temperature=0.7,
            )
            
            if result and isinstance(result, dict):
                # Update belief matrix from reflection (only for alive players)
                beliefs = result.get("belief_updates", [])
                for belief_update in beliefs:
                    player_name = belief_update.get("player")
                    role = belief_update.get("role")
                    confidence = belief_update.get("confidence", 0.5)
                    # Convert confidence (0-1) to probability
                    probability = max(0.0, min(1.0, confidence))
                    if player_name and role and player_name in remaining_players:
                        self.update_belief(player_name, role, probability)
                        # Ensure is_alive is True for players being updated
                        self.update_player_status(player_name, True)
                
                # Return reflection result including bias self-report
                return {
                    "reflection_summary": result.get("reflection_summary", ""),
                    "bias_self_report": result.get("bias_self_report", {}),
                    "belief_updates_count": len(beliefs)
                }
        except Exception as e:
            # Log error but don't fail the game
            print(f"Warning: Reflection failed for {self.player_name}: {e}")
            return None
        
        return None

    def to_dict(self) -> Dict[str, any]:
        """Convert memory to dictionary for serialization."""
        return {
            "player_name": self.player_name,
            "short_term": self.short_term,
            "belief_matrix": dict(self.belief_matrix),
            "evidence_archive": self.evidence_archive,
            "private_memory": self.private_memory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "MemoryManager":
        """Create MemoryManager from dictionary."""
        mm = cls(data["player_name"])
        mm.short_term = [tuple(item) for item in data.get("short_term", [])]
        mm.belief_matrix = defaultdict(
            lambda: {
                "Werewolf": 0.0, 
                "Seer": 0.0, 
                "Doctor": 0.0, 
                "Villager": 0.0, 
                "is_alive": True
            }
        )
        mm.belief_matrix.update(data.get("belief_matrix", {}))
        mm.evidence_archive = data.get("evidence_archive", [])
        mm.private_memory = [tuple(item) for item in data.get("private_memory", [])]
        return mm

