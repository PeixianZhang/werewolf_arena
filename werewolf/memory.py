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
        self.long_term_dialogue: List[Tuple[int, str, str]] = []  # (round_number, speaker, dialogue) - persistent across rounds
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

    def add_dialogue(self, speaker: str, dialogue: str, round_number: Optional[int] = None):
        """Add a dialogue entry to short-term memory and long-term dialogue.
        
        Args:
            speaker: Name of the player who spoke
            dialogue: What they said
            round_number: Current round number (if None, will not be stored in long_term)
        """
        self.short_term.append((speaker, dialogue))
        # Also store in long-term dialogue if round_number is provided
        if round_number is not None:
            self.long_term_dialogue.append((round_number, speaker, dialogue))

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
    
    def add_system_message(self, message: str, round_number: Optional[int] = None):
        """Add a system message to short-term memory and optionally long-term memory.
        
        Args:
            message: System message (e.g., "Current Date - Day X", death reports, vote records)
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        self.short_term.append(("SYSTEM", message))
        if round_number is not None:
            self.long_term_dialogue.append((round_number, "SYSTEM", message))
    
    def add_day_timestamp(self, day_number: int):
        """Add day timestamp at the start of each day.
        
        Args:
            day_number: Current day/round number
        """
        self.add_system_message(f"Current Date - Day {day_number}", round_number=day_number)
    
    def add_death_report(self, player_name: str, cause: str = "night", round_number: Optional[int] = None):
        """Add death report to short-term memory.
        
        Args:
            player_name: Name of the dead player
            cause: "night" for werewolf elimination, "vote" for exile
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        if cause == "night":
            self.add_system_message(f"[Nightly Report] Player {player_name} died last night.", round_number=round_number)
        elif cause == "vote":
            self.add_system_message(f"[Vote Record] Player {player_name} was exiled by majority vote.", round_number=round_number)
        # Update player status
        self.update_player_status(player_name, False)
    
    def add_vote_records(self, votes: Dict[str, str], round_number: Optional[int] = None):
        """Add vote records for all players to short-term memory.
        
        Args:
            votes: Dictionary mapping voter_name to voted_for_name
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        for voter, voted_for in votes.items():
            self.add_system_message(f"[Vote Record] Player {voter} voted for Player {voted_for}.", round_number=round_number)
    
    def add_private_memory(self, action_type: str, description: str, round_number: Optional[int] = None):
        """Add role-specific private memory.
        
        Args:
            action_type: Type of private action ("Seer", "Doctor", "Werewolf")
            description: Description of the private action
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        self.private_memory.append((action_type, description))
        # Also add to short_term with PRIVATE prefix
        self.short_term.append(("PRIVATE", f"[{action_type}] {description}"))
        if round_number is not None:
            self.long_term_dialogue.append((round_number, "PRIVATE", f"[{action_type}] {description}"))
    
    def add_seer_verification(self, verified_player: str, side: str, round_number: Optional[int] = None):
        """Record Seer's verification result.
        
        Args:
            verified_player: Name of the player verified
            side: "Werewolf" or "Villager" (Seer sees side, not exact role)
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        self.add_private_memory("Seer", f"I verified Player {verified_player}, they are {side}.", round_number=round_number)
    
    def add_doctor_protection(self, protected_player: str, round_number: Optional[int] = None):
        """Record Doctor's protection action.
        
        Args:
            protected_player: Name of the player protected
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        self.add_private_memory("Doctor", f"I protected Player {protected_player} tonight.", round_number=round_number)
    
    def add_werewolf_target(self, target_player: str, round_number: Optional[int] = None):
        """Record Werewolf team's target decision.
        
        Args:
            target_player: Name of the player targeted
            round_number: Current round number (if provided, will also store in long_term_dialogue)
        """
        self.add_private_memory("Werewolf", f"Our team decided to target Player {target_player}.", round_number=round_number)

    def extract_important_claims(self, current_round: int):
        """Extract important claims (role declarations, etc.) from short_term before clearing.
        
        This method scans the current round's dialogue for important claims like role declarations
        and stores them in evidence_archive for persistent memory.
        
        Args:
            current_round: Current round number
        """
        role_keywords = {
            "Seer": ["i am the seer", "i'm the seer", "i am seer", "i'm seer", 
                     "i am a seer", "i'm a seer", "i am the seer", "i claim to be seer",
                     "i reveal myself as seer", "i'm revealing as seer"],
            "Doctor": ["i am the doctor", "i'm the doctor", "i am doctor", "i'm doctor",
                       "i am a doctor", "i'm a doctor", "i claim to be doctor",
                       "i reveal myself as doctor", "i'm revealing as doctor"],
            "Werewolf": ["i am a werewolf", "i'm a werewolf", "i am werewolf", "i'm werewolf",
                         "i am the werewolf", "i'm the werewolf"]  # Less likely but possible
        }
        
        for speaker, dialogue in self.short_term:
            # Skip SYSTEM and PRIVATE messages
            if speaker in ["SYSTEM", "PRIVATE"]:
                continue
                
            dialogue_lower = dialogue.lower()
            for role, keywords in role_keywords.items():
                if any(keyword in dialogue_lower for keyword in keywords):
                    # Detect role claim, store in evidence_archive
                    self.add_evidence(
                        evidence_type="role_claim",
                        description=f"{speaker} claimed to be {role}: \"{dialogue}\"",
                        players_involved=[speaker],
                        round_number=current_round,
                        weight=0.9  # High weight for role claims
                    )
                    # Also update belief matrix with high probability
                    self.update_belief(speaker, role, 0.8)
    
    def clear_short_term(self, current_round: Optional[int] = None):
        """Clear short-term memory (typically at the end of a round).
        
        Before clearing, extracts important claims (role declarations) to evidence_archive.
        
        Args:
            current_round: Current round number (if provided, will extract important claims before clearing)
        """
        # Extract important claims before clearing
        if current_round is not None:
            self.extract_important_claims(current_round)
        self.short_term.clear()
    
    def generate_observations_summary(self, current_round: int, include_all_rounds: bool = False) -> List[str]:
        """Generate observations-style summary from MemoryManager data.
        
        This method converts structured memory data into a format similar to
        the legacy observations system, for use in reflect_on_roles().
        
        Uses long_term_dialogue for historical data and short_term for current round.
        
        Args:
            current_round: Current round number
            include_all_rounds: If True, include all rounds; if False, only current and previous rounds
            
        Returns:
            List of formatted observation strings, similar to Player.observations format
        """
        summary_lines = []
        
        # Combine current round (short_term) and historical (long_term_dialogue) data
        all_dialogues = []
        
        # Add current round dialogues from short_term
        for speaker, dialogue in self.short_term:
            all_dialogues.append((current_round, speaker, dialogue))
        
        # Add historical dialogues from long_term_dialogue (if include_all_rounds)
        if include_all_rounds:
            for round_num, speaker, dialogue in self.long_term_dialogue:
                if round_num < current_round:  # Only include previous rounds
                    all_dialogues.append((round_num, speaker, dialogue))
        
        # Sort by round number
        all_dialogues.sort(key=lambda x: x[0])
        
        # 1. Extract system events and dialogues grouped by round
        round_events = defaultdict(list)
        round_dialogues_dict = defaultdict(list)
        round_votes = defaultdict(list)
        
        for round_num, speaker, dialogue in all_dialogues:
            if speaker == "SYSTEM":
                if "Current Date - Day" in dialogue:
                    # Extract round number from "Current Date - Day X"
                    try:
                        day_num = int(dialogue.split("Day")[1].strip())
                        round_events[round_num].append(("timestamp", dialogue))
                    except:
                        round_events[round_num].append(("timestamp", dialogue))
                elif "Nightly Report" in dialogue or "died last night" in dialogue:
                    # Death events
                    round_events[round_num].append(("death", dialogue))
                elif "voted for" in dialogue and "was exiled" not in dialogue:
                    # Individual vote records
                    round_votes[round_num].append(dialogue)
                elif "was exiled" in dialogue:
                    # Exile announcement
                    round_events[round_num].append(("exile", dialogue))
                elif "Summary:" in dialogue:
                    # Summary messages
                    round_events[round_num].append(("summary", dialogue))
            elif speaker == "PRIVATE":
                # Private actions - format as observations
                if "verified" in dialogue.lower():
                    round_events[round_num].append(("private_seer", dialogue))
                elif "protected" in dialogue.lower():
                    round_events[round_num].append(("private_doctor", dialogue))
                elif "targeted" in dialogue.lower() or "decided to target" in dialogue.lower():
                    round_events[round_num].append(("private_werewolf", dialogue))
            else:
                # Player dialogue
                round_dialogues_dict[round_num].append((speaker, dialogue))
        
        # 2. Generate observations for each round (in chronological order)
        rounds_to_include = sorted(set(r for r, _, _ in all_dialogues)) if include_all_rounds else [current_round - 1, current_round] if current_round > 0 else [current_round]
        
        for round_num in rounds_to_include:
            if round_num < 0:
                continue
            
            # Add round timestamp
            if round_num in round_events:
                for event_type, event_text in round_events[round_num]:
                    if event_type == "timestamp":
                        summary_lines.append(f"Round {round_num}: {event_text}")
            
            # Add death announcements (from previous night, shown at start of day)
            if round_num > 0:
                prev_round = round_num - 1
                if prev_round in round_events:
                    for event_type, event_text in round_events[prev_round]:
                        if event_type == "death":
                            summary_lines.append(f"Round {round_num}: Moderator Announcement: {event_text}")
            
            # Add private actions (from previous night)
            if round_num > 0:
                prev_round = round_num - 1
                if prev_round in round_events:
                    for event_type, event_text in round_events[prev_round]:
                        if event_type == "private_seer":
                            summary_lines.append(f"Round {round_num}: During the night, {event_text.replace('PRIVATE: ', 'I ')}")
                        elif event_type == "private_doctor":
                            summary_lines.append(f"Round {round_num}: During the night, {event_text.replace('PRIVATE: ', 'I ')}")
                        elif event_type == "private_werewolf":
                            summary_lines.append(f"Round {round_num}: During the night, {event_text.replace('PRIVATE: ', 'We ')}")
            
            # Add round dialogues
            if round_num in round_dialogues_dict:
                for speaker, dialogue in round_dialogues_dict[round_num]:
                    summary_lines.append(f"{speaker}: {dialogue}")
            
            # Add vote records
            if round_num in round_votes:
                for vote_record in round_votes[round_num]:
                    summary_lines.append(f"Round {round_num}: After the debate, {vote_record}")
            
            # Add exile announcements
            if round_num in round_events:
                for event_type, event_text in round_events[round_num]:
                    if event_type == "exile":
                        summary_lines.append(f"Round {round_num}: Moderator Announcement: {event_text}")
            
            # Add summaries
            if round_num in round_events:
                for event_type, event_text in round_events[round_num]:
                    if event_type == "summary":
                        summary_lines.append(f"Round {round_num}: {event_text}")
        
        # 8. Add key evidence from archive (role claims, etc.)
        if include_all_rounds and self.evidence_archive:
            # Filter for role claims specifically
            role_claims = [ev for ev in self.evidence_archive if ev.get("type") == "role_claim"]
            if role_claims:
                summary_lines.append("\n--- IMPORTANT ROLE CLAIMS (Historical) ---")
                for ev in sorted(role_claims, key=lambda x: x["round"]):
                    summary_lines.append(f"Round {ev['round']}: {ev['description']}")
        
        return summary_lines

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
        
        # RECENT DIALOGUE (Current Round + Recent Historical)
        # Get current round dialogues
        current_dialogues = [(s, d) for s, d in self.short_term if s not in ["SYSTEM", "PRIVATE"]]
        
        # Get recent historical dialogues (last 2 rounds)
        historical_dialogues = []
        if self.long_term_dialogue:
            recent_rounds = [current_round - 2, current_round - 1] if current_round >= 2 else ([current_round - 1] if current_round >= 1 else [])
            for round_num, speaker, dialogue in self.long_term_dialogue:
                if round_num in recent_rounds and speaker not in ["SYSTEM", "PRIVATE"]:
                    historical_dialogues.append((round_num, speaker, dialogue))
        
        # Combine and display
        if current_dialogues or historical_dialogues:
            context_parts.append("\n=== RECENT DIALOGUE ===")
            # Add historical dialogues first
            for round_num, speaker, dialogue in historical_dialogues[-5:]:  # Last 5 from previous rounds
                context_parts.append(f"  - Round {round_num}: {speaker}: {dialogue}")
            # Add current round dialogues
            for speaker, dialogue in current_dialogues[-10:]:  # Last 10 from current round
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
    
    def generate_key_information(self, current_round: int, current_debate: List) -> List[str]:
        """Generate key_information list directly from short_term (includes private memory).
        
        Simplified approach: directly extract all information from short_term without
        complex processing through long_term_dialogue or evidence_archive.
        
        Args:
            current_round: Current round number
            current_debate: List of current round debate items (can be strings or tuples)
            
        Returns:
            List of formatted information strings
        """
        key_items = []
        
        # 1. Extract all information from short_term (already includes private memory)
        for speaker, dialogue in self.short_term:
            if speaker == "SYSTEM":
                # System events (death reports, vote records, etc.)
                key_items.append(f"Round {current_round}: {dialogue}")
            elif speaker == "PRIVATE":
                # Private memory (Seer verifications, Doctor protections, Werewolf targets)
                key_items.append(f"Round {current_round}: {dialogue}")
            else:
                # Player dialogue
                key_items.append(f"{speaker}: {dialogue}")
        
        # 2. Add current round debate
        for item in current_debate:
            if isinstance(item, str):
                key_items.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                key_items.append(f"{item[0]}: {item[1]}")
            else:
                key_items.append(str(item))
        
        return key_items
    
    def update_belief_from_deduction(self, deduction_result: Dict[str, Any], remaining_players: List[str]):
        """Update belief_matrix from deduction result.
        
        Args:
            deduction_result: Result from reflect_on_roles containing deductions
            remaining_players: List of currently alive players
        """
        deductions = deduction_result.get("deductions", [])
        
        for ded in deductions:
            player_name = ded.get("player")
            role = ded.get("role")
            confidence = ded.get("confidence", 5)  # Default to 5 (pure guess) if not provided
            
            if player_name and role and player_name in remaining_players:
                # Convert confidence (5-10 scale) to probability (0.0-1.0)
                # 5 = pure guess -> 0.0, 10 = absolutely sure -> 1.0
                probability = (confidence - 5) / 5.0
                probability = max(0.0, min(1.0, probability))
                
                self.update_belief(player_name, role, probability)
                self.update_player_status(player_name, True)

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
            "long_term_dialogue": self.long_term_dialogue,  # Include long-term dialogue
            "belief_matrix": dict(self.belief_matrix),
            "evidence_archive": self.evidence_archive,
            "private_memory": self.private_memory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "MemoryManager":
        """Create MemoryManager from dictionary."""
        mm = cls(data["player_name"])
        mm.short_term = [tuple(item) for item in data.get("short_term", [])]
        # Handle long_term_dialogue (may not exist in old saves)
        long_term_data = data.get("long_term_dialogue", [])
        mm.long_term_dialogue = [tuple(item) if isinstance(item, (list, tuple)) else item for item in long_term_data]
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

