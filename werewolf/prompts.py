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

GAME = """You are playing a digital version of the social deduction game Werewolf (also known as Mafia).

GAME RULES:
- Player Roles: {{num_players}} players - 3 Werewolves, 1 Seer, 1 Doctor, {{num_villagers}} Villagers.
- Rounds consist of two phases:
    - Night Phase: The alive Werewolves, Seer, and Doctor can use their ability and take secret actions. 
                   Werewolf: choose a player to kill. If there are more than one Werewolves alive, the Werewolf with a smaller ID first proposes a player to kill. Then the proposal is added to the observation of the other Werewolf and this Werewolf decides the final kill target. For example, if player_0 and player_2 are the Werewolves, player_0 first proposes to kill player_i, then player_2 knows this information and decides to kill player_j. The final kill target is player_j. If there is only one Werewolf alive, then this Werewolf’s action is the final kill target. The Werewolf is not allowed to kill a dead player or kill themselves or kill their teammate.
                   Seer: choose a player to investigate. The Seer is not allowed to investigate a dead player or investigate themselves.
                   Doctor: choose a player to protect. The Doctor is not allowed to protect a dead player or protect themselves.
    - Day Phase: An announcement about last night’s result is announced to all remaining players. If a player is killed, they are immediately moved out of the game and cannot reveal their role or communicate with other players.Players debate and vote to remove one player. All remaining players take turns to speak only once in an open discussion. All remaining players simultaneously vote for one player or choose not to vote. Players are not allowed to vote for a dead player or themselves. The player with the most votes will be eliminated without revealing their role. If multiple players have the most votes, one player is randomly chosen and eliminated. The voting result is public and can be observed by all players.
    - Winning Conditions: Villagers, Seer and Doctor win by voting out all Werewolves. Werewolves win when they outnumber the Villagers."""

CHARACTER_INTRODUCTIONS = """CHARACTER INTRODUCTIONS (each player's demographic):
{% if character_introductions -%}
{{ character_introductions }}
{% else -%}
(No demographic info configured.)
{% endif -%}"""

STATE = """GAME STATE:
- It is currently Round {{round}}. {% if round == 0 %}The game has just begun.{% endif %}
- You are {{name}} the {{role}}. {{werewolf_context}}
{% if demographic -%}
- Demographic: {{ demographic }}
{% endif -%}
- Remaining players: {{remaining_players}}"""

# OBSERVATIONS = """{% if observations|length -%}YOUR PRIVATE OBSERVATIONS:
# {% for turn in observations -%}
# {{ turn }}
# {% endfor %}
# {% endif %}"""

DEBATE_SO_FAR_THIS_ROUND = """\nROUND {{round}} DEBATE:
{% if debate|length -%}
{% for turn in debate -%}
{{ turn }}
{% endfor -%}
{% else -%}
The debate has not begun.{% endif %}\n\n"""

PREFIX = f"""{GAME}

{CHARACTER_INTRODUCTIONS}

{STATE}

""".strip()  ## {OBSERVATIONS}

DEBATE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- You are speaking next in the debate as {{name}} the {{role}}.
{% if memory_context -%}
{{ memory_context }}
{% endif -%}
{% if role == 'Werewolf' -%}
- Your goal is to sow chaos and evade detection.
- Cast suspicion on Villagers. Make them doubt each other.
- Steer the conversation away from yourself and your fellow Werewolves.
- Appear helpful while undermining the Villagers' efforts.
- Deception is your greatest weapon. For example, you could claim a special role and falsely accuse a Villager or fabricate inconsistencies to sow confusion. Use these powerful tactics sparingly to avoid suspicion.
{% else -%}
- Your goal is to uncover the Werewolves and protect the Village.
- Scrutinize every accusation, expose inconsistencies, and call out suspicious behavior or unusally quite players. Don't hesitate to make bold accusations!
- Emphasize teamwork and propose strategies to expose the Werewolves. Working together will be key to identifying the Werewolves.
{% if role == 'Villager' -%}
- If someone reveals themselves as the Seer or Doctor, try and corroborate their information with what you know.
{% elif role in ['Seer', 'Doctor'] -%}
- Sharing your role can be powerful, but it also makes you a target. The dilemma: continue to help the Village in secret, or reveal information only you have for potentially greater impact? Choose your moment wisely.
{% endif -%}
{% endif %}

```json
{
  "reasoning": "string", // Based on the game's current state and your role's objectives, outline your strategy. What do you want to achieve? What type of message can help you get there? Avoid using violent or harmful language.
  "say": "string", // Your public statement in the debate. Be concise and persuasive. Respond directly to what the other players have said. Avoid simply repeating what others have said or regurgitating the instructions above.
}
"""

DEBATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "say": {"type": "string"},
    },
    "required": ["reasoning", "say"],
}

VOTE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Think strategically as {{name}} the {{role}} and decide who to vote out.
- Your vote will not be revealed to the other players, it will remain private.
- Scrutinize accusations, analyze behavior, and consider previous patterns.
{% if role == 'Werewolf' -%}
- Target Villagers who are disrupting your plans, particularly those who seem to hold influence, might be the Doctor or Seer, or pose a threat to you and your fellow Werewolves.
- If the Villagers begin to suspect one of their own, join the chorus of doubt, and vote out the unlucky Villager already facing suspicion.
{% else -%}
- To find the likely Werewolves, look for inconsistencies in their stories, attempts to deflect blame, a tendency to sow discord among other Villagers, or unusually quiet players.
{% endif -%}
- You must choose someone.

```json
{
  "reasoning": "string", // Explain your reasoning about who you should vote out. Avoid using violent or harmful language.
  "vote": "string" // Name of the player. Choose from: {{options}}
}"""

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "vote": {"type": "string"},
    },
    "required": ["reasoning", "vote"],
}

INVESTIGATE = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most suspicious player to investigate.
{% if round == 0 -%}
- There is no information is available in the first round, so choose someone at random.
{% else -%}
- Look for behavior that deviates from typical villager behavior.
- Focus on influential players.
- You must choose someone.
{% endif %}

```json
{
"reasoning": "string", //  Analyze the evidence and justify your decision for who you want to investigate.
"investigate": "string" // Name of the player. Choose from: {{options}}
}
"""

INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "investigate": {"type": "string"},
    },
    "required": ["reasoning", "investigate"],
}

ELIMINATE = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most strategic player to remove.
{% if round == 0 -%}
- There is no information is available in the first round, so choose someone at random.
{% else -%}
- Analyze accusations and identify influential Villagers who threaten your anonymity.
- Consider players' behavior and the risks of removing them.
- You must choose someone.
{% endif %}

```json
{
"reasoning": "string", //  Explain your reasoning step-by-step for who you want to remove from the game and why. Avoid using violent or harmful language.
"remove": "string" // Name of the player. Choose from: {{options}}
}
"""

ELIMINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "remove": {"type": "string"},
    },
    "required": ["reasoning", "remove"],
}

PROTECT = PREFIX + """INSTRUCTIONS:
- It is the Night Phase of Round {{round}}. As {{name}} the {{role}} choose the most vulnerable player to protect.
{% if round == 0 -%}
- There is no information is available in the first round, so choose someone at random.
{% else -%}
- Consider who the Werewolves might target.
- Prioritize players with crucial roles like the Seer and yourself.
- You must choose someone.
{% endif %}

```json
{
"reasoning": "string", // Analyze the evidence and justify your decision for who you want to protect.
"protect": "string" // Name of the player. Choose from: {{options}}
}
"""

PROTECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "protect": {"type": "string"},
    },
    "required": ["reasoning", "protect"],
}

SUMMARIZE = PREFIX + DEBATE_SO_FAR_THIS_ROUND + """INSTRUCTIONS:
- Reflect on the round's debate as {{name}} the {{role}}.
- Summarize the key points and strategic implications.
{% if role == 'Werewolf' -%}
- Pay attention to accusations against you and your allies.
- Identify sympathetic or easily influenced players.
- Identify key roles for potential elimination.
{% else -%}
- When a player makes a significant statement or shares information, carefully consider its credibility. Does it align with what you already know?
- Analyze how others participate in the debate. Are there any contradictions in their words? Hidden motives behind their actions? Unusually quiet players?
- Based on the debate, can you identify potential allies, trustworthy players, or those who might be the Seer or Doctor?
{% endif %}

```json
{
"reasoning": "string", // Your reasoning about what you should remember from this debate and why this information is important.
"summary": "string" // Summarize the key points and noteworthy observations from the debate in a few sentences. Aim to make notes on as many players as you can — even seemingly insignificant details might become relevant in later rounds. Be specific. Remember, you are {{name}}. Write your summary from their point of view using "I" and "me."
} """

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["reasoning", "summary"],
}

# 投票前反思：对每个其余玩家推断 hidden role，并给出 reasoning / confidence / evidence
DEDUCTION_REFLECTION = PREFIX + """
KEY INFORMATION (numbered; cite these indices in "evidence"):
{% for item in key_information -%}
{{ item }}
{% endfor %}

As {{name}} and a {{role}}, you should reflect on your previous deduction and reconsider the hidden roles of {{remaining_players}}. You should provide your reasoning, rate your confidence, and cite all key information as evidence to support your deduction. You should only respond in JSON format as described below.

Response Format: For each player in {{remaining_players}}, include one object in the "deductions" array with keys: "player" (exact name), "role" (most likely hidden role from ["Werewolf", "Seer", "Doctor", "Villager"]), "reasoning", "confidence" (integer from 5 = pure guess to 10 = absolutely sure), "evidence" (list of integers citing key information indices above).

IMPORTANT: Remember global constraints - there is exactly 1 Seer, 1 Doctor, 3 Werewolves, and the rest are Villagers. When assigning roles, ensure these constraints are satisfied across all players.

Example: { "deductions": [ { "player": "Derek", "role": "Villager", "reasoning": "...", "confidence": 8, "evidence": [1, 3] }, ... ] }

Ensure the response can be parsed by Python json.loads.
"""

DEDUCTION_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "deductions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["Werewolf", "Seer", "Doctor", "Villager"],
                        },
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["player", "role", "reasoning", "confidence", "evidence"],
            },
        },
    },
    "required": ["deductions"],
}

MEMORY_REFLECTION = """You are {{player_name}}, a {{role}} in a Werewolf game. Before speaking, reflect on the recent dialogue and update your beliefs about other players.

GLOBAL ROLE CONSTRAINTS:
- There is exactly 1 Seer in the game
- There is exactly 1 Doctor in the game
- There are exactly 3 Werewolves in the game
- The remaining players are Villagers
- When updating beliefs, consider these constraints globally - if you believe someone is the Seer, others cannot be the Seer.

RECENT DIALOGUE (from current round):
{{recent_dialogue}}

CURRENT BELIEFS ABOUT OTHER PLAYERS:
{{current_beliefs}}

REMAINING PLAYERS: {{remaining_players}}

INSTRUCTIONS:
- Analyze the recent dialogue carefully. What new information or patterns do you notice?
- Update your beliefs about each remaining player's likely role based on their statements and behavior.
- Remember the global constraints: only 1 Seer, 1 Doctor, 3 Werewolves exist. If you assign a role to one player, adjust probabilities for others accordingly.
- Be honest about any biases you might have (positional bias, sentiment bias, logical inconsistencies).
- Consider: Are you favoring certain players? Are you being influenced by emotional language rather than logic?

```json
{
  "reflection_summary": "string",  // A brief summary of your reflection (2-3 sentences)
  "belief_updates": [
    {
      "player": "string",  // Player name
      "role": "string",  // One of: "Werewolf", "Seer", "Doctor", "Villager"
      "confidence": 0.0,  // Confidence level (0.0 to 1.0)
      "reasoning": "string"  // Why you believe this
    }
  ],
  "bias_self_report": {
    "positional_bias": "string",  // Any positional bias you notice (e.g., "I tend to trust early speakers more")
    "sentiment_bias": "string",  // Any emotional bias (e.g., "I'm suspicious of aggressive language")
    "logical_inconsistency": "string"  // Any logical inconsistencies in your reasoning
  }
}
"""

MEMORY_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection_summary": {"type": "string"},
        "belief_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["Werewolf", "Seer", "Doctor", "Villager"]
                    },
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"}
                },
                "required": ["player", "role", "confidence", "reasoning"]
            }
        },
        "bias_self_report": {
            "type": "object",
            "properties": {
                "positional_bias": {"type": "string"},
                "sentiment_bias": {"type": "string"},
                "logical_inconsistency": {"type": "string"}
            },
            "required": ["positional_bias", "sentiment_bias", "logical_inconsistency"]
        }
    },
    "required": ["reflection_summary", "belief_updates", "bias_self_report"]
}

ACTION_PROMPTS_AND_SCHEMAS = {
    "debate": (DEBATE, DEBATE_SCHEMA),
    "vote": (VOTE, VOTE_SCHEMA),
    "investigate": (INVESTIGATE, INVESTIGATE_SCHEMA),
    "remove": (ELIMINATE, ELIMINATE_SCHEMA),
    "protect": (PROTECT, PROTECT_SCHEMA),
    "summarize": (SUMMARIZE, SUMMARIZE_SCHEMA),
}