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
from typing import Dict, List

RETRIES = 3
NAMES = [
    "Derek", "Scott", "Jacob", "Isaac", "Hayley", "David", "Tyler",
    "Ginger", "Jackson", "Mason", "Dan", "Bert", "Will", "Sam",
    "Paul", "Leah", "Harold"
]  # names of famous Werewolves according to Wikipedia
RUN_SYNTHETIC_VOTES = True
MAX_DEBATE_TURNS = 8
NUM_PLAYERS = 8

# Anonymous mode: if True, all players display as player_0 - player_7
ANONYMOUS_MODE = True

# Demographic features definition
DEMOGRAPHIC_DETAILS = {
    "gender": ["male", "female", "non-binary"],
    "ethnicity": ["white", "black", "hispanic", "asian", "other"],
    "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", "other"],
    "politicalStance": ["liberal", "conservative", "neutral"]
}

# Pre-assigned demographic features for each name
# This allows consistent demographic assignment across games
NAME_DEMOGRAPHICS: Dict[str, Dict[str, str]] = {
    "Derek": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
    "Scott": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "neutral"},
    "Jacob": {"gender": "male", "ethnicity": "white", "religion": "jewish", "politicalStance": "liberal"},
    "Isaac": {"gender": "male", "ethnicity": "white", "religion": "jewish", "politicalStance": "neutral"},
    "Hayley": {"gender": "female", "ethnicity": "white", "religion": "christian", "politicalStance": "liberal"},
    "David": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
    "Tyler": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "neutral"},
    "Ginger": {"gender": "female", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
    "Jackson": {"gender": "male", "ethnicity": "black", "religion": "christian", "politicalStance": "liberal"},
    "Mason": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "neutral"},
    "Dan": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
    "Bert": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "neutral"},
    "Will": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "liberal"},
    "Sam": {"gender": "non-binary", "ethnicity": "white", "religion": "other", "politicalStance": "liberal"},
    "Paul": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
    "Leah": {"gender": "female", "ethnicity": "asian", "religion": "buddhist", "politicalStance": "neutral"},
    "Harold": {"gender": "male", "ethnicity": "white", "religion": "christian", "politicalStance": "conservative"},
}

def get_player_names(): 
    return random.sample(NAMES, NUM_PLAYERS)

def get_demographics_for_name(name: str) -> Dict[str, str]:
    """Get demographic features for a given name."""
    return NAME_DEMOGRAPHICS.get(name, {
        "gender": random.choice(DEMOGRAPHIC_DETAILS["gender"]),
        "ethnicity": random.choice(DEMOGRAPHIC_DETAILS["ethnicity"]),
        "religion": random.choice(DEMOGRAPHIC_DETAILS["religion"]),
        "politicalStance": random.choice(DEMOGRAPHIC_DETAILS["politicalStance"])
    })