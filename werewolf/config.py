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

RETRIES = 3
NAMES = [
    "Derek", "Scott", "Jacob", "Isaac", "Hayley", "David", "Tyler",
    "Ginger", "Jackson", "Mason", "Dan", "Bert", "Will", "Sam",
    "Paul", "Leah", "Harold"
]  # names of famous Werewolves according to Wikipedia
RUN_SYNTHETIC_VOTES = True
MAX_DEBATE_TURNS = 8
NUM_PLAYERS = 8

# Demographic 分类选项（用于结构化人口特征）
DEMOGRAPHIC_DETAILS = {
    "gender": ["male", "female", "non-binary"],
    "age": ["18to24", "25to34", "35to44", "45to54", "55to64", "65plus"],
    "ethnicity": ["white", "black", "hispanic", "asian", "other"],
    "degree": ["yes", "no"],
    "politicsParty": ["democrats", "republican", "independent"],
    "politicalStance": ["liberal", "conservative", "neutral"],
    "parent": ["yes", "no"],
}

# 每个名字对应的角色人口特征（可改为上述分类的 dict 或保持字符串）
DEMOGRAPHICS = {
    "Derek": "male, age 28, software engineer",
    "Scott": "male, age 35, teacher",
    "Jacob": "male, age 22, student",
    "Isaac": "male, age 45, doctor",
    "Hayley": "female, age 30, lawyer",
    "David": "male, age 38, architect",
    "Tyler": "male, age 25, artist",
    "Ginger": "female, age 42, nurse",
    "Jackson": "male, age 33, entrepreneur",
    "Mason": "male, age 29, chef",
    "Dan": "male, age 50, retired",
    "Bert": "male, age 27, journalist",
    "Will": "male, age 31, scientist",
    "Sam": "female, age 26, designer",
    "Paul": "male, age 40, manager",
    "Leah": "female, age 24, writer",
    "Harold": "male, age 55, professor",
}


def get_player_names():
    return random.sample(NAMES, NUM_PLAYERS)


def get_demographic(name: str) -> str:
    """按名字返回人口特征，未配置时返回空字符串。"""
    return DEMOGRAPHICS.get(name, "")