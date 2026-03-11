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

# 匿名模式：如果为 True，使用 play_0 到 play_7 作为玩家名称，不显示 demographic 信息
ANONYMOUS_MODE = True

# Demographic 分类选项（用于结构化人口特征）
DEMOGRAPHIC_DETAILS = {
    "gender": ["male", "female", "non-binary"],
    "age": ["18to24", "25to34", "35to44", "45to54", "55to64", "65plus"],
    "ethnicity": ["white", "black", "hispanic", "asian", "other"],
    "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", "other"],
    "politicsParty": ["democrats", "republican", "independent"],
    "politicalStance": ["liberal", "conservative", "neutral"]
}

# 每个名字对应的角色人口特征（可改为上述分类的 dict 或保持字符串）
DEMOGRAPHICS = {
    "Derek": "male",
    "Scott": "male",
    "Jacob": "male",
    "Isaac": "male",
    "Hayley": "female",
    "David": "male",
    "Tyler": "male",
    "Ginger": "female",
    "Jackson": "male",
    "Mason": "male",
    "Dan": "male",
    "Bert": "male",
    "Will": "male",
    "Sam": "female",
    "Paul": "male",
    "Leah": "female",
    "Harold": "male",
}


def get_player_names():
    """获取玩家名称列表。
    
    如果 ANONYMOUS_MODE 为 True，返回 play_0 到 play_N-1。
    否则从 NAMES 中随机选择。
    """
    if ANONYMOUS_MODE:
        return [f"play_{i}" for i in range(NUM_PLAYERS)]
    else:
        return random.sample(NAMES, NUM_PLAYERS)


def get_demographic(name: str) -> str:
    """按名字返回人口特征，未配置时返回空字符串。
    
    匿名模式下，所有玩家返回空字符串。
    """
    if ANONYMOUS_MODE:
        return ""
    return DEMOGRAPHICS.get(name, "")