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

"""utility functions."""

from typing import Any, Dict, Optional
import yaml
import json
import re
from abc import ABC
from abc import abstractmethod
import marko


def parse_json(text: str) -> dict[str, Any] | None:
    result_json = parse_json_markdown(text)

    if not result_json:
        result_json = parse_json_str(text)
    return result_json


def parse_json_markdown(text: str) -> dict[str, Any] | None:
    ast = marko.parse(text)

    for c in ast.children:
        # find the first json block (```json or ```JSON)
        if hasattr(c, "lang") and c.lang.lower() == "json":
            json_str = c.children[0].children
            return parse_json_str(json_str)

    return None


def parse_json_str(text: str) -> dict[str, Any] | None:
    try:
        # use yaml.safe_load which handles missing quotes around field names.
        result_json = yaml.safe_load(text)
    except yaml.parser.ParserError:
        return None

    return result_json


def parse_internal_analysis(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse and extract internal_analysis JSON from LLM output, with fallback recovery.
    
    This function attempts multiple strategies to extract the internal_analysis JSON:
    1. Try parsing the full JSON response
    2. Try extracting JSON from [Internal Analysis] section
    3. Try regex-based extraction
    4. Return default structure if all fail
    
    Args:
        raw_text: Raw text output from LLM
        
    Returns:
        Dictionary containing internal_analysis fields, or None if extraction fails
    """
    default_analysis = {
        "suspicion_score": 0.5,
        "bias_detected": {
            "positional": "",
            "sentiment": "",
            "logical_clash": ""
        },
        "memory_weight": 0.5
    }
    
    # Strategy 1: Try parsing as full JSON
    try:
        parsed = parse_json(raw_text)
        if parsed and isinstance(parsed, dict):
            if "internal_analysis" in parsed:
                return parsed["internal_analysis"]
            # Sometimes the whole response is the internal_analysis
            if "suspicion_score" in parsed and "bias_detected" in parsed:
                return parsed
    except Exception:
        pass
    
    # Strategy 2: Extract [Internal Analysis] section
    internal_section_match = re.search(
        r'\[Internal Analysis\]\s*\n?\s*(\{.*?\})',
        raw_text,
        re.DOTALL | re.IGNORECASE
    )
    if internal_section_match:
        json_str = internal_section_match.group(1)
        try:
            # Try to fix common JSON issues
            json_str = json_str.strip()
            # Remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "suspicion_score" in parsed:
                return parsed
        except Exception:
            pass
    
    # Strategy 3: Regex-based extraction of individual fields
    try:
        analysis = {}
        
        # Extract suspicion_score
        suspicion_match = re.search(
            r'"suspicion_score"\s*:\s*([0-9.]+)',
            raw_text,
            re.IGNORECASE
        )
        if suspicion_match:
            analysis["suspicion_score"] = float(suspicion_match.group(1))
        else:
            analysis["suspicion_score"] = 0.5
        
        # Extract bias_detected fields
        bias_detected = {}
        
        positional_match = re.search(
            r'"positional"\s*:\s*"([^"]*)"',
            raw_text,
            re.IGNORECASE
        )
        bias_detected["positional"] = positional_match.group(1) if positional_match else ""
        
        sentiment_match = re.search(
            r'"sentiment"\s*:\s*"([^"]*)"',
            raw_text,
            re.IGNORECASE
        )
        bias_detected["sentiment"] = sentiment_match.group(1) if sentiment_match else ""
        
        logical_match = re.search(
            r'"logical_clash"\s*:\s*"([^"]*)"',
            raw_text,
            re.IGNORECASE
        )
        bias_detected["logical_clash"] = logical_match.group(1) if logical_match else ""
        
        analysis["bias_detected"] = bias_detected
        
        # Extract memory_weight
        weight_match = re.search(
            r'"memory_weight"\s*:\s*([0-9.]+)',
            raw_text,
            re.IGNORECASE
        )
        if weight_match:
            analysis["memory_weight"] = float(weight_match.group(1))
        else:
            analysis["memory_weight"] = 0.5
        
        return analysis
    except Exception:
        pass
    
    # Strategy 4: Return default structure
    return default_analysis


class Deserializable(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[Any, Any]):
        pass
