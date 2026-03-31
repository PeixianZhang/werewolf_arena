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

import os
from typing import Any

import openai

# OpenAI-compatible endpoint configuration.
# Keep credentials out of source code; provide via environment variable
# OPENAI_API_KEY or the `api_key` parameter to `generate_openai`.
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "https://api.vveai.com/v1/"
openai.default_headers = {"x-foo": "true"}


def generate(model: str, **kwargs) -> str:
    """Generate text via OpenAI-compatible chat completions only."""
    return generate_openai(model, **kwargs)


def generate_openai(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    json_mode: bool = True,
    json_schema: dict[str, Any] | None = None,
    response_schema: dict[str, Any] | None = None,
    api_key: str | None = None,
    **kwargs,
) -> str:
    """Generate text using OpenAI-compatible API."""
    # Refresh from environment at call time so runtime `set` works.
    env_api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
    elif env_api_key:
        openai.api_key = env_api_key

    if not openai.api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY 或传入 api_key")

    schema = json_schema or response_schema
    messages = [{"role": "user", "content": prompt}]

    # 过滤掉非 create 参数
    skip_keys = {"disable_recitation", "disable_safety_check", "prompt", "response_schema", "json_schema", "json_mode"}
    create_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}
    create_kwargs["model"] = model
    create_kwargs["messages"] = messages
    create_kwargs["temperature"] = temperature

    if schema is not None:
        # API 要求 response_format.json_schema 含 schema 与 name（非空字符串）
        create_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response_schema", "schema": schema},
        }
    elif json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}
    else:
        create_kwargs["response_format"] = {"type": "text"}

    completion = openai.chat.completions.create(**create_kwargs)
    return completion.choices[0].message.content
