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

# 使用模块级配置，与 openai.chat.completions.create 一致
openai.api_key = "sk-NMIm91dG9OKhGYlY18B4B139466b4aBdAf45A7870cF1D152"
openai.base_url = "https://api.vveai.com/v1/"
openai.default_headers = {"x-foo": "true"}


def generate(model: str, **kwargs) -> str:
    """Generate text using OpenAI GPT. All calls use OpenAI only."""
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
    """Generate text using OpenAI API (GPT)，调用方式与模板一致。"""

    if not openai.api_key and not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY 或传入 api_key")
    if api_key:
        openai.api_key = api_key

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
