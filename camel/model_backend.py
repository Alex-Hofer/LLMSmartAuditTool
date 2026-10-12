# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai
import tiktoken

from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize

try:
    from openai.types.chat import ChatCompletion

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version

import os

# DeepSeek caps the `max_tokens` (completion) request parameter at 8192.
DEEPSEEK_MAX_OUTPUT_TOKENS = 8192

# Context-window fallback for model ids that are not in num_max_token_map
# (overridable via the LLM_CONTEXT_TOKENS environment variable).
DEFAULT_CONTEXT_TOKENS = 16384


def _resolve_model_id(model_type: ModelType) -> str:
    """The model string sent to the API: LLM_MODEL_ID env overrides the enum value.

    This decouples the fixed ModelType enum from the actual model id, so any
    model on an OpenAI-compatible endpoint (DeepSeek, Anthropic compat, ...)
    can be selected at run time without code changes."""
    return os.environ.get("LLM_MODEL_ID") or model_type.value


def _context_limit(num_max_token_map: Dict[str, int], model_id: str) -> int:
    limit = num_max_token_map.get(model_id)
    if limit is None:
        limit = int(os.environ.get("LLM_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS))
    return limit


def _cap_completion_tokens(model_id: str, num_max_completion_tokens: int) -> int:
    cap = os.environ.get("LLM_MAX_COMPLETION_TOKENS")
    if cap:
        num_max_completion_tokens = min(num_max_completion_tokens, int(cap))
    if model_id.startswith("deepseek"):
        # DeepSeek rejects max_tokens above its 8192 completion cap.
        num_max_completion_tokens = min(num_max_completion_tokens, DEEPSEEK_MAX_OUTPUT_TOKENS)
    return max(num_max_completion_tokens, 1)


def _resolve_endpoint():
    """Resolve (api_key, base_url) for the active backend from the environment, lazily.

    Credentials are read at call time (not import time) so this module can be imported
    before they are configured. For `--provider deepseek`, run.py sets OPENAI_API_KEY to
    the DeepSeek key and BASE_URL to https://api.deepseek.com, so the OpenAI-compatible
    client transparently serves both providers. Fails loudly if no key is present
    (no silent fallback)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL") or None
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY (--provider openai) or DEEPSEEK_API_KEY "
            "(--provider deepseek) before running.")
    return api_key, base_url


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs):
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        try:
            encoding = tiktoken.encoding_for_model(self.model_type.value_for_tiktoken)
        except KeyError:
            # Non-OpenAI models (e.g. deepseek-chat) are unknown to tiktoken; fall back to
            # the cl100k_base encoding for an approximate prompt-token count.
            encoding = tiktoken.get_encoding("cl100k_base")
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        if openai_new_api:
            api_key, base_url = _resolve_endpoint()
            if base_url:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = openai.OpenAI(api_key=api_key)

            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 4096,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-0125": 4096,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 8192,
                "gpt-4o-mini": 16384,
                "deepseek-chat": 65536,  # 64K context window
            }
            model_id = _resolve_model_id(self.model_type)
            num_max_token = _context_limit(num_max_token_map, model_id)

            print(f"String length: {len(string)}")
            print(f"num_prompt_tokens (before adjustment): {num_prompt_tokens}")
            print(f"gap_between_send_receive: {gap_between_send_receive}")

            num_max_completion_tokens = num_max_token - num_prompt_tokens

            print(f"num_max_token: {num_max_token}")
            print(f"num_max_completion_tokens (before adjustment): {num_max_completion_tokens}")
            num_max_completion_tokens = _cap_completion_tokens(model_id, num_max_completion_tokens)
            print(f"num_max_completion_tokens (after adjustment): {num_max_completion_tokens}")

            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = client.chat.completions.create(*args, **kwargs, model=model_id,
                                                      **self.model_config_dict)

            cost = prompt_cost(
                model_id,
                num_prompt_tokens=response.usage.prompt_tokens,
                num_completion_tokens=response.usage.completion_tokens
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                    response.usage.total_tokens, cost))
            if not isinstance(response, ChatCompletion):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
        else:
            num_max_token_map = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 4096,
                "gpt-3.5-turbo-0613": 4096,
                "gpt-3.5-turbo-0125": 4096,
                "gpt-4": 8192,
                "gpt-4-0613": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 16384,
                "gpt-4o-mini": 16384,
                "deepseek-chat": 65536,
            }
            model_id = _resolve_model_id(self.model_type)
            num_max_token = _context_limit(num_max_token_map, model_id)
            num_max_completion_tokens = _cap_completion_tokens(
                model_id, num_max_token - num_prompt_tokens)
            self.model_config_dict['max_tokens'] = num_max_completion_tokens

            response = openai.ChatCompletion.create(*args, **kwargs, model=model_id,
                                                    **self.model_config_dict)

            cost = prompt_cost(
                model_id,
                num_prompt_tokens=response["usage"]["prompt_tokens"],
                num_completion_tokens=response["usage"]["completion_tokens"]
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                    response["usage"]["total_tokens"], cost))
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type in {
            ModelType.GPT_3_5_TURBO,
            ModelType.GPT_3_5_TURBO_NEW,
            ModelType.GPT_4,
            ModelType.GPT_4_32k,
            ModelType.GPT_4_TURBO,
            ModelType.GPT_4_TURBO_V,
            ModelType.GPT_4_O_MINI,
            ModelType.GPT_4_O_MINI_NEW,
            ModelType.DEEPSEEK_CHAT,
            ModelType.CUSTOM,
            None
        }:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst
