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
import argparse
import logging
import os
import sys

from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


def _fail(message):
    """Exit with a clear, single-line error (no silent fallback)."""
    sys.exit("ERROR: " + message)


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--config', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")
parser.add_argument('--org', type=str, default="DefaultOrganization",
                    help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
                    help="Prompt of software")
parser.add_argument('--name', type=str, default="Gomoku",
                    help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--provider', type=str, default="openai", choices=["openai", "deepseek", "custom"],
                    help="LLM backend provider. 'openai' (default) uses the OpenAI API; "
                         "'deepseek' routes the OpenAI-compatible client to https://api.deepseek.com "
                         "(default model deepseek-chat, override with --model); "
                         "'custom' uses any OpenAI-compatible endpoint given by the BASE_URL "
                         "environment variable (key from OPENAI_API_KEY, model from --model).")
parser.add_argument('--model', type=str, default="GPT_4_O_MINI",
                    help="Model name. For provider 'openai': an alias from "
                         "{'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4_O_MINI'} or a literal "
                         "OpenAI model id. For provider 'deepseek': a literal model id "
                         "(default deepseek-chat). For provider 'custom': a literal model id (required).")
parser.add_argument('--task-file', dest='task_file', type=str, default="",
                    help="Path to a file whose contents are used as --task (e.g. a .sol contract). "
                         "Robust for large contracts where passing source via --task is impractical. "
                         "Overrides --task when set.")
parser.add_argument('--path', type=str, default="",
                    help="Your file directory, ChatDev will build upon your software in the Incremental mode")
args = parser.parse_args()

# Load the task (contract source) from a file when --task-file is given.
if args.task_file:
    if not os.path.isfile(args.task_file):
        _fail("--task-file '{}' does not exist.".format(args.task_file))
    with open(args.task_file, "r", encoding="utf-8") as _tf:
        args.task = _tf.read()

# ----------------------------------------
#   Provider / credential wiring
#   MUST happen before importing chatdev.chat_chain: some modules (e.g. camel.web_spider)
#   construct an OpenAI client at import time, which needs OPENAI_API_KEY (and BASE_URL).
# ----------------------------------------
# The model id actually sent to the API is decoupled from the ModelType enum:
# when LLM_MODEL_ID is set, camel/model_backend.py uses it verbatim. Clear any
# stale value first; the branches below set it as needed.
os.environ.pop("LLM_MODEL_ID", None)
# Report label for chatdev/chat_chain.py (keeps the audit report truthful).
os.environ["LLM_PROVIDER"] = args.provider

# The --model default is an OpenAI alias; treat it as "not specified" for the
# non-openai providers.
_model_specified = args.model and args.model != parser.get_default("model")

if args.provider == "deepseek":
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_key:
        _fail("--provider deepseek requires the DEEPSEEK_API_KEY environment variable to be set.")
    # DeepSeek is OpenAI-API compatible: reuse the OpenAI SDK with a custom base_url.
    # Downstream code reads OPENAI_API_KEY + BASE_URL, so map the DeepSeek key onto them.
    os.environ["OPENAI_API_KEY"] = deepseek_key
    os.environ["BASE_URL"] = os.environ.get("BASE_URL") or "https://api.deepseek.com"
    if _model_specified and args.model != ModelType.DEEPSEEK_CHAT.value:
        # Honor the requested DeepSeek model (e.g. deepseek-reasoner).
        os.environ["LLM_MODEL_ID"] = args.model
    selected_model_type = ModelType.DEEPSEEK_CHAT
elif args.provider == "custom":
    # Any OpenAI-compatible endpoint (e.g. Anthropic's compatibility layer).
    if not os.environ.get("BASE_URL"):
        _fail("--provider custom requires the BASE_URL environment variable "
              "(the OpenAI-compatible endpoint URL).")
    if not os.environ.get("OPENAI_API_KEY"):
        _fail("--provider custom requires the OPENAI_API_KEY environment variable "
              "(the API key for the custom endpoint).")
    if not _model_specified:
        _fail("--provider custom requires an explicit --model (a literal model id).")
    os.environ["LLM_MODEL_ID"] = args.model
    selected_model_type = ModelType.CUSTOM
else:  # openai
    if not os.environ.get("OPENAI_API_KEY"):
        _fail("--provider openai requires the OPENAI_API_KEY environment variable to be set.")
    selected_model_type = None  # resolved below from args.model

# Now it is safe to import modules that build an LLM client at import time.
from chatdev.chat_chain import ChatChain

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")

# ----------------------------------------
#          Init ChatChain
# ----------------------------------------
config_path, config_phase_path, config_role_path = get_config(args.config)
args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
             'GPT_4': ModelType.GPT_4,
             #  'GPT_4_32K': ModelType.GPT_4_32k,
             'GPT_4_TURBO': ModelType.GPT_4_TURBO,
             #  'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V,
             'GPT_4_O_MINI': ModelType.GPT_4_O_MINI
             }
if openai_new_api:
    args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW
    args2type['GPT_4_O_MINI'] = ModelType.GPT_4_O_MINI_NEW

if args.provider == "openai":
    if args.model in args2type:
        selected_model_type = args2type[args.model]
    else:
        # Not a known alias: treat it as a literal OpenAI model id
        # (e.g. gpt-4o, o3-mini) sent verbatim via LLM_MODEL_ID.
        os.environ["LLM_MODEL_ID"] = args.model
        selected_model_type = ModelType.CUSTOM

chat_chain = ChatChain(config_path=config_path,
                       config_phase_path=config_phase_path,
                       config_role_path=config_role_path,
                       task_prompt=args.task,
                       project_name=args.name,
                       org_name=args.org,
                       model_type=selected_model_type,
                       code_path=args.path)

# ----------------------------------------
#          Init Log
# ----------------------------------------
logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------

chat_chain.pre_processing()

# ----------------------------------------
#          Personnel Recruitment
# ----------------------------------------

chat_chain.make_recruitment()

# ----------------------------------------
#          Chat Chain
# ----------------------------------------

chat_chain.execute_chain()

# ----------------------------------------
#          Post Processing
# ----------------------------------------

chat_chain.post_processing()
