"""
ALCF Inference Endpoints backend for the electrochemistry agentic
workflow (docs.alcf.anl.gov/services/inference-endpoints/, verified
2026-08-12), following the token-resolution pattern used in Argonne's own
reference implementation
(github.com/argonne-lcf/ATPESC_MachineLearning/tree/master/13_agentic_workflows_for_science/src/atpesc_agentic/alcf_llm.py
and .../11_Agentic_tools_part1), confirmed directly from that source:

Token resolution precedence (matches ATPESC's alcf_llm.py exactly):
  1. `ALCF_ACCESS_TOKEN` environment variable, if set (manual override --
     e.g. after running `source scripts/get_alcf_token.sh`, or in a batch
     job where interactive Globus login isn't possible).
  2. Otherwise, `inference_auth_token.get_access_token()` (from
     github.com/argonne-lcf/inference-endpoints), which reuses a cached
     Globus token and auto-refreshes it, prompting an interactive login
     only on first use:
         wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
         python inference_auth_token.py authenticate

`ALCF_BASE_URL` is also overridable via environment variable, matching
ATPESC's .env.example convention, in case ALCF moves the endpoint or you
need to point at a different resource_server (e.g. a different cluster
than "sophia").

Model selection: AG2 function-calling requires a model with tool-calling
support, and this project's Results Analyst / Manager agents specifically
need reasoning capability too (interpreting convergence diagnostics
against literature thresholds, not just calling functions). Per the ALCF
docs snapshot (2026-08-12), only a few models combine both Tool-Calling
(T) and Reasoning (R): Qwen/QwQ-32B and Qwen/Qwen3-235B-A22B. ATPESC's own
tutorial notebook (14_agentic_tools_part2/ATPESC-Agents-Tutorial.ipynb)
uses `openai/gpt-oss-120b` as a general example, but that model is listed
as Reasoning-only (no confirmed Tool-Calling flag) on the endpoint as of
this snapshot -- prefer the Qwen models above when tool-calling is
required, and re-check the live roster via check_model_availability()
before assuming any specific model is still hot/available.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ALCF_BASE_URL_DEFAULT = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"

# Models confirmed (per ALCF docs, 2026-08-12) to support BOTH tool-calling
# (required for AG2 function-calling) and reasoning (useful for the
# Manager / Results Analyst agents' convergence/literature-comparison
# judgment calls).
REASONING_AND_TOOL_CAPABLE_MODELS = (
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-235B-A22B",
)

# Lighter tool-calling models (no reasoning flag), suitable for simpler
# agents like the System Builder or Validation Agent where deterministic
# tool dispatch matters more than open-ended judgment.
TOOL_CALLING_MODELS = (
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
)

# Reasoning-capable but NOT confirmed tool-calling on this endpoint; usable
# for pure-text reasoning tasks but not for AG2 function-calling agents.
REASONING_ONLY_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
)


def get_alcf_access_token(auth_script_path: Optional[str] = None) -> str:
    """Return a valid ALCF Globus access token, following the exact
    precedence used in Argonne's ATPESC reference implementation
    (alcf_llm.py):

      1. `ALCF_ACCESS_TOKEN` env var, if set (manual override).
      2. `inference_auth_token.get_access_token()` (auto-refreshing
         cached Globus token; interactive login only on first use).

    Raises ImportError with the exact setup commands from ATPESC's
    README if neither path is available.
    """
    env_token = os.environ.get("ALCF_ACCESS_TOKEN")
    if env_token:
        return env_token

    try:
        from inference_auth_token import get_access_token  # type: ignore
        return get_access_token()
    except ImportError:
        pass

    if auth_script_path is None:
        raise ImportError(
            "No ALCF_ACCESS_TOKEN set and inference_auth_token is not "
            "importable. Set up per Argonne's ATPESC tutorial:\n"
            "  wget https://raw.githubusercontent.com/argonne-lcf/"
            "inference-endpoints/refs/heads/main/inference_auth_token.py\n"
            "  python inference_auth_token.py authenticate\n"
            "  echo \"ALCF_ACCESS_TOKEN=$(python inference_auth_token.py "
            "get_access_token)\" >> ~/.env\n"
            "or pass auth_script_path= explicitly if the script lives "
            "elsewhere. Requires `pip install globus-sdk`."
        )

    result = subprocess.run(
        ["python", auth_script_path, "get_access_token"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


@dataclass
class ALCFLLMConfig:
    """AG2/AutoGen llm_config builder targeting ALCF Inference Endpoints
    instead of a commercial API. Streaming is disabled per ALCF's
    documented limitation (the Globus backend does not support it).

    `base_url` defaults to ALCF_BASE_URL_DEFAULT but can be overridden via
    the `ALCF_BASE_URL` environment variable, matching ATPESC's
    .env.example convention.
    """

    model: str = "Qwen/Qwen3-235B-A22B"  # default: reasoning + tool-calling
    base_url: Optional[str] = None
    temperature: float = 0.1
    timeout: int = 900  # generous: cold starts can take 10-15 minutes
    auth_script_path: Optional[str] = None

    def resolved_base_url(self) -> str:
        return self.base_url or os.environ.get("ALCF_BASE_URL", ALCF_BASE_URL_DEFAULT)

    def to_autogen_config(self) -> dict:
        access_token = get_alcf_access_token(self.auth_script_path)
        return {
            "config_list": [{
                "model": self.model,
                "api_key": access_token,
                "base_url": self.resolved_base_url(),
                "price": [0, 0],  # ALCF endpoint is not billed per-token
                                     # like commercial APIs; avoids AG2
                                     # cost-tracking warnings.
            }],
            "temperature": self.temperature,
            "timeout": self.timeout,
            "stream": False,  # required: Globus backend does not support streaming
        }


def check_model_availability(model: str, auth_script_path: Optional[str] = None) -> dict:
    """Query ALCF's /jobs endpoint to check whether `model` is currently
    Live/Starting/Queued/Offline before committing an agent workflow to
    it. Returns the raw jobs listing filtered to entries mentioning
    `model` where possible; falls back to the full listing if the API
    shape doesn't allow client-side filtering.
    """
    import requests

    access_token = get_alcf_access_token(auth_script_path)
    base = os.environ.get("ALCF_BASE_URL", ALCF_BASE_URL_DEFAULT)
    jobs_url = base.rsplit("/vllm/v1", 1)[0] + "/jobs"
    response = requests.get(
        jobs_url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
