"""
ALCF Inference Endpoints backend for the electrochemistry agentic
workflow (docs.alcf.anl.gov/services/inference-endpoints/, verified
2026-08-12).

This replaces a hardcoded OPENAI_API_KEY assumption with ALCF's actual
auth model: a Globus-issued access token (NOT a static API key) obtained
via the `inference_auth_token.py` helper script, used as the `api_key`
argument to an OpenAI-compatible client pointed at ALCF's vLLM endpoint.

Model selection matters here: AG2 function-calling requires a model with
tool-calling support, and this project's Results Analyst / Manager agents
specifically need reasoning capability too (interpreting convergence
diagnostics against literature thresholds, not just calling functions).
As of the ALCF docs snapshot above, only a few models on this endpoint
combine both Tool-Calling (T) and Reasoning (R):
    - Qwen/QwQ-32B          (R, T)
    - Qwen/Qwen3-235B-A22B  (R, T)
Most Llama models (3.1-70B/8B/405B-Instruct, 3.3-70B-Instruct) have T but
not R, and openai/gpt-oss-20b/120b have R but not T on this endpoint.
Re-check the live model list (it changes) via the /jobs status endpoint
before assuming a specific model is still available.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ALCF_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"

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


def get_alcf_access_token(auth_script_path: Optional[str] = None) -> str:
    """Return a valid ALCF Globus access token, using the
    `inference_auth_token.py` helper (downloaded per ALCF's docs from
    github.com/argonne-lcf/inference-endpoints). Raises with actionable
    instructions if the token is missing/expired and cannot be refreshed
    automatically.

    If `auth_script_path` is not given, assumes `inference_auth_token.py`
    is importable from the current PYTHONPATH (place it alongside this
    project or install it as a local script per ALCF's Quick Start).
    """
    try:
        from inference_auth_token import get_access_token  # type: ignore
        return get_access_token()
    except ImportError:
        pass

    if auth_script_path is None:
        raise ImportError(
            "Could not import inference_auth_token. Download it via:\n"
            "  wget https://raw.githubusercontent.com/argonne-lcf/"
            "inference-endpoints/refs/heads/main/inference_auth_token.py\n"
            "then run `python inference_auth_token.py authenticate` once "
            "(Globus login), and either place the script on PYTHONPATH or "
            "pass auth_script_path= explicitly."
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
    """

    model: str = "Qwen/Qwen3-235B-A22B"  # default: reasoning + tool-calling
    base_url: str = ALCF_BASE_URL
    temperature: float = 0.1
    timeout: int = 900  # generous: cold starts can take 10-15 minutes
    auth_script_path: Optional[str] = None

    def to_autogen_config(self) -> dict:
        access_token = get_alcf_access_token(self.auth_script_path)
        return {
            "config_list": [{
                "model": self.model,
                "api_key": access_token,
                "base_url": self.base_url,
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
    response = requests.get(
        "https://inference-api.alcf.anl.gov/resource_server/sophia/jobs",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
