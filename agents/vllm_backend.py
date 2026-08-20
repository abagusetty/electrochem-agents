"""
Local vLLM-XPU agent backend: serve the LLM on the Aurora allocation itself.

WHY THIS REPLACES THE ALCF INFERENCE ENDPOINT AS THE DEFAULT
-------------------------------------------------------------
The `frameworks` module ships `vllm 0.26.1+xpu` and `vllm-xpu-kernels`, so the
agent's LLM can run on the same nodes as the science. That removes the three
things that made the remote-endpoint path fragile for a long campaign:

  1. NO OUTBOUND NETWORK NEEDED. Aurora compute nodes generally cannot reach
     the internet. A remote endpoint call from inside a job either fails or
     hangs on a socket timeout -- inside a 6-hour allocation, silently.
  2. NO INTERACTIVE LOGIN. `inference_auth_token.py authenticate` is a
     browser-based Globus flow that nothing in a batch job can perform, and
     tokens expire mid-campaign.
  3. NO SHARED-SERVICE CONTENTION. Endpoint models go hot/cold and are queued
     behind other users; a local server's latency is yours alone.

Same shape as Argonne's own Aurora multi-agent work (arXiv:2604.07681:
planner-executor with gpt-oss-120b + MCP + Parsl on Aurora).

Cost: GPU tiles spent on inference instead of MLIP/DFT. Budget for it --
a served 120B model wants a meaningful share of a node.

TOOL CALLING IS NOT ON BY DEFAULT
---------------------------------
These agents must invoke real functions (`find_local_abnormalities`,
`distill_patterns`, JDFTx drivers). vLLM does NOT enable tool calling unless
you pass BOTH `--enable-auto-tool-choice` AND a `--tool-call-parser` matching
the model's template. Get the parser wrong and the model emits tool calls as
prose, the agent appears to "work", and nothing is ever actually executed.
`TOOL_CALL_PARSERS` below maps the common families; verify against
`vllm serve --help` for your build.

Nothing in this module has been run.
"""

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

# Tool-call parser by model family. vLLM ships parsers per chat template; the
# name must match the model's actual template, not its vendor.
TOOL_CALL_PARSERS: Dict[str, str] = {
    "qwen": "hermes",
    "hermes": "hermes",
    "llama-3.1": "llama3_json",
    "llama-3.2": "llama3_json",
    "llama-3.3": "llama3_json",
    "mistral": "mistral",
    "gpt-oss": "openai",
    "granite": "granite",
    "deepseek": "deepseek_v3",
}


def guess_tool_call_parser(model: str) -> Optional[str]:
    """Best-effort parser for `model`. Returns None when unknown.

    None is deliberate: a wrong parser is worse than none, because the server
    starts, the model emits tool calls as text, and the agents look functional
    while executing nothing.
    """
    lowered = model.lower()
    for key, parser in TOOL_CALL_PARSERS.items():
        if key in lowered:
            return parser
    return None


@dataclass
class VLLMServerConfig:
    """How to serve one model on this allocation."""

    model: str                                  # local path, or HF id if staged
    served_model_name: Optional[str] = None     # name clients use; defaults to model
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = "xpu"
    # One rank per PVC tile. A 120B model will not fit on one tile; size this
    # to the model and remember every tile here is a tile not doing science.
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = 32768
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"
    trust_remote_code: bool = False

    # Tool calling. `auto` resolves via guess_tool_call_parser.
    enable_tool_calling: bool = True
    tool_call_parser: Optional[str] = "auto"
    chat_template: Optional[str] = None

    api_key: str = "EMPTY"                      # local server; not a secret
    extra_args: List[str] = field(default_factory=list)
    log_path: Optional[str] = None
    startup_timeout_s: float = 1800.0           # large models load slowly

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    @property
    def client_model_name(self) -> str:
        return self.served_model_name or self.model

    def resolved_tool_parser(self) -> Optional[str]:
        if not self.enable_tool_calling:
            return None
        if self.tool_call_parser and self.tool_call_parser != "auto":
            return self.tool_call_parser
        return guess_tool_call_parser(self.model)

    def command(self) -> List[str]:
        """The `vllm serve ...` argv."""
        cmd = ["vllm", "serve", self.model,
               "--host", self.host, "--port", str(self.port),
               "--device", self.device,
               "--dtype", self.dtype,
               "--tensor-parallel-size", str(self.tensor_parallel_size),
               "--gpu-memory-utilization", str(self.gpu_memory_utilization),
               "--api-key", self.api_key]
        if self.served_model_name:
            cmd += ["--served-model-name", self.served_model_name]
        if self.pipeline_parallel_size > 1:
            cmd += ["--pipeline-parallel-size", str(self.pipeline_parallel_size)]
        if self.max_model_len:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.chat_template:
            cmd += ["--chat-template", self.chat_template]

        parser = self.resolved_tool_parser()
        if self.enable_tool_calling:
            if parser is None:
                raise ValueError(
                    f"enable_tool_calling=True but no tool-call parser is known "
                    f"for model {self.model!r}. Set tool_call_parser= explicitly "
                    f"(one of {sorted(set(TOOL_CALL_PARSERS.values()))}, or check "
                    "`vllm serve --help` for your build). Refusing to start a "
                    "server whose agents would silently fail to invoke tools."
                )
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
        return cmd + list(self.extra_args)

    def server_env(self) -> Dict[str, str]:
        """Environment for the server process.

        Offline flags are not paranoia: compute nodes have no outbound network,
        so an un-staged model would hang on a socket timeout rather than fail.
        """
        return {
            "VLLM_TARGET_DEVICE": "xpu",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ZE_FLAT_DEVICE_HIERARCHY": os.environ.get(
                "ZE_FLAT_DEVICE_HIERARCHY", "FLAT"),
            "ONEAPI_DEVICE_SELECTOR": os.environ.get(
                "ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"base_url": self.base_url,
                  "resolved_tool_parser": self.resolved_tool_parser()})
        return d


class VLLMServer:
    """Start / health-check / stop a local vLLM server.

    Use as a context manager so the server dies with the driver even if the
    campaign raises -- an orphaned server holds GPU tiles for the rest of the
    allocation.
    """

    def __init__(self, config: VLLMServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._log_handle = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "VLLMServer":
        if self.is_healthy():
            # Someone already serves this port -- reuse rather than fight it.
            return self

        cmd = self.config.command()
        env = dict(os.environ)
        env.update(self.config.server_env())

        log_path = Path(self.config.log_path or "vllm_server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(log_path, "w")
        self._log_handle.write(f"# {' '.join(shlex.quote(c) for c in cmd)}\n")
        self._log_handle.flush()

        self.process = subprocess.Popen(
            cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, env=env)
        return self

    def wait_until_ready(self, timeout_s: Optional[float] = None,
                         poll_s: float = 5.0) -> "VLLMServer":
        timeout_s = timeout_s or self.config.startup_timeout_s
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {self.process.returncode} "
                    f"during startup. See {self.config.log_path or 'vllm_server.log'}. "
                    "Common causes on XPU: model not staged locally (no network "
                    "on compute nodes), tensor-parallel-size larger than the "
                    "visible tile count, or an unsupported dtype."
                )
            if self.is_healthy():
                return self
            time.sleep(poll_s)
        raise TimeoutError(
            f"vLLM server not healthy after {timeout_s:.0f}s at "
            f"{self.config.health_url}. Large models genuinely take minutes to "
            "load; raise startup_timeout_s before assuming failure."
        )

    def is_healthy(self) -> bool:
        try:
            import requests
            return requests.get(self.config.health_url, timeout=5).status_code == 200
        except Exception:                                       # noqa: BLE001
            return False

    def list_models(self) -> Dict[str, Any]:
        import requests
        response = requests.get(
            f"http://{self.config.host}:{self.config.port}/v1/models",
            headers={"Authorization": f"Bearer {self.config.api_key}"}, timeout=30)
        response.raise_for_status()
        return response.json()

    def verify_tool_calling(self) -> Dict[str, Any]:
        """Prove the server actually emits a structured tool call.

        Worth the one round-trip. A server with a wrong or missing
        `--tool-call-parser` starts cleanly, answers chat requests, and returns
        tool invocations as prose -- so the agents run, produce plausible
        transcripts, and never execute a single function.
        """
        import requests

        probe_tool = {
            "type": "function",
            "function": {
                "name": "get_barrier",
                "description": "Return the CO dimerization barrier for a facet.",
                "parameters": {
                    "type": "object",
                    "properties": {"facet": {"type": "string"}},
                    "required": ["facet"],
                },
            },
        }
        payload = {
            "model": self.config.client_model_name,
            "messages": [{"role": "user",
                          "content": "What is the barrier on Cu(310)? Use the tool."}],
            "tools": [probe_tool],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 256,
        }
        response = requests.post(
            f"{self.config.base_url}/chat/completions", json=payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"}, timeout=300)
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        return {
            "tool_calls_returned": len(calls),
            "ok": bool(calls),
            "parser": self.config.resolved_tool_parser(),
            "first_call": calls[0] if calls else None,
            "content_preview": (message.get("content") or "")[:200],
            "diagnosis": ("ok" if calls else
                          "NO STRUCTURED TOOL CALL. The parser is wrong or "
                          "absent for this model's chat template. Agents would "
                          "run and execute nothing. Fix tool_call_parser before "
                          "starting a campaign."),
        }

    def stop(self, timeout_s: float = 30.0) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "VLLMServer":
        return self.start().wait_until_ready()

    def __exit__(self, *exc_info) -> None:
        self.stop()


@dataclass
class VLLMLLMConfig:
    """AG2/AutoGen `llm_config` pointing at a local vLLM-XPU server.

    Drop-in peer of `agents.llm_backend.ALCFLLMConfig`: same `to_autogen_config`
    contract, no Globus token, no outbound network.
    """

    model: str
    base_url: Optional[str] = None
    api_key: str = "EMPTY"
    temperature: float = 0.1
    timeout: int = 900
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @classmethod
    def from_server(cls, config: VLLMServerConfig,
                    temperature: float = 0.1) -> "VLLMLLMConfig":
        return cls(model=config.client_model_name, base_url=config.base_url,
                   api_key=config.api_key, temperature=temperature,
                   host=config.host, port=config.port)

    def resolved_base_url(self) -> str:
        return (self.base_url or os.environ.get("VLLM_BASE_URL")
                or f"http://{self.host}:{self.port}/v1")

    def to_autogen_config(self) -> dict:
        return {
            "config_list": [{
                "model": self.model,
                "api_key": self.api_key,
                "base_url": self.resolved_base_url(),
                # Local inference is not metered; silences AG2 cost tracking.
                "price": [0, 0],
            }],
            "temperature": self.temperature,
            "timeout": self.timeout,
            "stream": False,
        }


def serve_and_configure(server_config: VLLMServerConfig,
                        temperature: float = 0.1,
                        verify_tools: bool = True):
    """Start a server, confirm it can emit tool calls, return (server, config).

    Caller owns the server: use it as a context manager or call `.stop()`.
    `verify_tools=True` fails fast rather than letting a whole campaign run
    against a model that cannot invoke anything.
    """
    server = VLLMServer(server_config).start().wait_until_ready()
    if verify_tools:
        report = server.verify_tool_calling()
        if not report["ok"]:
            server.stop()
            raise RuntimeError(
                "vLLM server started but tool calling does not work.\n"
                f"  parser: {report['parser']}\n"
                f"  model reply: {report['content_preview']!r}\n"
                f"  {report['diagnosis']}"
            )
    return server, VLLMLLMConfig.from_server(server_config, temperature=temperature)
