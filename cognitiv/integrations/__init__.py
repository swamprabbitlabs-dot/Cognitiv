"""
cognitiv.integrations — Connect cognitiv to LLM backends.

Currently supports llama.cpp via its OpenAI-compatible server API.

Usage:
    from cognitiv.integrations import LlamaCppBridge, LlamaCppConfig

    bridge = LlamaCppBridge(base_url="http://localhost:8080")
    brain.set_llm_callback(bridge.complete)
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LlamaCppConfig:
    """Configuration for the llama.cpp server connection."""

    base_url: str = "http://localhost:8080"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 150
    repeat_penalty: float = 1.1
    n_ctx: int = 2048
    n_gpu_layers: int = 99
    port: int = 8080
    host: str = "127.0.0.1"
    n_threads: int = 0
    request_timeout: float = 30.0
    startup_timeout: float = 60.0
    startup_poll_interval: float = 0.5


class LlamaCppBridge:
    """Bridge between cognitiv and a llama.cpp server.

    Provides a `complete(prompt) -> str` callback that cognitiv's
    brain.set_llm_callback() expects.
    """

    def __init__(
        self,
        config: LlamaCppConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        self._config = config or LlamaCppConfig()
        if base_url:
            self._config.base_url = base_url

        self._process: subprocess.Popen | None = None
        self._base_url = self._config.base_url.rstrip("/")

        self._request_count = 0
        self._total_tokens = 0
        self._total_time = 0.0

    @classmethod
    def from_model(
        cls,
        model_path: str,
        llama_server_path: str = "llama-server",
        config: LlamaCppConfig | None = None,
    ) -> LlamaCppBridge:
        """Launch a llama-server process and connect to it."""
        config = config or LlamaCppConfig()
        bridge = cls(config=config)
        bridge._launch_server(model_path, llama_server_path)
        return bridge

    def complete(self, prompt: str) -> str:
        """Send a completion request. This is the cognitiv LLM callback."""
        url = f"{self._base_url}/v1/chat/completions"

        payload = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "repeat_penalty": self._config.repeat_penalty,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(
                req, timeout=self._config.request_timeout
            ) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach llama-server at {self._base_url}: {e}"
            ) from e

        elapsed = time.monotonic() - start

        try:
            text = body["choices"][0]["message"]["content"].strip()
            tokens = body.get("usage", {}).get("completion_tokens", 0)
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response format from llama-server: {body}"
            ) from e

        self._request_count += 1
        self._total_tokens += tokens
        self._total_time += elapsed

        return text

    def complete_raw(self, prompt: str, **kwargs) -> dict:
        """Send a raw completion request, returning the full response dict."""
        url = f"{self._base_url}/v1/chat/completions"

        payload = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", self._config.temperature),
            "top_p": kwargs.get("top_p", self._config.top_p),
            "max_tokens": kwargs.get("max_tokens", self._config.max_tokens),
        }
        payload.update({k: v for k, v in kwargs.items()
                        if k not in ("temperature", "top_p", "max_tokens")})

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            req, timeout=self._config.request_timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_healthy(self) -> bool:
        """Check if the server is responding."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("status") == "ok"
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Return request statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "total_time": round(self._total_time, 2),
            "avg_time": round(
                self._total_time / max(self._request_count, 1), 3
            ),
            "avg_tokens": round(
                self._total_tokens / max(self._request_count, 1), 1
            ),
        }

    def shutdown(self) -> None:
        """Stop the llama-server process (if we launched it)."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def _launch_server(
        self, model_path: str, llama_server_path: str
    ) -> None:
        """Start the llama-server subprocess."""
        cfg = self._config

        cmd = [
            llama_server_path,
            "--model", model_path,
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--ctx-size", str(cfg.n_ctx),
            "--n-gpu-layers", str(cfg.n_gpu_layers),
        ]

        if cfg.n_threads > 0:
            cmd.extend(["--threads", str(cfg.n_threads)])

        print(f"[cognitiv] Launching llama-server...")
        print(f"[cognitiv] Command: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._base_url = f"http://{cfg.host}:{cfg.port}"
        deadline = time.monotonic() + cfg.startup_timeout

        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode}).\n"
                    f"stderr: {stderr[:500]}"
                )

            if self.is_healthy():
                print(f"[cognitiv] llama-server ready at {self._base_url}")
                return

            time.sleep(cfg.startup_poll_interval)

        self.shutdown()
        raise TimeoutError(
            f"llama-server did not become healthy within {cfg.startup_timeout}s"
        )
