"""Real HTTP inference adapters. No automatic fallback and no fabricated inventory."""
import os
from abc import ABC, abstractmethod
from pathlib import Path
import threading
import time

import httpx

from backend.configuration import validate_url


class BackendError(RuntimeError):
    pass


class Cancelled(BackendError):
    pass


class Adapter(ABC):
    capabilities = {"list_models": True, "inference": True, "load": False, "unload": False, "cancel_remote": False}

    def __init__(self, base_url, backend, token="", timeout=120, transport=None):
        self.base_url, self.backend, self.token = base_url, backend, token
        self.timeout, self.transport = timeout, transport

    def request(self, method, path, **kwargs):
        if not self.base_url:
            raise BackendError("Setup required: configure the backend address")
        if self.backend in {"qualcomm", "uno_q"} and not self.token:
            raise BackendError("Setup required: configure the backend authentication token in the environment")
        try:
            base = validate_url(self.base_url, self.backend)
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            with httpx.Client(timeout=self.timeout, follow_redirects=False, transport=self.transport, trust_env=False) as client:
                # Inference POSTs are never automatically retried (billing/duplicate execution).
                response = client.request(method, base + path, headers=headers, **kwargs)
            if response.status_code in {401, 403}:
                raise BackendError("Authentication failed: check credentials and account permissions")
            if response.status_code == 429:
                raise BackendError("Rate limited: retry later; request was not automatically repeated")
            if not 200 <= response.status_code < 300:
                raise BackendError(f"Backend HTTP {response.status_code}; check service configuration")
            if len(response.content) > 5 * 1024 * 1024:
                raise BackendError("Backend response exceeds 5 MiB")
            return response.json()
        except (httpx.TimeoutException, TimeoutError):
            raise BackendError("Backend request timed out") from None
        except (httpx.HTTPError, ValueError):
            raise BackendError("Backend unavailable, endpoint invalid, or malformed JSON response") from None

    def check_availability(self):
        try:
            models = self.list_models(refresh=True)
            return {"status": "reachable", "models": len(models), "inference_verified": False}
        except BackendError as e:
            return {"status": "setup_required_or_unavailable", "error": str(e)}

    @abstractmethod
    def list_models(self, refresh=False): ...

    def get_model_info(self, model_id):
        return next((m for m in self.list_models() if m["id"] == model_id), None)

    def validate_configuration(self, model_id):
        info = self.get_model_info(model_id)
        if info is None:
            raise BackendError("Selected model is not available in the runtime catalogue")
        if info.get("inference_ready") is False:
            raise BackendError("Registered model file is not confirmed by the inference runtime; explicitly load it in the runtime first")
        return True

    def load_model(self, model_id):
        raise BackendError("Runtime model loading unsupported; start or reconfigure the runtime explicitly")

    def unload_model(self, model_id):
        raise BackendError("Runtime model unloading unsupported")

    def get_job_status(self, job_id):
        raise BackendError("Remote job status unsupported by this provider")

    def cancel_job(self, job_id):
        raise BackendError("Remote cancellation unsupported; local cancellation discards late results")

    def get_runtime_information(self):
        return {"backend": self.backend, "capabilities": self.capabilities, **self.check_availability()}

    @abstractmethod
    def generate_scenarios(self, model_id, messages, generation, cancel=None): ...


def content_from_response(data, expected_model):
    if not isinstance(data, dict) or data.get("model") != expected_model:
        raise BackendError("Backend returned a missing or inconsistent model identifier")
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise BackendError("Model output truncated or filtered; adjust supported generation settings")
        # Imagine OpenAPI documents delta even for its non-streaming response.
        content = choice.get("message", choice.get("delta", {}))["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError()
        return content
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise BackendError("Malformed chat-completion response: missing text content") from None


class ChatAdapter(Adapter):
    def list_models(self, refresh=False):
        data = self.request("GET", "/v1/models")
        try:
            models = data["data"]
            if not isinstance(models, list) or any(not isinstance(m.get("id"), str) or not m["id"] for m in models):
                raise ValueError()
            return [{"id": m["id"], "metadata": m, "status": "listed_unverified"} for m in models]
        except (KeyError, TypeError, AttributeError, ValueError):
            raise BackendError("Malformed runtime model catalogue") from None

    def generate_scenarios(self, model_id, messages, generation, cancel=None):
        if cancel and cancel.is_set():
            raise Cancelled("Cancelled before inference")
        data = self.request("POST", "/v1/chat/completions", json={"model": model_id, "messages": messages, "stream": False, **generation})
        if cancel and cancel.is_set():
            raise Cancelled("Cancelled locally; remote inference may have completed")
        return content_from_response(data, model_id)


class QualcommAdapter(ChatAdapter):
    """Imagine OpenAPI 1.0.0 inspected 2026-10-17: Bearer JWT and category->IDs catalogue."""
    def list_models(self, refresh=False):
        data = self.request("GET", "/v2/models")
        if not isinstance(data, dict) or any(not isinstance(v, list) or any(not isinstance(i, str) or not i for i in v) for v in data.values()):
            raise BackendError("Malformed Imagine catalogue; expected model-type to model-ID arrays")
        return [{"id": i, "type": category, "status": "listed_unverified"} for category, ids in data.items() for i in ids]

    def generate_scenarios(self, model_id, messages, generation, cancel=None):
        if cancel and cancel.is_set():
            raise Cancelled("Cancelled before inference")
        data = self.request("POST", "/v2/chat/completions", json={"model": model_id, "messages": messages, "stream": False, **generation})
        if cancel and cancel.is_set():
            raise Cancelled("Cancelled locally; hosted inference may have completed and may be billed")
        return content_from_response(data, model_id)


class UnoAdapter(Adapter):
    capabilities = {**Adapter.capabilities, "cancel_remote": True, "job_status": True}

    def list_models(self, refresh=False):
        data = self.request("GET", "/v1/models")
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise BackendError("Malformed board model inventory")
        if any(not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in data["models"]):
            raise BackendError("Malformed board model metadata")
        return data["models"]

    def get_runtime_information(self):
        return self.request("GET", "/v1/health")

    def get_job_status(self, job_id):
        return self.request("GET", f"/v1/jobs/{job_id}")

    def cancel_job(self, job_id):
        return self.request("POST", f"/v1/jobs/{job_id}/cancel")

    def generate_scenarios(self, model_id, messages, generation, cancel=None):
        if cancel and cancel.is_set():
            raise Cancelled("Cancelled before starting a board job")
        job = self.request("POST", "/v1/jobs", json={"model_id": model_id, "messages": messages, "generation": generation})
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id.isalnum():
            raise BackendError("Malformed board job identifier")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if cancel and cancel.is_set():
                self.cancel_job(job_id)
                raise Cancelled("Board cancellation requested; runtime may finish in background")
            status = self.get_job_status(job_id)
            if status.get("status") == "completed":
                if not isinstance(status.get("output"), str):
                    raise BackendError("Malformed board inference result")
                return status["output"]
            if status.get("status") in {"failed", "cancelled"}:
                raise BackendError("Board inference failed or was cancelled; inspect board diagnostics")
            threading.Event().wait(0.3)
        self.cancel_job(job_id)
        raise BackendError("Board job timed out; cancellation requested")


def make_adapter(backend, settings, transport=None):
    cfg = settings[backend]
    classes = {"laptop_local": ChatAdapter, "qualcomm": QualcommAdapter, "uno_q": UnoAdapter}
    token = os.getenv({"qualcomm": "QUALCOMM_API_KEY", "uno_q": "UNO_Q_TOKEN"}.get(backend, "LAPTOP_API_KEY"), "")
    return classes[backend](cfg["base_url"], backend, token, transport=transport)


def discover_files(directories):
    files = []
    for directory in directories[:10]:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            continue
        for p in root.glob("*.gguf"):
            resolved = p.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            files.append({"path": str(resolved), "size_bytes": resolved.stat().st_size, "format": "GGUF", "quantization": "unknown", "status": "file_present_runtime_unverified"})
            if len(files) >= 200:
                return files
    return files
