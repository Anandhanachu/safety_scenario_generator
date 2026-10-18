"""Real HTTP inference adapters. No automatic fallback and no fabricated inventory."""
import os
from abc import ABC, abstractmethod
from pathlib import Path
import threading
import time
import math
import json
from urllib.parse import urlsplit

import httpx

from backend.configuration import validate_url


class BackendError(RuntimeError):
    pass


class Cancelled(BackendError):
    pass


DEFAULT_BACKEND_TIMEOUT_SECONDS = 900


def backend_timeout_seconds(value=None):
    """Return the bounded HTTP/job timeout used by configured backends."""
    raw = value if value is not None else os.getenv("SAFESCENARIO_BACKEND_TIMEOUT_SECONDS", str(DEFAULT_BACKEND_TIMEOUT_SECONDS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_BACKEND_TIMEOUT_SECONDS
    return max(30, min(value, 900))


class Adapter(ABC):
    capabilities = {"list_models": True, "inference": True, "load": False, "unload": False, "cancel_remote": False}

    def __init__(self, base_url, backend, token="", timeout=120, transport=None):
        self.base_url, self.backend, self.token = base_url, backend, token
        self.timeout, self.transport = timeout, transport
        self.applied_settings = {k: 'not measured' for k in ('temperature', 'seed', 'max_tokens')}
        self.ollama_fast_mode = False
        self.requested_runtime_options = {}
        self.runtime_metrics = {}

    def record_settings(self, data):
        """Only explicit runtime reports establish applied/unsupported settings."""
        if not isinstance(data, dict):
            return
        applied = data.get('applied_settings', {})
        unsupported = data.get('unsupported_settings', [])
        for key in self.applied_settings:
            value = applied.get(key) if isinstance(applied, dict) else None
            if value == 'not supported' or isinstance(unsupported, list) and key in unsupported:
                self.applied_settings[key] = 'not supported'
            elif type(value) in (int, float) and math.isfinite(value):
                self.applied_settings[key] = value

    def generate_with_settings(self, model_id, messages, generation, cancel=None):
        self.applied_settings = {k: 'not measured' for k in ('temperature', 'seed', 'max_tokens')}
        self.requested_runtime_options = {}
        self.runtime_metrics = {}
        output = self.generate_scenarios(model_id, messages, generation, cancel)
        return {'output': output, 'requested_settings': dict(generation), 'applied_settings': dict(self.applied_settings),
                'requested_runtime_options': dict(self.requested_runtime_options), 'runtime_metrics': dict(self.runtime_metrics)}

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
                try:
                    self.record_settings(response.json())
                except ValueError:
                    pass
                raise BackendError(f"Backend HTTP {response.status_code}; check service configuration")
            if len(response.content) > 5 * 1024 * 1024:
                raise BackendError("Backend response exceeds 5 MiB")
            return response.json()
        except (httpx.TimeoutException, TimeoutError):
            raise BackendError(f"Backend request timed out after {self.timeout} seconds") from None
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


def _text_content(value):
    """Decode documented Imagine/OpenAI text containers, never binary parts."""
    if isinstance(value, str):
        return value if value.strip() else None
    if not isinstance(value, dict):
        return None
    if set(value) == {'String'}:
        return _text_content(value['String'])
    if set(value) == {'Vec'} and isinstance(value['Vec'], list) and all(isinstance(v, str) for v in value['Vec']):
        joined = ''.join(value['Vec'])
        return joined if joined.strip() else None
    # ChatDeltaContent may be a Message or an otherwise unconstrained object.
    if 'content' in value:
        return _text_content(value['content'])
    if set(value) == {'Text'} and isinstance(value['Text'], dict):
        return _text_content(value['Text'].get('text'))
    if set(value) == {'MultiModal'} and isinstance(value['MultiModal'], list):
        parts = [_text_content(part) for part in value['MultiModal']]
        joined = ''.join(part for part in parts if part)
        return joined if joined.strip() else None
    return None


def _response_shape(data):
    """Return non-content diagnostics safe to retain in failed run history."""
    def shape(value):
        if isinstance(value, dict):
            details = {'type': 'dict', 'keys': sorted(value)}
            for key in ('String', 'Vec', 'MultiModal'):
                if key in value:
                    tagged = value[key]
                    details[key] = {'type': type(tagged).__name__,
                                    'length': len(tagged) if isinstance(tagged, (str, list, dict)) else None}
            return details
        return {'type': type(value).__name__,
                'length': len(value) if isinstance(value, (str, list)) else None}
    if not isinstance(data, dict):
        return type(data).__name__
    choices = data.get('choices')
    first = choices[0] if isinstance(choices, list) and choices else None
    keys = sorted(first) if isinstance(first, dict) else []
    nested = {key: {**shape(first[key]),
                    'content': shape(first[key].get('content')) if isinstance(first[key], dict) else None}
              for key in ('message', 'delta') if isinstance(first, dict) and key in first}
    return f"top={sorted(data)}; choice={keys}; nested={nested}"


def content_from_response(data, expected_model):
    if not isinstance(data, dict) or data.get("model") != expected_model:
        raise BackendError("Backend returned a missing or inconsistent model identifier")
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise BackendError("Model output truncated or filtered; adjust supported generation settings")
        # Imagine OpenAPI documents delta even for its non-streaming response.
        content = None
        for part in (choice.get('message'), choice.get('delta'), choice.get('text')):
            content = _text_content(part)
            if content:
                break
        if not isinstance(content, str) or not content.strip():
            raise ValueError()
        if '</think>' in content:
            content = content.split('</think>', 1)[1].strip()
        elif '<think>' in content:
            content = ''
        if not isinstance(content, str) or not content.strip():
            raise ValueError()
        return content
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise BackendError("Malformed chat-completion response: missing text content; " + _response_shape(data)) from None


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
        payload = {"model": model_id, "messages": messages, "stream": False, **generation}
        if self.ollama_fast_mode:
            self.requested_runtime_options = {'reasoning_effort': 'none', 'response_format': {'type': 'json_object'}}
            # Both pipeline prompts carry an application-generated schema. Ask
            # Ollama to enforce it while decoding instead of spending a retry on JSON.
            for message in messages:
                if message.get('role') != 'user':
                    continue
                try:
                    contract = json.loads(message['content'])
                except (ValueError, TypeError, KeyError):
                    continue
                if isinstance(contract, dict) and isinstance(contract.get('schema'), dict):
                    self.requested_runtime_options['response_format'] = {
                        'type': 'json_schema', 'json_schema': {'name': 'safescenario', 'schema': contract['schema']}}
                    break
            payload.update(self.requested_runtime_options)
        data = self.request("POST", "/v1/chat/completions", json=payload)
        self.record_settings(data)
        usage = data.get('usage', {})
        if isinstance(usage, dict):
            self.runtime_metrics = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                    if type(usage.get(key)) is int and usage[key] >= 0}
            details = usage.get('completion_tokens_details', {})
            if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int:
                self.runtime_metrics['reasoning_tokens'] = details['reasoning_tokens']
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
        # Imagine's published SamplingParams do not include seed. Keep it in
        # requested settings for fair-run reporting, but do not send an
        # undocumented field. Greedy decoding is requested explicitly.
        supported = {key: value for key, value in generation.items() if key in {'temperature', 'max_tokens'}}
        self.requested_runtime_options = {'do_sample': False}
        if 'seed' in generation:
            self.applied_settings['seed'] = 'not supported'
        data = self.request("POST", "/v2/chat/completions",
                            json={"model": model_id, "messages": messages, "stream": False,
                                  **supported, **self.requested_runtime_options})
        self.record_settings(data)
        usage = data.get('usage', {})
        if isinstance(usage, dict):
            self.runtime_metrics = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                    if type(usage.get(key)) is int and usage[key] >= 0}
            details = usage.get('completion_tokens_details', {})
            if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int:
                self.runtime_metrics['reasoning_tokens'] = details['reasoning_tokens']
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
                self.record_settings(status)
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
    timeout = backend_timeout_seconds(settings.get("backend_timeout_seconds"))
    adapter = classes[backend](cfg["base_url"], backend, token, timeout=timeout, transport=transport)
    if backend == 'laptop_local':
        adapter.ollama_fast_mode = cfg.get('ollama_fast_mode', urlsplit(cfg['base_url']).port == 11434)
    return adapter


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
