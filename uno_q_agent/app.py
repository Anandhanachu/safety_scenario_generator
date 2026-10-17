"""Versioned, fail-closed authenticated Linux agent; launch on the board explicitly."""
from concurrent.futures import ThreadPoolExecutor
import copy
from collections import deque
import os
import platform
import secrets
import socket
import threading
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field

from backend.configuration import defaults
from backend.persistence import now
from backend.schemas import Generation, Strict
from models.adapters import BackendError, Cancelled, make_adapter
from uno_q_agent.bridge_client import BridgeClient
from uno_q_agent.model_manager import register_file, registered_files
from backend.middleware import BodyLimit


class Message(Strict):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=200000)


class InferenceJob(Strict):
    model_id: str = Field(min_length=1, max_length=500)
    messages: list[Message] = Field(min_length=1, max_length=10)
    generation: Generation = Field(default_factory=Generation)


class HardwareTest(Strict):
    name: str = Field(pattern="^ping$")
    confirmed: bool = False


class ModelRegistration(Strict):
    path: str = Field(min_length=1, max_length=2000)


auth = HTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(auth)):
    expected = os.getenv("UNO_Q_AGENT_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Agent token is not configured; access disabled")
    if not credentials or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "Invalid authentication")


app = FastAPI(title="SafeScenario UNO Q Linux Agent", version="1.0", dependencies=[Depends(authenticate)])
app.add_middleware(BodyLimit)
jobs, lock = {}, threading.RLock()
pool = ThreadPoolExecutor(max_workers=2)
logs = deque(maxlen=100)
bridge = BridgeClient()


def runtime():
    return make_adapter("laptop_local", defaults())


@app.get("/v1/health")
def health():
    memory = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            memory = {line.split(":")[0]: line.split(":")[1].strip() for line in file if line.startswith(("MemTotal:", "MemAvailable:"))}
    except OSError:
        pass
    with lock:
        running_tasks = sum(j["status"] in {"waiting", "running"} for j in jobs.values())
    return {"agent_version": "1.0", "hostname": socket.gethostname(), "platform": platform.platform(),
            "board_identity": "Not independently verified; this is the configured Linux agent",
            "checked_at": now(), "memory": memory, "mcu": bridge.capabilities(),
            "runtime": runtime().check_availability(), "running_tasks": running_tasks,
            "capabilities": {"load": False, "unload": False, "cancel": "cooperative", "hardware_tests": "read-only firmware ping"}}


@app.get("/v1/models")
def models():
    try:
        inventory = runtime().list_models(refresh=True)
    except BackendError as e:
        manual = registered_files()
        if manual:
            return {"models": manual, "runtime_error": str(e)}
        raise HTTPException(503, str(e)) from None
    return {"models": inventory + [m for m in registered_files() if m["id"] not in {i["id"] for i in inventory}]}


@app.post("/v1/models/register")
def register_model(request: ModelRegistration):
    try:
        return register_file(request.path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


def execute(identity, request):
    with lock:
        jobs[identity]["status"] = "running"
        jobs[identity]["started"] = now()
    try:
        adapter = runtime()
        adapter.validate_configuration(request.model_id)
        requested = request.generation.model_dump()
        response = adapter.generate_with_settings(request.model_id, [m.model_dump() for m in request.messages], requested, jobs[identity]['cancel']) if hasattr(adapter, 'generate_with_settings') else {'output': adapter.generate_scenarios(request.model_id, [m.model_dump() for m in request.messages], requested, jobs[identity]['cancel']), 'applied_settings': {k: 'not measured' for k in requested}}
        output = response['output']
        with lock:
            jobs[identity].update(status="cancelled" if jobs[identity]["cancel"].is_set() else "completed", output=None if jobs[identity]["cancel"].is_set() else output)
            jobs[identity].update(requested_settings=requested, applied_settings=response['applied_settings'])
    except Cancelled:
        with lock:
            jobs[identity].update(status="cancelled")
    except BackendError as e:
        with lock:
            jobs[identity].update(status="failed", error=str(e))
    except Exception:
        with lock:
            jobs[identity].update(status="failed", error="Unexpected agent inference error")
    finally:
        with lock:
            jobs[identity]["finished"] = now()
            logs.append({"at": now(), "job_id": identity, "status": jobs[identity]["status"]})


@app.post("/v1/jobs", status_code=202)
def start_job(request: InferenceJob):
    with lock:
        if sum(j["status"] in {"waiting", "running"} for j in jobs.values()) >= 4:
            raise HTTPException(429, "Agent queue full")
        # Bound retained task history; no model prompts or credentials are logged.
        if len(jobs) >= 100:
            finished = next((i for i, j in jobs.items() if j["status"] not in {"waiting", "running"}), None)
            if finished:
                del jobs[finished]
        identity = uuid4().hex
        jobs[identity] = {"id": identity, "model_id": request.model_id, "status": "waiting", "created": now(), "cancel": threading.Event()}
        pool.submit(execute, identity, request)
    return {"id": identity, "status": "waiting"}


@app.get("/v1/jobs/{identity}")
def status(identity: str):
    with lock:
        if identity not in jobs:
            raise HTTPException(404, "Job not found; agent jobs are ephemeral across restarts")
        return copy.deepcopy({k: v for k, v in jobs[identity].items() if k != "cancel"})


@app.post("/v1/jobs/{identity}/cancel")
def cancel(identity: str):
    with lock:
        if identity not in jobs:
            raise HTTPException(404, "Unknown job")
        if jobs[identity]["status"] not in {"waiting", "running"}:
            return {"status": "already_finished"}
        jobs[identity]["cancel"].set()
    return {"status": "cancellation_requested", "note": "Runtime request may finish; late output is discarded"}


@app.get("/v1/logs")
def log_entries():
    with lock:
        return list(logs)


@app.post("/v1/hardware-tests")
def hardware_test(request: HardwareTest):
    if not request.confirmed:
        raise HTTPException(400, "Explicit user confirmation required")
    if request.name not in bridge.capabilities()["tests"]:
        raise HTTPException(409, "Matching firmware/Bridge unavailable")
    start = time.perf_counter()
    try:
        response = bridge.call("safescenario_ping")
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(503, "MCU communication failed") from None
    return {"kind": "observed_mcu_rpc", "response": response,
            "round_trip_ms": round((time.perf_counter() - start) * 1000, 3),
            "note": "Linux-to-MCU RPC round-trip, not firmware execution time or robot stopping time"}
