"""Optional local REST interface; Streamlit uses the same services directly."""
from functools import lru_cache
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.schemas import RunRequest
from backend.services import Services
from backend.middleware import BodyLimit

auth = HTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(auth)):
    expected = os.getenv("SAFESCENARIO_API_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Configure SAFESCENARIO_API_TOKEN to enable API access")
    if not credentials or not secrets.compare_digest(expected, credentials.credentials):
        raise HTTPException(401, "Invalid authentication")


@lru_cache
def services():
    return Services()


app = FastAPI(title="SafeScenario AI API", version="1.0")
app.add_middleware(BodyLimit)


@app.get("/health")
def health():
    return {"status": "running", "version": "1.0", "inference_verified": False}


@app.post("/v1/runs", dependencies=[Depends(authenticate)], status_code=202)
def start(request: RunRequest):
    s = services()
    return {"id": s.manager.start(request, s.dataset(), s.matrix(), s.settings())}


@app.get("/v1/runs/{identity}", dependencies=[Depends(authenticate)])
def get(identity: str):
    try:
        return services().manager.get(identity)
    except KeyError:
        raise HTTPException(404, "Unknown run") from None


@app.post("/v1/runs/{identity}/cancel", dependencies=[Depends(authenticate)])
def cancel(identity: str):
    try:
        return services().manager.cancel(identity)
    except KeyError:
        raise HTTPException(404, "No active job with this identifier") from None
