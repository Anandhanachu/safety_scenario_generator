import os
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import socket

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SAFESCENARIO_DATA_DIR", str(ROOT / "runtime"))).resolve()


def validate_url(value: str, kind: str):
    p = urlsplit(value)
    if not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError("Endpoint must have a hostname and no credentials, query or fragment")
    if kind == "qualcomm":
        hosts = os.getenv("QUALCOMM_ALLOWED_HOSTS", "aisuite.cirrascale.com").split(",")
        if p.scheme != "https" or p.hostname not in [h.strip() for h in hosts]:
            raise ValueError("Cloud endpoint requires HTTPS and an explicitly allowlisted hostname")
    else:
        if p.scheme not in {"http", "https"}:
            raise ValueError("Endpoint must use HTTP or HTTPS")
        try:
            addresses = socket.getaddrinfo(p.hostname, p.port or 80, type=socket.SOCK_STREAM)
        except OSError:
            raise ValueError("Endpoint hostname cannot be resolved") from None
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if kind == "laptop_local" and not ip.is_loopback:
                raise ValueError("Laptop runtime must bind to loopback")
            if kind == "uno_q" and not (ip.is_private or ip.is_loopback):
                raise ValueError("Board must use a private network or configured private VPN address")
            if ip.is_unspecified or ip.is_multicast or ip.is_link_local:
                raise ValueError("Unspecified, multicast and link-local targets are not permitted")
    return value.rstrip("/")


def defaults():
    return {
        "laptop_local": {"base_url": os.getenv("LAPTOP_BASE_URL", "http://127.0.0.1:8080"), "model_id": ""},
        "qualcomm": {"base_url": os.getenv("QUALCOMM_BASE_URL", "https://aisuite.cirrascale.com/apis"), "model_id": ""},
        "uno_q": {"base_url": os.getenv("UNO_Q_BASE_URL", ""), "model_id": ""},
        "model_dirs": [p for p in os.getenv("LOCAL_MODEL_DIRS", "").split(os.pathsep) if p],
        "retrieval_limit": 12,
        "relevance_threshold": 0.05,
    }
