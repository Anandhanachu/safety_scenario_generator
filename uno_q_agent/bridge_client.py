"""Optional read-only firmware handshake over Arduino Router MessagePack RPC.

Protocol: https://github.com/arduino/arduino-router . No actuator RPC is exposed.
"""
import os
import socket
import time


class BridgeClient:
    def call(self, method):
        if method not in {"safescenario_capabilities", "safescenario_ping"}:
            raise ValueError("RPC method not allowlisted")
        if os.getenv("UNO_Q_BRIDGE_ENABLED", "false").lower() != "true":
            raise RuntimeError("Bridge disabled; install the matching sketch and explicitly enable it")
        try:
            import msgpack
        except ImportError:
            raise RuntimeError("Install optional msgpack dependency on the board") from None
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect("/var/run/arduino-router.sock")
            connection.sendall(msgpack.packb([0, 1, method, []]))
            unpacker = msgpack.Unpacker(raw=False, max_buffer_size=65536)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                unpacker.feed(chunk)
                for message in unpacker:
                    if isinstance(message, list) and len(message) == 4 and message[0] == 1 and message[1] == 1:
                        if message[2] is not None:
                            raise RuntimeError("Firmware RPC returned an error")
                        return message[3]
        raise RuntimeError("No valid MCU RPC response")

    def capabilities(self):
        try:
            response = self.call("safescenario_capabilities")
            verified = response == "safescenario-readonly-v1"
            return {"status": "connected" if verified else "unsupported_firmware", "tests": ["ping"] if verified else []}
        except (OSError, RuntimeError, ValueError):
            return {"status": "setup_required_or_unavailable", "tests": []}
