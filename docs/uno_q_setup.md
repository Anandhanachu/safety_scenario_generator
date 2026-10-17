# Arduino UNO Q: Linux agent, Wi-Fi and optional MCU probe

Use the [official UNO Q user manual](https://docs.arduino.cc/tutorials/uno-q/user-manual/) and [Arduino Router project](https://github.com/arduino/arduino-router) for board provisioning and current Bridge prerequisites. None of the steps below is executed automatically by the dashboard.

## Initial setup

1. Complete Arduino's initial wired/USB setup. Configure the board's Wi-Fi through your supported Arduino/Linux workflow and confirm its private address. Do not assume it is already on Wi-Fi.
2. Copy this repository to the board Linux MPU. Ensure Python and target-compatible dependencies are available. Do not attempt to run an LLM on the MCU.
3. Install a compatible llama.cpp build and instruction-tuned GGUF explicitly. Check memory/storage and license first. Start `llama-server` on the board's loopback address and port 8080, as described in [local models](local_models.md). Actual model feasibility depends on the board's resources.
4. Create a Linux virtual environment and install the minimal board dependencies (FastAPI, uvicorn, httpx, Pydantic, python-dotenv and msgpack). Dashboard/scientific packages are not required by the agent. Board-wheel availability and memory use remain unverified.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-board.txt
cp .env.example .env
```

5. Set **UNO_Q_AGENT_TOKEN** on the board to a long random token using your environment/secrets workflow. Set `LAPTOP_BASE_URL=http://127.0.0.1:8080` for its local inference server. Do not commit tokens.
6. Start the agent, substituting the board's actual Wi-Fi interface address:

```bash
.venv/bin/python -m uvicorn uno_q_agent.app:app --host <BOARD_PRIVATE_IP> --port 8090
```

Binding to a specific private interface avoids broad/public exposure. Allow access only from the intended laptop/network using your network configuration. The default example HTTP transport does not encrypt the token; use a protected private network or explicitly designed TLS/private VPN deployment. No public port forwarding is required.

## Connect from the laptop

1. Set `UNO_Q_TOKEN` on the laptop to match the board token and restart the dashboard.
2. In Settings, save `http://<BOARD_PRIVATE_IP>:8090` as the UNO Q base URL. A resolvable private hostname also works. The app validates resolved addresses and refuses public or link-local endpoints.
3. Open Device Control and click **Connect / refresh status**. The response shows actual hostname, platform, Linux memory data when available, runtime status, MCU status and task count. Hardware identity is not independently attested.
4. Refresh board models, then choose and save a returned UNO Q model ID in Model Studio. Click **Test real scenario inference**.
5. Use Scenario Builder or Model Comparison. The adapter starts an authenticated job, polls its real status and requests cancellation when appropriate. The agent never exposes an arbitrary shell or file-management endpoint.

Open the agent's `/docs` locally for the versioned API schema: `/v1/health`, `/v1/models`, `/v1/jobs`, `/v1/jobs/{id}`, `/v1/jobs/{id}/cancel`, `/v1/logs`, `/v1/hardware-tests`. All operational routes require the configured Bearer token. Agent jobs/logs are bounded and ephemeral across restarts.

If catalogue discovery cannot identify an installed file, configure `LOCAL_MODEL_DIRS` on the board and use **Register installed board model file** in Device Control (`POST /v1/models/register`). Registration verifies that the file exists inside an allowed directory, persists its path and labels it runtime-unverified. It cannot run until you explicitly configure the board runtime and select an ID that the runtime actually advertises.

## Optional read-only MCU test

The supplied `uno_q_agent/firmware/readonly_probe/readonly_probe.ino` registers `safescenario_capabilities` and `safescenario_ping` with Arduino Router Bridge. It controls no pins, motors or actuators. Compile and upload it explicitly using a supported Arduino UNO Q toolchain; firmware compilation/behavior was not verified on real hardware.

After installing the matching sketch and ensuring Arduino Router is running, set `UNO_Q_BRIDGE_ENABLED=true` on the board and restart the agent. The client uses standard MessagePack RPC through `/var/run/arduino-router.sock`, with an allowlist, size bound and timeout. A matching capability string is required before Device Control enables the explicitly confirmed ping test. Reported timing is Linux-to-MCU RPC round-trip including router/transport time, not MCU-only execution time and not robot stopping time.

Automatic network discovery, USB provisioning, firmware upload, Wi-Fi reconfiguration, service installation, model loading/unloading and actuator tests are not implemented. A private VPN can be configured outside the app; its resolved address must meet the app's private-address policy. Disconnecting the dashboard clears its session connection indicator; it does not stop the board service.

No physical UNO Q was connected during development. Board package installation, model feasibility, Wi-Fi reachability and firmware RPC require on-device verification.
