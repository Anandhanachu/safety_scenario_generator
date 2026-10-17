"""A toy watchdog controller state machine; no physics or learned behavior."""
from typing import Literal
from pydantic import Field
from backend.schemas import Strict


class SimulationInput(Strict):
    rule: Literal["watchdog", "unsupported"] = "watchdog"
    sensor_age_ms: int = Field(default=0, ge=0, le=60000)
    communication_ok: bool = True
    watchdog_enabled: bool = True
    stale_threshold_ms: int = Field(default=200, ge=1, le=60000)
    expected_state: Literal["STOP", "MOVE"] = "STOP"


def simulate(inputs: SimulationInput):
    observed = None
    if inputs.rule == "watchdog":
        observed = "STOP" if inputs.watchdog_enabled and (not inputs.communication_ok or inputs.sensor_age_ms > inputs.stale_threshold_ms) else "MOVE"
    return {"simulation_version": "watchdog-1.0", "kind": "software_simulation", "inputs": inputs.model_dump(),
            "observed_state": observed, "outcome": "inconclusive" if observed is None else "pass" if observed == inputs.expected_state else "fail",
            "assumptions": "STOP is an instantaneous logical command; no actuator, sensor, braking, traction or physical timing model.",
            "limitations": "Does not establish physical stopping distance, collision avoidance or real robot safety."}
