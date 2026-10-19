"""Build the bundled illustrative demo KB deterministically.

The records remain pending expert review.  This script does not turn the
engineering rankings into measured probabilities; it only keeps the checked-in
demo data consistent and reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "multi_fault_knowledge_base.json"

THIRD_GROUP = {
    "camera_failure": (
        "Unavailable pedestrian detection",
        "Camera failure can prevent reliable pedestrian detection",
        "Stop before continuing without reliable pedestrian detection",
        "Inject camera unavailability and verify the product stops before continuing without reliable pedestrian detection",
    ),
    "communication_loss": (
        "Unavailable remote status reporting",
        "Communication loss can prevent safety-relevant status from reaching remote supervision",
        "Enter the documented safe state when safety-relevant status cannot be reported",
        "Interrupt communication and verify the product enters the documented safe state when safety-relevant status cannot be reported",
    ),
    "reduced_traction": (
        "Loss of directional control",
        "Reduced traction can prevent the product from maintaining the intended travel direction",
        "Reduce motion and stop when directional control cannot be maintained",
        "Apply reduced traction and verify the product reduces motion and stops when directional control cannot be maintained",
    ),
    "processing_delay": (
        "Stale motion command execution",
        "Processing delay can cause a motion command to be executed after its safe operating context has changed",
        "Reject stale motion commands and enter the documented safe state",
        "Inject processing delay and verify stale motion commands are rejected and the documented safe state is entered",
    ),
    "lidar_failure": (
        "Unavailable pedestrian range estimation",
        "Lidar failure can prevent reliable estimation of pedestrian distance",
        "Stop before continuing without reliable pedestrian range estimation",
        "Inject lidar unavailability and verify the product stops before continuing without reliable pedestrian range estimation",
    ),
    "ultrasonic_failure": (
        "Unavailable near-field obstacle detection",
        "Ultrasonic sensor failure can prevent reliable detection of nearby obstacles",
        "Stop or maintain a controlled safe distance when near-field detection is unavailable",
        "Disable ultrasonic sensing and verify the product stops or maintains the documented safe distance",
    ),
    "motor_control_failure": (
        "Uncontrolled drive command",
        "Motor control failure can cause an unintended or uncommanded drive response",
        "Disable drive output and enter the documented safe state on motor-control disagreement",
        "Inject a motor-control disagreement and verify drive output is disabled and the safe state is entered",
    ),
    "battery_fault": (
        "Unexpected loss of energy reserve",
        "A battery fault can cause an unexpected loss of energy during autonomous operation",
        "Report the battery fault and stop in a controlled location before energy reserve is exhausted",
        "Inject a battery fault and verify the product reports it and stops in a controlled location",
    ),
}

CONDITION_LABEL = {
    "day_dry": "day dry",
    "night_dry": "night dry",
    "rain": "rain",
    "reduced_visibility": "reduced visibility",
}


def record(record_id, kind, title, description, fault, condition, hazard_id=None):
    item = {
        "id": record_id,
        "type": kind,
        "title": title,
        "description": description,
        "source_ids": [],
        "mitigation_ids": [],
        "test_ids": [],
        "related_ids": [],
        "review_status": "pending_review",
        "source": "expert_judgment",
        "illustrative": True,
        "reviewed_by": None,
        "tags": {"fault_type": [fault], "operating_condition": [condition]},
        "hazard_ids": [hazard_id] if hazard_id else [],
        "severity": None,
        "likelihood": None,
        "risk_conflict": False,
        "expert_support_review": "not_reviewed",
    }
    return item


def main():
    data = json.loads(PATH.read_text(encoding="utf-8"))
    # The bundled demo treats rain and reduced visibility as frequent exposure
    # profiles. These remain explicit illustrative expert-judgment inputs.
    for item in data["records"]:
        if item["type"] == "hazard" and item["tags"]["operating_condition"][0] in {"rain", "reduced_visibility"}:
            item["likelihood"] = "L3"

    existing = {item["id"] for item in data["records"]}
    additions = []
    for fault, (hazard_title, hazard_text, mitigation_text, test_text) in THIRD_GROUP.items():
        for condition, condition_label in CONDITION_LABEL.items():
            hazard_id = f"H-{fault}-{condition}-3"
            mitigation_id = f"M-{fault}-{condition}-3"
            test_id = f"T-{fault}-{condition}-3"
            if hazard_id in existing:
                continue
            suffix = f" Operating condition: {condition_label}."
            hazard = record(hazard_id, "hazard", f"{hazard_title} ({condition_label})", hazard_text + "." + suffix,
                            fault, condition)
            hazard.update(
                mitigation_ids=[mitigation_id],
                test_ids=[test_id],
                severity="S3",
                likelihood="L3" if condition in {"rain", "reduced_visibility"} else "L2" if condition == "night_dry" else "L1",
            )
            mitigation = record(mitigation_id, "mitigation", mitigation_text + ".", mitigation_text + "." + suffix,
                                fault, condition, hazard_id)
            acceptance = record(test_id, "acceptance_test", test_text + ".", test_text + "." + suffix,
                                fault, condition, hazard_id)
            additions.extend([hazard, mitigation, acceptance])
    data["records"].extend(additions)
    data["dataset_id"] = "multi-fault-draft-288"
    data["dataset_name"] = "Illustrative multi-fault safety knowledge base (8 faults, 3 scenarios per profile)"
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
