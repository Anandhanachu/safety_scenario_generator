"""Versioned contracts. AI proposals never contain calculated risk or human decisions."""
from typing import Literal
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False, strict=True)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


Backend = Literal["laptop_local", "qualcomm", "uno_q"]


class ScenarioInput(Strict):
    time_of_day: Literal["day", "night"] = "day"
    weather: Literal["dry", "wet", "rain"] = "dry"
    visibility: Literal["clear", "reduced"] = "clear"
    surface: str = Field(default="paved sidewalk", min_length=1, max_length=200)
    pedestrian_density: Literal["low", "medium", "high"] = "medium"
    location: str = Field(default="sidewalk", min_length=1, max_length=200)
    speed_mps: float = Field(default=1.0, ge=0, le=5)
    navigation_mode: str = Field(default="autonomous", min_length=1, max_length=100)
    sensors: list[str] = Field(default_factory=list, max_length=20)
    documented_capabilities: str = Field(default="", max_length=2000)
    faults: list[str] = Field(default_factory=list, max_length=5)
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def unique_faults(self):
        if len(set(self.faults)) != len(self.faults):
            raise ValueError("Faults must be unique")
        if any(not x or len(x) > 200 for x in self.faults + self.sensors):
            raise ValueError("Fault and sensor labels must have 1–200 characters")
        return self


class Generation(Strict):
    temperature: float = Field(default=0.0, ge=0, le=0)
    seed: Literal[42] = 42
    max_tokens: int = Field(default=1800, ge=64, le=8192)


class Selection(Strict):
    backend: Backend
    model_id: str = Field(min_length=1, max_length=500)
    generation: Generation = Field(default_factory=Generation)


class Hazard(Strict):
    hazard_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=3000)
    consequences: list[str] = Field(min_length=1, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    hypothesis: bool = True


class Candidate(Strict):
    scenario_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    input_conditions: ScenarioInput
    scenario_narrative: str = Field(min_length=1, max_length=8000)
    hazards: list[Hazard] = Field(min_length=1, max_length=20)
    mitigation_ids: list[str] = Field(default_factory=list, max_length=50)
    acceptance_test_ids: list[str] = Field(default_factory=list, max_length=50)


class ModelOutput(Strict):
    schema_version: Literal["1.0"]
    scenarios: list[Candidate] = Field(min_length=1, max_length=10)


class RunRequest(Strict):
    inputs: ScenarioInput
    selections: list[Selection] = Field(min_length=1, max_length=3)
    retrieval_limit: int = Field(default=12, ge=1, le=40)
    relevance_threshold: float = Field(default=0.05, ge=0, le=1)
    citation_link_validity_threshold: float = Field(default=1.0, ge=0, le=1)
    method: Literal["full_pipeline", "llm_only"] = "full_pipeline"
    generation_mode: Literal['record_selection', 'scenario_drafts'] = 'record_selection'
    max_scenarios: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def unique_backends(self):
        if len({s.backend for s in self.selections}) != len(self.selections):
            raise ValueError("Choose at most one model per backend")
        if len({s.generation.max_tokens for s in self.selections}) > 1:
            raise ValueError('Fair comparison requires identical token limits')
        return self
