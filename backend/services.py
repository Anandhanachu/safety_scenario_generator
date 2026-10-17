from backend.configuration import DATA_DIR, ROOT, defaults
from backend.orchestration import RunManager
from backend.persistence import Store
from evaluation.benchmark import BenchmarkRunner
from knowledge_base.core import Dataset, parse_upload
from risk_engine.calculator import RiskMatrix


class Services:
    def __init__(self, data_dir=DATA_DIR):
        self.store = Store(data_dir / "safescenario.sqlite")
        self.manager = RunManager(self.store)
        self.benchmarks = BenchmarkRunner(self.manager, self.store)
        if not self.store.setting("active_kb"):
            sample = parse_upload((ROOT / "data/expanded_knowledge_base.json").read_bytes())
            self.activate_dataset(sample)
        else:
            original = parse_upload((ROOT / "data/sample_knowledge_base.json").read_bytes())
            # Upgrade only the untouched bundled starter; preserve all custom/edited datasets.
            if self.dataset().version == original.version:
                self.activate_dataset(parse_upload((ROOT / "data/expanded_knowledge_base.json").read_bytes()))
        if not self.store.setting("active_matrix"):
            self.activate_matrix(RiskMatrix.model_validate_json((ROOT / "configs/risk_matrix.example.json").read_text(encoding="utf-8")))

    def settings(self):
        return self.store.setting("preferences", defaults())

    def dataset(self):
        return Dataset.model_validate(self.store.get(self.store.setting("active_kb"))["payload"])

    def matrix(self):
        return RiskMatrix.model_validate(self.store.get(self.store.setting("active_matrix"))["payload"])

    def activate_dataset(self, dataset):
        identity = self.store.save("knowledge_base", dataset.model_dump())
        self.store.set_setting("active_kb", identity)

    def activate_matrix(self, matrix):
        identity = self.store.save("risk_matrix", matrix.model_dump())
        self.store.set_setting("active_matrix", identity)
