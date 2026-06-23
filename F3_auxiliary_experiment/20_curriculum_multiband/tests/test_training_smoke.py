import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))


spec = importlib.util.spec_from_file_location(
    "train_curriculum",
    CODE_DIR / "03_train_curriculum.py",
)
train_curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_curriculum)
run_training = train_curriculum.run_training


class TinyDataset(Dataset):
    def __init__(self, domain):
        self.domain = domain
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 4

    def __getitem__(self, index):
        torch.manual_seed(index + (0 if self.domain == "f3" else 20))
        input_data = torch.randn(1, 64, 64) * 0.1
        label = torch.randn(1, 64, 64) * 0.03
        projector = (
            torch.tensor([10.0, 12.0, 30.0, 35.0])
            if self.domain == "f3"
            else torch.tensor([32.0, 38.0, 85.0, 100.0])
        )
        row = {
            "input": input_data,
            "label": label,
            "target": input_data + label,
            "projector": projector,
            "domain": self.domain,
        }
        if self.domain == "synthetic":
            row["clean_input"] = input_data
        return row


def test_one_joint_epoch_records_both_domains(tmp_path):
    f3_train = TinyDataset("f3")
    f3_val = TinyDataset("f3")
    result = run_training(
        epochs=1,
        f3_train=f3_train,
        f3_val=f3_val,
        synthetic_train=TinyDataset("synthetic"),
        synthetic_val=TinyDataset("synthetic"),
        output_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=2,
        steps_per_epoch=1,
        validation_batches=1,
        base_c=4,
    )
    assert result.history[0]["f3_train"]["correlation"] >= -1.0
    assert "synthetic_val" in result.history[0]
    assert result.history[0]["uses_f3_wide_target"] is False
    assert result.history[0]["f3_val"]["leakage"] < 0.03
    assert f3_train.epoch == 1
    assert f3_val.epoch == 0


def test_failed_gate_saves_separate_diagnostic_candidate(tmp_path):
    result = run_training(
        epochs=1,
        f3_train=TinyDataset("f3"),
        f3_val=TinyDataset("f3"),
        synthetic_train=TinyDataset("synthetic"),
        synthetic_val=TinyDataset("synthetic"),
        output_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=2,
        steps_per_epoch=1,
        validation_batches=1,
        base_c=4,
    )

    checkpoint = tmp_path / "diagnostic_candidate_model.pth"
    metadata_path = tmp_path / "diagnostic_candidate_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.gate_passed is False
    assert checkpoint.exists()
    assert metadata["gate_passed"] is False
    assert metadata["uses_f3_wide_target"] is False
    assert metadata["evaluation_authorized_by_user"] is True
    assert metadata["selection_metric"] == "f3_correlation_then_phase"
    assert len(metadata["sha256"]) == 64
