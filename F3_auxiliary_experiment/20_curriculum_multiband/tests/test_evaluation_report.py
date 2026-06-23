import importlib.util
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
spec = importlib.util.spec_from_file_location(
    "blind_evaluate",
    CODE_DIR / "05_blind_evaluate.py",
)
blind_evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind_evaluate)


def test_diagnostic_provenance_is_not_reported_as_gate_passed():
    lines = blind_evaluate.provenance_lines({
        "lock_sha256": "abc",
        "gate_passed": False,
        "diagnostic_ungated_evaluation": True,
        "evaluation_authorized_by_user": True,
    })

    text = "\n".join(lines)
    assert "did not pass" in text
    assert "authorized" in text
    assert "after diagnostic checkpoint verification" in text
