import hashlib
import json
from pathlib import Path


FORBIDDEN_TOKENS = (
    "18_real_domain_phase_consistent",
    "wide_reference",
    "wide_prediction",
    "真实样本",
)


def assert_training_paths_are_safe(paths):
    for path in paths:
        normalized = str(Path(path)).replace("\\", "/").lower()
        for token in FORBIDDEN_TOKENS:
            if token.lower() in normalized:
                raise ValueError(f"Forbidden training data path: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model_lock(checkpoint_path, lock_path, metadata):
    payload = {
        **metadata,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sha256": sha256_file(checkpoint_path),
    }
    Path(lock_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def verify_model_lock(checkpoint_path, lock_path):
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    actual = sha256_file(checkpoint_path)
    if actual != payload["sha256"]:
        raise ValueError("Checkpoint hash does not match the locked model.")
    return payload
