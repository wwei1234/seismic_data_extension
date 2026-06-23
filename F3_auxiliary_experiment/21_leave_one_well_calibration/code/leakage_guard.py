import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_fold_lock(
    checkpoint_path,
    manifest_path,
    lock_path,
    common_checkpoint_sha256,
):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("uses_heldout_well_wide_target") is not False:
        raise ValueError("Fold manifest uses a held-out wide target.")
    payload = {
        **manifest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": sha256_payload(manifest),
        "common_checkpoint_sha256": common_checkpoint_sha256,
    }
    Path(lock_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def create_common_lock(checkpoint_path, manifest_path, lock_path, metadata):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("uses_f3_wide_target") is not False:
        raise ValueError("Common pretraining cannot use an F3 wide target.")
    payload = {
        **metadata,
        "uses_f3_wide_target": False,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": sha256_payload(manifest),
    }
    Path(lock_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def verify_fold_lock(checkpoint_path, manifest_path, lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != lock["checkpoint_sha256"]:
        raise ValueError("Fold checkpoint hash mismatch.")
    if sha256_payload(manifest) != lock["manifest_sha256"]:
        raise ValueError("Fold manifest hash mismatch.")
    if manifest.get("uses_heldout_well_wide_target") is not False:
        raise ValueError("Held-out wide target use is forbidden.")
    return lock


def authorize_heldout_reference(
    checkpoint_path,
    manifest_path,
    lock_path,
    requested_well,
    requested_axis,
    requested_number,
):
    lock = verify_fold_lock(checkpoint_path, manifest_path, lock_path)
    if requested_well != lock["heldout_well"]:
        raise ValueError("Requested well is not the held-out well.")
    expected = (
        lock["heldout_inline"]
        if requested_axis == "inline"
        else lock["heldout_crossline"]
    )
    if int(requested_number) != int(expected):
        raise ValueError("Requested section is not authorized for this fold.")
    return lock
