"""Package one saved run without raw images or a Python environment."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def export_run_bundle(
    run_dir: str | Path,
    *,
    config_path: str | Path = "configs/option1_baselines.json",
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create a downloadable ZIP with one run, config, and split manifest."""
    run_root = _resolve(run_dir)
    if not (run_root / "run_config.json").is_file() or not (run_root / "comparison.json").is_file():
        raise FileNotFoundError(f"Saved run is incomplete: {run_root}")
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    if run_config.get("source_test_evaluated"):
        raise ValueError("A training bundle must not contain source-test scores.")

    config_file = _resolve(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    manifest_file = _resolve(manifest_path or config["manifest_path"])
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_file}")
    if output_path is None:
        archive_path = ROOT / "outputs" / "exports" / f"ft11-fer-{run_config['run_id']}.zip"
    else:
        archive_path = _resolve(output_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, bytes] = {
        "config/option1_baselines.json": config_file.read_bytes(),
        "data/fer2013_split_manifest.json": manifest_file.read_bytes(),
        "README.txt": (
            "FT11 FER2013 experiment bundle\n"
            f"Run: {run_config['run_id']}\n"
            f"Split SHA-256: {run_config['manifest_sha256']}\n"
            "Contains run metrics, histories, plots, selected rows, and checkpoints.\n"
            "Contains no raw images and no Python environment.\n"
        ).encode("utf-8"),
    }
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        relative = path.relative_to(run_root).as_posix()
        entries[f"run/{run_root.name}/{relative}"] = path.read_bytes()

    checksum_lines = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(entries.items())
    ]
    entries["SHA256SUMS.txt"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(name, content)
    return archive_path
