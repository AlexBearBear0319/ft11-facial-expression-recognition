"""Create one duplicate-aware, deterministic FER2013 split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (48, 48)
PREPROCESSING = {
    "size": [48, 48],
    "channels": 1,
    "conversion": "Pillow convert('L')",
    "resize": "Pillow bilinear",
    "scale": "float32 pixels divided by 255",
    "augmentation": "none",
}


def canonical_label(folder_name: str) -> str:
    label = folder_name.strip().casefold()
    if label not in LABELS:
        raise ValueError(f"Unknown class folder: {folder_name}")
    return label


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    """Apply the same grayscale, resize, and scaling steps to every input."""
    image = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(values.astype(np.float32) / 255.0).unsqueeze(0)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load the exact grayscale 48x48 input used for hashing and training."""
    with Image.open(path) as image:
        return preprocess_pil_image(image)


def _scan_split(directory: Path, source_split: str) -> tuple[list[dict[str, Any]], list[str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"ImageFolder directory does not exist: {directory}")

    class_directories = sorted(path for path in directory.iterdir() if path.is_dir())
    found = {canonical_label(path.name) for path in class_directories}
    missing = sorted(set(LABELS) - found)
    if missing:
        raise ValueError(f"{directory} is missing class folders: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    unassigned_files: list[str] = []
    for class_directory in class_directories:
        label = canonical_label(class_directory.name)
        for path in sorted(class_directory.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(directory).as_posix()
            record: dict[str, Any] = {
                "source_split": source_split,
                "relative_path": relative_path,
                "label": label,
                "class_id": LABELS.index(label),
                "image_sha256": None,
                "split": "excluded",
                "exclusion_reason": None,
            }
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                record["exclusion_reason"] = "unsupported_file_type"
                record["audit_detail"] = path.suffix or "no file extension"
                records.append(record)
                continue
            try:
                with Image.open(path) as image:
                    pixels = np.asarray(
                        image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                record["image_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                record["exclusion_reason"] = "unreadable_image"
                record["audit_detail"] = f"{type(error).__name__}: {error}"
            records.append(record)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            unassigned_files.append(path.name)
    return records, unassigned_files


def _stratified_group_split(
    records_by_hash: dict[str, list[dict[str, Any]]],
    seed: int,
    validation_fraction: float,
) -> tuple[set[str], dict[str, Any]]:
    validation_hashes: set[str] = set()
    class_summary: dict[str, Any] = {}

    for class_id, label in enumerate(LABELS):
        group_hashes = sorted(
            image_hash
            for image_hash, group in records_by_hash.items()
            if group[0]["class_id"] == class_id
        )
        if not group_hashes:
            class_summary[label] = {"train_records": 0, "validation_records": 0, "validation_groups": 0}
            continue

        sizes = {image_hash: len(records_by_hash[image_hash]) for image_hash in group_hashes}
        total_records = sum(sizes.values())
        target = max(1, int(round(total_records * validation_fraction)))
        if len(group_hashes) == 1:
            class_summary[label] = {
                "train_records": total_records,
                "validation_records": 0,
                "validation_groups": 0,
                "note": "Only one exact-image group exists; it stays in train.",
            }
            continue

        rng = np.random.default_rng(np.random.SeedSequence([seed, class_id]))
        order = [group_hashes[index] for index in rng.permutation(len(group_hashes))]
        chosen: list[str] = []
        count = 0
        for index, image_hash in enumerate(order):
            if len(order) - index <= 1:
                break
            proposed = count + sizes[image_hash]
            if not chosen or abs(proposed - target) <= abs(count - target):
                chosen.append(image_hash)
                count = proposed
            elif count >= target:
                break

        validation_hashes.update(chosen)
        class_summary[label] = {
            "train_records": total_records - count,
            "validation_records": count,
            "validation_groups": len(chosen),
            "target_validation_records": target,
        }

    return validation_hashes, class_summary


def _counts(records: list[dict[str, Any]], split_key: str = "split") -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in ("train", "validation", "test", "excluded"):
        selected = [row for row in records if row.get(split_key) == split]
        by_split[split] = {
            "records": len(selected),
            "classes": {
                label: sum(1 for row in selected if row.get("label") == label)
                for label in LABELS
            },
        }
    return by_split


def _stable_manifest_hash(payload: dict[str, Any]) -> str:
    stable_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at_utc", "manifest_sha256"}
    }
    canonical = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(
    train_dir: str | Path,
    test_dir: str | Path,
    output_path: str | Path,
    source_id: str = "local-fer2013",
    seed: int = 42,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    if not 0.05 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")

    train_root = Path(train_dir).expanduser().resolve()
    test_root = Path(test_dir).expanduser().resolve()
    train_records, train_unassigned = _scan_split(train_root, "train")
    test_records, test_unassigned = _scan_split(test_root, "test")
    records = train_records + test_records

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image_hash = record.get("image_sha256")
        if image_hash:
            grouped[image_hash].append(record)

    conflicts: list[dict[str, Any]] = []
    cross_source_duplicates: list[dict[str, Any]] = []
    eligible_train_groups: dict[str, list[dict[str, Any]]] = {}

    for image_hash, group in sorted(grouped.items()):
        labels = sorted({row["label"] for row in group})
        sources = {row["source_split"] for row in group}
        paths = [f"{row['source_split']}/{row['relative_path']}" for row in group]

        if len(labels) > 1:
            conflicts.append({
                "image_sha256": image_hash,
                "labels": labels,
                "paths": paths,
                "source_test_rows_retained": sum(row["source_split"] == "test" for row in group),
                "source_train_rows_excluded": sum(row["source_split"] == "train" for row in group),
            })
            for row in group:
                if row["source_split"] == "test":
                    row["split"] = "test"
                else:
                    row["exclusion_reason"] = "exact_duplicate_has_conflicting_labels"
            continue

        if sources == {"train", "test"}:
            cross_source_duplicates.append({"image_sha256": image_hash, "label": labels[0], "paths": paths})
            for row in group:
                row["split"] = "test" if row["source_split"] == "test" else "excluded"
                if row["source_split"] == "train":
                    row["exclusion_reason"] = "exact_duplicate_also_exists_in_source_test"
            continue

        if sources == {"test"}:
            for row in group:
                row["split"] = "test"
            continue

        eligible_train_groups[image_hash] = group

    validation_hashes, class_split_summary = _stratified_group_split(
        eligible_train_groups, seed, validation_fraction
    )
    for image_hash, group in eligible_train_groups.items():
        split = "validation" if image_hash in validation_hashes else "train"
        for row in group:
            row["split"] = split

    # Preserve every discovered file in the manifest, including files that cannot be trained on.
    for row in records:
        if row.get("exclusion_reason") is None:
            row.pop("exclusion_reason", None)
    records.sort(key=lambda row: (row["source_split"], row["relative_path"].casefold()))

    duplicate_group_sizes = [len(group) for group in grouped.values() if len(group) > 1]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_id": source_id,
        "label_order": list(LABELS),
        "preprocessing": PREPROCESSING,
        "split_settings": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "source_test": "kept as supplied and excluded from model selection; unreadable or unsupported files remain listed as excluded",
        },
        "duplicate_policy": {
            "hash": "SHA-256 of the grayscale 48x48 pixels used by the models",
            "same_label_source_train_duplicates": "keep all rows in one split group",
            "train_and_source_test_duplicate": "keep source-test rows in test; exclude source-train copies",
            "conflicting_label_duplicate": "retain every source-test row in test; exclude matching source-train rows; exclude conflicting source-train-only groups from model splits; preserve and list all source paths",
            "unreadable_or_unsupported": "preserve the source file and list it as excluded with a reason",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_split_summary": class_split_summary,
        "duplicate_audit": {
            "duplicate_groups": len(duplicate_group_sizes),
            "records_in_duplicate_groups": sum(duplicate_group_sizes),
            "largest_duplicate_group": max(duplicate_group_sizes, default=1),
            "train_test_duplicate_groups": cross_source_duplicates,
            "conflicting_label_duplicate_groups": conflicts,
        },
        "unassigned_root_files": {"train": train_unassigned, "test": test_unassigned},
        "counts": _counts(records),
        "source_counts": {
            source_split: {
                "records": sum(row["source_split"] == source_split for row in records),
                "classes": {
                    label: sum(row["source_split"] == source_split and row["label"] == label for row in records)
                    for label in LABELS
                },
            }
            for source_split in ("train", "test")
        },
        "records": records,
    }
    payload["manifest_sha256"] = _stable_manifest_hash(payload)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_manifest_from_config(
    config_path: str | Path,
    train_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the configured split used by both the CLI and the notebook."""
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    source_root = Path(os.environ.get("FER2013_ROOT", config["source_root"])).expanduser()
    if not source_root.is_absolute():
        source_root = (ROOT / source_root).resolve()
    actual_train_dir = Path(train_dir).expanduser() if train_dir else source_root / "train"
    actual_test_dir = Path(test_dir).expanduser() if test_dir else source_root / "test"
    split_settings = config["split"]
    destination = Path(output_path or config["manifest_path"]).expanduser()
    if not destination.is_absolute():
        destination = (ROOT / destination).resolve()
    return build_manifest(
        actual_train_dir,
        actual_test_dir,
        destination,
        source_id=config["source_id"],
        seed=int(split_settings["seed"]),
        validation_fraction=float(split_settings["validation_fraction"]),
    )


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, Any]], train_dir: str | Path, test_dir: str | Path):
        self.rows = rows
        self.roots = {"train": Path(train_dir), "test": Path(test_dir)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = load_image_tensor(self.roots[row["source_split"]] / row["relative_path"])
        return image, int(row["class_id"])


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256", None)
    actual = _stable_manifest_hash(manifest)
    manifest["manifest_sha256"] = actual
    if expected != actual:
        raise ValueError(f"Manifest hash mismatch: expected {expected}, got {actual}")
    if manifest.get("label_order") != list(LABELS):
        raise ValueError("Manifest label order does not match the shared pipeline.")
    if manifest.get("preprocessing") != PREPROCESSING:
        raise ValueError("Manifest preprocessing does not match the shared pipeline.")
    return manifest


def validate_manifest_config(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Reject a manifest made with a different source or split configuration."""
    if manifest.get("source_id") != config["source_id"]:
        raise ValueError("Manifest source_id does not match the shared experiment configuration.")
    expected_split = config["split"]
    actual_split = manifest["split_settings"]
    if int(actual_split["seed"]) != int(expected_split["seed"]):
        raise ValueError("Manifest split seed does not match the shared experiment configuration.")
    if float(actual_split["validation_fraction"]) != float(expected_split["validation_fraction"]):
        raise ValueError("Manifest validation fraction does not match the shared experiment configuration.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Audit FER2013 folders and save the shared split manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/option1_baselines.json"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validation-fraction", type=float)
    args = parser.parse_args()
    if any(value is not None for value in (args.source_root, args.train_dir, args.test_dir, args.output, args.source_id, args.seed, args.validation_fraction)):
        config_file = args.config.expanduser()
        if not config_file.is_absolute():
            config_file = (ROOT / config_file).resolve()
        config = json.loads(config_file.read_text(encoding="utf-8"))
        source_root = args.source_root or Path(os.environ.get("FER2013_ROOT", config["source_root"]))
        train_dir = args.train_dir or source_root / "train"
        test_dir = args.test_dir or source_root / "test"
        split = config["split"]
        destination = args.output or config["manifest_path"]
        manifest = build_manifest(
            train_dir,
            test_dir,
            destination,
            source_id=args.source_id or config["source_id"],
            seed=args.seed if args.seed is not None else int(split["seed"]),
            validation_fraction=args.validation_fraction if args.validation_fraction is not None else float(split["validation_fraction"]),
        )
    else:
        manifest = build_manifest_from_config(args.config)
        destination = Path(args.config)
        config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
        destination = ROOT / config["manifest_path"]
    duplicate_audit = manifest["duplicate_audit"]
    excluded_reasons: dict[str, int] = {}
    for row in manifest["records"]:
        reason = row.get("exclusion_reason")
        if row["split"] == "excluded" and reason:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
    print(json.dumps({
        "manifest": str(destination),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_counts": manifest["source_counts"],
        "counts": manifest["counts"],
        "duplicate_audit": {
            "duplicate_groups": duplicate_audit["duplicate_groups"],
            "records_in_duplicate_groups": duplicate_audit["records_in_duplicate_groups"],
            "largest_duplicate_group": duplicate_audit["largest_duplicate_group"],
            "train_test_duplicate_groups": len(duplicate_audit["train_test_duplicate_groups"]),
            "conflicting_label_duplicate_groups": len(duplicate_audit["conflicting_label_duplicate_groups"]),
        },
        "excluded_records_by_reason": excluded_reasons,
        "class_split_summary": manifest["class_split_summary"],
    }, indent=2))


if __name__ == "__main__":
    _cli()
