"""Check the shared data, training, checkpoint, and frozen evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fer.data import LABELS, build_manifest, build_manifest_from_config, read_manifest
from src.fer.evaluate import evaluate_checkpoints
from src.fer.export_bundle import export_run_bundle
from src.fer.models import MODEL_IDS, make_model, parameter_count
from src.fer.train import run_experiments


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, size=(48, 48), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="ft11-fer-smoke-") as temp:
        root = Path(temp)
        train_dir, test_dir = root / "train", root / "test"
        first_train: dict[str, Path] = {}
        first_test: dict[str, Path] = {}

        for label_index, label in enumerate(LABELS):
            for index in range(4):
                image_path = train_dir / label / f"train_{index}.png"
                _write_image(image_path, seed=label_index * 100 + index)
                if index == 0:
                    first_train[label] = image_path
            for index in range(2):
                image_path = test_dir / label / f"test_{index}.png"
                _write_image(image_path, seed=10_000 + label_index * 100 + index)
                if index == 0:
                    first_test[label] = image_path

        _copy_image(first_train["happy"], train_dir / "happy" / "same_image_copy.png")
        _copy_image(first_test["fear"], train_dir / "fear" / "also_in_test.png")
        conflict = root / "conflict.png"
        _write_image(conflict, seed=999_991)
        _copy_image(conflict, train_dir / "angry" / "conflict.png")
        _copy_image(conflict, train_dir / "disgust" / "conflict.png")
        _copy_image(conflict, test_dir / "angry" / "conflict_a.png")
        _copy_image(conflict, test_dir / "disgust" / "conflict_d.png")

        first_manifest_path = root / "manifest-one.json"
        second_manifest_path = root / "manifest-two.json"
        first = build_manifest(train_dir, test_dir, first_manifest_path, "smoke-data", seed=42)
        second = build_manifest(train_dir, test_dir, second_manifest_path, "smoke-data", seed=42)
        assert first["manifest_sha256"] == second["manifest_sha256"]
        loaded_manifest = read_manifest(first_manifest_path)
        assert loaded_manifest["manifest_sha256"] == first["manifest_sha256"]

        active_hashes = {
            split: {row["image_sha256"] for row in first["records"] if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        assert active_hashes["train"].isdisjoint(active_hashes["validation"])
        assert active_hashes["train"].isdisjoint(active_hashes["test"])
        assert active_hashes["validation"].isdisjoint(active_hashes["test"])

        conflict_hash = hashlib.sha256(
            np.asarray(
                Image.open(conflict).convert("L").resize((48, 48), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            ).tobytes()
        ).hexdigest()
        conflict_rows = [row for row in first["records"] if row.get("image_sha256") == conflict_hash]
        assert sum(row["source_split"] == "test" and row["split"] == "test" for row in conflict_rows) == 2
        assert sum(row["source_split"] == "train" and row["split"] == "excluded" for row in conflict_rows) == 2

        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "source_root": str(root),
            "source_id": "smoke-data",
            "manifest_path": str(first_manifest_path),
            "output_root": str(root / "run-output"),
            "split": {"seed": 42, "validation_fraction": 0.15},
            "training": {
                "seed": 42,
                "epochs": 1,
                "batch_size": 16,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "patience": 1,
                "selection_metric": "validation_macro_f1",
                "tie_break": "lower_validation_loss",
            },
        }, indent=2), encoding="utf-8")

        configured_manifest = build_manifest_from_config(
            config_path,
            train_dir=train_dir,
            test_dir=test_dir,
            output_path=root / "manifest-from-config.json",
        )
        assert configured_manifest["manifest_sha256"] == first["manifest_sha256"]

        result = run_experiments(
            config_path,
            model_ids=list(MODEL_IDS),
            epochs_override=1,
            train_dir_override=train_dir,
            test_dir_override=test_dir,
            manifest_override=first_manifest_path,
        )
        assert result["source_test_evaluated"] is False
        assert set(result["models"]) == set(MODEL_IDS)
        assert parameter_count(make_model("cnn")) == 24_115
        run_dir = root / "run-output" / result["run_id"]
        assert (run_dir / "run_config.json").is_file()
        assert (run_dir / "comparison.csv").is_file()
        assert (run_dir / "class_counts.png").is_file()

        checkpoint_hashes = {
            model_id: _file_sha256(run_dir / "models" / model_id / "best.pt")
            for model_id in MODEL_IDS
        }
        original_adam, original_sgd = torch.optim.Adam, torch.optim.SGD

        def optimizer_not_allowed(*args, **kwargs):
            raise AssertionError("Frozen evaluation must not create an optimizer.")

        torch.optim.Adam = optimizer_not_allowed
        torch.optim.SGD = optimizer_not_allowed
        try:
            validation_result = evaluate_checkpoints(
                config_path,
                run_dir,
                split="validation",
                train_dir_override=train_dir,
                test_dir_override=test_dir,
                manifest_override=first_manifest_path,
                subset_csv=run_dir / "validation_subset.csv",
                output_dir=root / "validation-evaluation",
            )
        finally:
            torch.optim.Adam, torch.optim.SGD = original_adam, original_sgd

        for model_id in MODEL_IDS:
            trained = result["models"][model_id]["validation"]
            rescored = validation_result["models"][model_id]["metrics"]
            assert trained["accuracy"] == rescored["accuracy"]
            assert trained["macro_f1"] == rescored["macro_f1"]
            assert trained["confusion_matrix"] == rescored["confusion_matrix"]
            assert abs(trained["loss"] - rescored["loss"]) < 1e-6
            model_dir = run_dir / "models" / model_id
            assert (model_dir / "training_history.csv").is_file()
            assert (model_dir / "training_curves.png").is_file()
            assert (root / "validation-evaluation" / "models" / model_id / "validation_confusion_matrix.csv").is_file()
        assert all(
            checkpoint_hashes[model_id] == _file_sha256(run_dir / "models" / model_id / "best.pt")
            for model_id in MODEL_IDS
        )

        bundle_path = export_run_bundle(
            run_dir,
            config_path=config_path,
            manifest_path=first_manifest_path,
            output_path=root / "run-bundle.zip",
        )
        with zipfile.ZipFile(bundle_path) as archive:
            bundle_names = set(archive.namelist())
            assert "config/option1_baselines.json" in bundle_names
            assert "data/fer2013_split_manifest.json" in bundle_names
            assert "SHA256SUMS.txt" in bundle_names
            assert any(name.endswith("/models/resnet18/best.pt") for name in bundle_names)
            assert not any(".venv/" in name or name.endswith(".png.raw") for name in bundle_names)
            assert not any(name.startswith("data/") and "/" in name[len("data/"):] for name in bundle_names if name != "data/fer2013_split_manifest.json")
            checksums = archive.read("SHA256SUMS.txt").decode("utf-8").splitlines()
            for line in checksums:
                digest, member = line.split("  ", 1)
                assert hashlib.sha256(archive.read(member)).hexdigest() == digest

        try:
            evaluate_checkpoints(
                config_path,
                run_dir,
                split="test",
                train_dir_override=train_dir,
                test_dir_override=test_dir,
                manifest_override=first_manifest_path,
                output_dir=root / "blocked-test-evaluation",
            )
        except ValueError as error:
            assert "--confirm-final-test" in str(error)
        else:
            raise AssertionError("Evaluation must block test scoring without explicit confirmation.")

        cli_output = root / "cli-test-evaluation"
        command = [
            sys.executable, "-m", "src.fer.evaluate",
            "--config", str(config_path),
            "--checkpoint-dir", str(run_dir),
            "--split", "test",
            "--confirm-final-test",
            "--train-dir", str(train_dir),
            "--test-dir", str(test_dir),
            "--manifest", str(first_manifest_path),
            "--output-dir", str(cli_output),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        cli_results = json.loads((cli_output / "comparison.json").read_text(encoding="utf-8"))
        assert cli_results["evaluation_split"] == "test"
        assert cli_results["source_test_evaluated"] is True
        assert "optimizer" not in completed.stdout.casefold()
        assert all(
            checkpoint_hashes[model_id] == _file_sha256(run_dir / "models" / model_id / "best.pt")
            for model_id in MODEL_IDS
        )

        return {
            "manifest_hash_repeatable": True,
            "config_and_notebook_manifest_share_split_settings": True,
            "active_split_hashes_do_not_overlap": True,
            "source_test_retained_for_conflict_groups": True,
            "models_trained_for_one_synthetic_epoch": list(MODEL_IDS),
            "cnn_parameter_count": parameter_count(make_model("cnn")),
            "validation_evaluation_matches_training_checkpoint": True,
            "evaluation_did_not_construct_optimizer_or_change_checkpoints": True,
            "test_evaluation_cli_used_synthetic_data_only": True,
            "run_bundle_contains_checkpoint_config_manifest_and_checksums": True,
            "source_test_evaluated": "synthetic fixture only",
        }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
