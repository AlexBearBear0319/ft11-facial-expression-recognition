"""Create a downloadable archive for one saved FT11 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fer.export_bundle import export_run_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a saved run, config, split manifest, and checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/option1_baselines.json"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = export_run_bundle(
        args.run_dir,
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
    )
    print(f"Run bundle: {output}")


if __name__ == "__main__":
    main()
