from __future__ import annotations

import argparse
from dataclasses import replace

from .config import ExtractionConfig
from .extractor import ResidualExtractionPipeline
from .model_adapter import FakeModelAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metageniuses-extract",
        description="Extract hidden-state residual vectors for SAE training.",
    )
    parser.add_argument("--config", required=True, help="Path to extraction JSON config.")
    parser.add_argument(
        "--adapter",
        choices=["transformers", "fake"],
        default="transformers",
        help="Model adapter backend.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run with the same run_id/output_root.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = ExtractionConfig.from_json_file(args.config)
    if args.resume:
        cfg = replace(cfg, runtime=replace(cfg.runtime, resume=True))
    pipeline = ResidualExtractionPipeline()

    adapter = None
    if args.adapter == "fake":
        adapter = FakeModelAdapter()

    artifact_root = pipeline.run(cfg=cfg, adapter=adapter)
    print(f"Extraction completed: {artifact_root}")


if __name__ == "__main__":
    main()
