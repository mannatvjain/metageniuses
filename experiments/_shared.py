"""
Shared helpers for experiment scripts.
"""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _candidate_sae_dirs():
    env_dir = None
    if "METAGENIUSES_SAE_DIR" in __import__("os").environ:
        env_dir = Path(__import__("os").environ["METAGENIUSES_SAE_DIR"]).expanduser()

    candidates = [
        env_dir,
        REPO_ROOT / "data" / "sae_model",
        REPO_ROOT.parent / "data" / "sae_model",
        REPO_ROOT.parent,
        Path.home() / "cs" / "biohack",
        Path("/Users/mannatvjain/Developer/metageniuses/data/sae_model"),
    ]
    return [path for path in candidates if path is not None]


def resolve_sae_dir():
    """
    Return a directory containing features.npy, sequence_ids.json, and sae_config.json.
    """
    required = ("features.npy", "sequence_ids.json", "sae_config.json")
    for candidate in _candidate_sae_dirs():
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(
        "Could not find SAE artifacts. Set METAGENIUSES_SAE_DIR or place "
        "features.npy, sequence_ids.json, and sae_config.json in data/sae_model."
    )


def resolve_analysis_dir():
    env_dir = None
    if "METAGENIUSES_ANALYSIS_DIR" in __import__("os").environ:
        env_dir = Path(__import__("os").environ["METAGENIUSES_ANALYSIS_DIR"]).expanduser()

    candidates = [
        env_dir,
        REPO_ROOT / "results" / "analyze",
        Path.home()
        / "Documents"
        / "Codex"
        / "2026-04-25-find-the-sae-files-in-the"
        / "sae_analysis_source_binary_signature",
    ]
    required = ("feature_stats.csv", "kmer_enrichment.csv", "probe_metrics.csv")
    for candidate in candidates:
        if candidate is not None and all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(
        "Could not find analysis outputs. Set METAGENIUSES_ANALYSIS_DIR or "
        "generate the SAE analysis outputs first."
    )


def load_label_map(jsonl_path):
    labels = {}
    sequences = {}
    metadata = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["sequence_id"]] = int(row["source"])
            sequences[row["sequence_id"]] = row.get("sequence", "")
            metadata[row["sequence_id"]] = row
    return labels, sequences, metadata


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
