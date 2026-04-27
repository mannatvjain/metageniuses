"""
MetaGeniuses API backend.

Serves experiment results and feature explorer data from results/ directories.
Returns 404 if no data exists for a given endpoint.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MetaGeniuses API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_json(path: Path):
    """Load JSON file if it exists, raise 404 otherwise."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"No data at {path.name}")


# ---------------------------------------------------------------------------
# Feature Explorer
# ---------------------------------------------------------------------------

@app.get("/api/features")
def get_features():
    return _load_json(RESULTS_DIR / "features" / "features.json")


@app.get("/api/features/{feature_id}")
def get_feature(feature_id: int):
    return _load_json(RESULTS_DIR / "features" / f"feature_{feature_id}.json")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

EXPERIMENT_PATHS = {
    1: "organism_detectors/api_results.json",
    2: "linear_probe_pathogen/api_results.json",
    3: "sae_health_check/api_results.json",
    4: "sequence_umap/api_results.json",
    5: "feature_clustering/api_results.json",
    6: "cross_delivery/api_results.json",
}


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: int):
    path = EXPERIMENT_PATHS.get(experiment_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment {experiment_id}")
    return _load_json(RESULTS_DIR / path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
