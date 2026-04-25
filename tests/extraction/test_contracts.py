import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import ExtractionConfig
from metageniuses.extraction.contracts import iter_layer_batches, load_manifest
from metageniuses.extraction.extractor import ResidualExtractionPipeline
from metageniuses.extraction.model_adapter import FakeModelAdapter


class TestContracts(unittest.TestCase):
    def test_iter_layer_batches_matches_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"sequence_id": "r1", "sequence": "ACGT"}),
                        json.dumps({"sequence_id": "r2", "sequence": "TTAA"}),
                    ]
                )
                + "\n"
            )
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {"path": str(input_path), "format": "jsonl"},
                    "preprocess": {
                        "max_length": 64,
                        "max_invalid_fraction": 1.0,
                    },
                    "model": {"model_id": "fake/model"},
                    "layer_selection": {"layers": [2]},
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "contracts_run",
                        "batch_size": 2,
                    },
                }
            )

            pipeline = ResidualExtractionPipeline()
            artifact_root = pipeline.run(cfg, adapter=FakeModelAdapter(d_model=10, num_transformer_layers=4))
            manifest = load_manifest(artifact_root)
            expected_rows = manifest["layers"]["2"]["rows"]

            observed_rows = 0
            for vectors, metadata in iter_layer_batches(artifact_root, transformer_layer=2, batch_size=3):
                self.assertEqual(len(vectors), len(metadata))
                self.assertLessEqual(len(vectors), 3)
                for vector in vectors:
                    self.assertEqual(len(vector), 10)
                observed_rows += len(vectors)

            self.assertEqual(expected_rows, observed_rows)


if __name__ == "__main__":
    unittest.main()

