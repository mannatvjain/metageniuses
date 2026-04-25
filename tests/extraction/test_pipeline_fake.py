import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import ExtractionConfig
from metageniuses.extraction.extractor import ResidualExtractionPipeline
from metageniuses.extraction.model_adapter import FakeModelAdapter


class TestPipelineFake(unittest.TestCase):
    def test_runs_with_selected_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"sequence_id": "r1", "sequence": "ACGTAC"}),
                        json.dumps({"sequence_id": "r2", "sequence": "GGGTTT"}),
                        json.dumps({"sequence_id": "r3", "sequence": "acgtnnxx"}),
                    ]
                )
                + "\n"
            )
            cfg = ExtractionConfig.from_dict(
                {
                    "input": {
                        "path": str(input_path),
                        "format": "jsonl",
                        "sequence_key": "sequence",
                        "id_key": "sequence_id",
                    },
                    "preprocess": {
                        "uppercase": True,
                        "allowed_chars": "ACGTUN",
                        "replace_invalid_with": "N",
                        "max_invalid_fraction": 0.30,
                        "min_length": 1,
                        "max_length": 128,
                        "strip_whitespace": True,
                    },
                    "model": {
                        "model_id": "fake/metagene-tiny",
                        "local_files_only": True,
                    },
                    "layer_selection": {
                        "last_n_layers": 2,
                    },
                    "runtime": {
                        "output_root": str(Path(tmpdir) / "out"),
                        "run_id": "test_run",
                        "batch_size": 2,
                        "max_rows_per_shard": 10000,
                    },
                }
            )

            pipeline = ResidualExtractionPipeline()
            adapter = FakeModelAdapter(num_transformer_layers=8, d_model=12)
            artifact_root = pipeline.run(cfg, adapter=adapter)

            manifest = json.loads((artifact_root / "manifest.json").read_text())
            self.assertEqual(manifest["layer_selection"]["selected_transformer_layers"], [7, 8])
            self.assertEqual(manifest["model"]["d_model"], 12)
            self.assertEqual(manifest["stats"]["total_sequences_kept"], 3)
            self.assertGreater(manifest["stats"]["total_rows_written"], 0)

            layer7_rows = manifest["layers"]["7"]["rows"]
            layer8_rows = manifest["layers"]["8"]["rows"]
            self.assertEqual(layer7_rows, layer8_rows)
            self.assertEqual(layer7_rows * 2, manifest["stats"]["total_rows_written"])


if __name__ == "__main__":
    unittest.main()

