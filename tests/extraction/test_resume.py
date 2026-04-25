import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import ExtractionConfig
from metageniuses.extraction.extractor import ResidualExtractionPipeline
from metageniuses.extraction.model_adapter import FakeModelAdapter


class TestResume(unittest.TestCase):
    def test_resume_continues_without_duplicate_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"sequence_id": "r1", "sequence": "ACGTAC"}),
                        json.dumps({"sequence_id": "r2", "sequence": "GGGTTT"}),
                        json.dumps({"sequence_id": "r3", "sequence": "TTAACG"}),
                    ]
                )
                + "\n"
            )
            base_payload = {
                "input": {
                    "path": str(input_path),
                    "format": "jsonl",
                    "sequence_key": "sequence",
                    "id_key": "sequence_id",
                },
                "preprocess": {
                    "max_length": 128,
                    "max_invalid_fraction": 1.0,
                },
                "model": {"model_id": "fake/model"},
                "layer_selection": {"layers": [2]},
                "runtime": {
                    "output_root": str(Path(tmpdir) / "out"),
                    "run_id": "resume_case",
                    "batch_size": 2,
                    "max_rows_per_shard": 10000,
                },
            }

            pipeline = ResidualExtractionPipeline()
            adapter = FakeModelAdapter(d_model=8, num_transformer_layers=4)

            first_payload = json.loads(json.dumps(base_payload))
            first_payload["runtime"]["max_reads"] = 2
            first_cfg = ExtractionConfig.from_dict(first_payload)
            artifact_root = pipeline.run(first_cfg, adapter=adapter)
            manifest1 = json.loads((artifact_root / "manifest.json").read_text())
            self.assertEqual(manifest1["stats"]["total_sequences_kept"], 2)

            second_payload = json.loads(json.dumps(base_payload))
            second_payload["runtime"]["max_reads"] = 3
            second_payload["runtime"]["resume"] = True
            second_cfg = ExtractionConfig.from_dict(second_payload)
            artifact_root_2 = pipeline.run(second_cfg, adapter=adapter)
            self.assertEqual(artifact_root, artifact_root_2)

            manifest2 = json.loads((artifact_root_2 / "manifest.json").read_text())
            self.assertEqual(manifest2["stats"]["total_sequences_kept"], 3)

            seq_ids = []
            with (artifact_root_2 / "sequences.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    seq_ids.append(json.loads(line)["sequence_id"])
            self.assertEqual(seq_ids, ["r1", "r2", "r3"])

    def test_existing_run_dir_requires_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "reads.jsonl"
            input_path.write_text(json.dumps({"sequence_id": "r1", "sequence": "ACGT"}) + "\n")

            payload = {
                "input": {"path": str(input_path), "format": "jsonl"},
                "preprocess": {"max_invalid_fraction": 1.0},
                "model": {"model_id": "fake/model"},
                "layer_selection": {"layers": [2]},
                "runtime": {
                    "output_root": str(Path(tmpdir) / "out"),
                    "run_id": "same_dir",
                    "batch_size": 1,
                    "max_reads": 1,
                },
            }
            cfg = ExtractionConfig.from_dict(payload)
            pipeline = ResidualExtractionPipeline()
            adapter = FakeModelAdapter(d_model=8, num_transformer_layers=4)
            pipeline.run(cfg, adapter=adapter)

            with self.assertRaises(ValueError):
                pipeline.run(cfg, adapter=adapter)


if __name__ == "__main__":
    unittest.main()

