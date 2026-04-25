import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from metageniuses.extraction.config import PreprocessConfig
from metageniuses.extraction.preprocess import preprocess_record
from metageniuses.extraction.schemas import SequenceRecord


class TestPreprocess(unittest.TestCase):
    def test_replaces_invalid_characters(self) -> None:
        cfg = PreprocessConfig(
            uppercase=True,
            allowed_chars="ACGTN",
            replace_invalid_with="N",
            max_invalid_fraction=0.5,
            min_length=1,
            max_length=50,
        )
        record = SequenceRecord(sequence_id="x1", sequence="acgtxx", metadata={})
        result = preprocess_record(record, cfg)
        self.assertIsNotNone(result.record)
        assert result.record is not None
        self.assertEqual(result.record.sequence, "ACGTNN")
        self.assertEqual(result.invalid_char_count, 2)

    def test_skips_when_invalid_fraction_too_high(self) -> None:
        cfg = PreprocessConfig(
            uppercase=True,
            allowed_chars="ACGTN",
            replace_invalid_with="N",
            max_invalid_fraction=0.1,
            min_length=1,
            max_length=50,
        )
        record = SequenceRecord(sequence_id="x2", sequence="NNNNX", metadata={})
        result = preprocess_record(record, cfg)
        self.assertIsNone(result.record)
        self.assertEqual(result.reason, "too_many_invalid_characters")


if __name__ == "__main__":
    unittest.main()

