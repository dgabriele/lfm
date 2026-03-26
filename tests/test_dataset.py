"""Tests for the dataset generation and reader modules."""

from __future__ import annotations

import pytest
from pathlib import Path

from lfm.data.dataset.config import DatasetGenerateConfig, LLMGateConfig, ProcessedSample
from lfm.data.dataset.manifest import DatasetManifest
from lfm.data.loaders.base import RawSample
from lfm.data.sanitize import SanitizeConfig


class TestRawSample:
    def test_namedtuple_fields(self):
        s = RawSample("eng", "Hello world", "leipzig", "eng_news.txt")
        assert s.language == "eng"
        assert s.text == "Hello world"
        assert s.source == "leipzig"
        assert s.source_file == "eng_news.txt"

    def test_tuple_unpacking(self):
        s = RawSample("deu", "Hallo Welt", "leipzig", "deu_news.txt")
        lang, text, source, source_file = s
        assert lang == "deu"
        assert text == "Hallo Welt"


class TestProcessedSample:
    def test_typed_dict(self):
        s = ProcessedSample(
            seq=0,
            language="eng",
            source="leipzig",
            source_file="eng_news.txt",
            raw="Hello world",
            ipa="hɛloʊ wɝld",
            ipa_length=11,
        )
        assert s["seq"] == 0
        assert s["language"] == "eng"
        assert s["ipa_length"] == 11


class TestDatasetGenerateConfig:
    def test_defaults(self):
        cfg = DatasetGenerateConfig(source="leipzig")
        assert cfg.source == "leipzig"
        assert cfg.max_samples == 50000
        assert cfg.min_samples == 100
        assert cfg.seed == 42
        assert isinstance(cfg.sanitize, SanitizeConfig)
        assert isinstance(cfg.llm_gate, LLMGateConfig)

    def test_custom_sanitize(self):
        cfg = DatasetGenerateConfig(
            source="leipzig",
            sanitize=SanitizeConfig(number_policy="reject"),
        )
        assert cfg.sanitize.number_policy == "reject"

    def test_llm_gate_disabled(self):
        cfg = DatasetGenerateConfig(
            source="leipzig",
            llm_gate=LLMGateConfig(enabled=False),
        )
        assert not cfg.llm_gate.enabled


class TestDatasetManifest:
    def test_create(self):
        m = DatasetManifest.create(
            name="test",
            description="Test dataset",
            sources=["leipzig"],
            languages={"eng": 100, "deu": 200},
            total_samples=300,
            rejected_samples=50,
            sanitize_config={"number_policy": "reject"},
            generate_config={"source": "leipzig"},
        )
        assert m.name == "test"
        assert m.total_samples == 300
        assert m.languages["deu"] == 200
        assert m.created_at != ""

    def test_save_and_load(self, tmp_path: Path):
        m = DatasetManifest.create(
            name="roundtrip",
            description="Test roundtrip",
            sources=["leipzig"],
            languages={"eng": 42},
            total_samples=42,
            rejected_samples=10,
            sanitize_config={},
            generate_config={},
        )
        path = tmp_path / "manifest.yaml"
        m.save(path)
        assert path.is_file()

        loaded = DatasetManifest.load(path)
        assert loaded.name == "roundtrip"
        assert loaded.total_samples == 42
        assert loaded.languages == {"eng": 42}


class TestDatasetReader:
    """Tests that require h5py for HDF5 I/O."""

    @pytest.fixture
    def dataset_dir(self, tmp_path: Path):
        """Create a minimal dataset for testing."""
        h5py = pytest.importorskip("h5py")
        yaml = pytest.importorskip("yaml")

        # Write samples.h5
        h5_path = tmp_path / "samples.h5"
        samples = [
            {"seq": 0, "language": "eng", "source": "test", "source_file": "test.txt",
             "raw": "Hello world", "ipa": "hɛloʊ wɝld", "ipa_length": 11},
            {"seq": 1, "language": "eng", "source": "test", "source_file": "test.txt",
             "raw": "Good morning", "ipa": "ɡʊd mɔɹnɪŋ", "ipa_length": 11},
            {"seq": 2, "language": "deu", "source": "test", "source_file": "deu.txt",
             "raw": "Hallo Welt", "ipa": "haloː vɛlt", "ipa_length": 10},
        ]

        str_dt = h5py.string_dtype()
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("samples")
            grp.create_dataset("seq", data=[s["seq"] for s in samples], dtype="int64")
            grp.create_dataset("ipa_length", data=[s["ipa_length"] for s in samples], dtype="int32")
            for field in ("language", "source", "source_file", "raw", "ipa"):
                grp.create_dataset(
                    field,
                    data=[s[field].encode("utf-8") for s in samples],
                    dtype=str_dt,
                )

        # Write manifest.yaml
        manifest = DatasetManifest.create(
            name="test",
            description="Test dataset",
            sources=["test"],
            languages={"eng": 2, "deu": 1},
            total_samples=3,
            rejected_samples=0,
            sanitize_config={},
            generate_config={},
        )
        manifest.save(tmp_path / "manifest.yaml")

        return tmp_path

    def test_reader_init(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        assert len(reader) == 3

    def test_reader_manifest(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        m = reader.manifest
        assert m.name == "test"
        assert m.total_samples == 3

    def test_load_ipa_tuples(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        tuples = reader.load_ipa_tuples()
        assert len(tuples) == 3
        langs = {t[0] for t in tuples}
        assert "eng" in langs
        assert "deu" in langs

    def test_load_ipa_tuples_filtered(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        tuples = reader.load_ipa_tuples(languages=["eng"])
        assert len(tuples) == 2
        assert all(t[0] == "eng" for t in tuples)

    def test_load_ipa_tuples_capped(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        tuples = reader.load_ipa_tuples(max_samples_per_language=1)
        # 1 eng + 1 deu
        assert len(tuples) == 2

    def test_iter_samples(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        samples = list(reader.iter_samples())
        assert len(samples) == 3
        assert samples[0]["language"] == "eng"
        assert samples[2]["language"] == "deu"

    def test_languages(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        assert reader.languages() == ["deu", "eng"]

    def test_missing_dir(self):
        from lfm.data.dataset.reader import DatasetReader

        with pytest.raises(FileNotFoundError):
            DatasetReader("/nonexistent/path")

    def test_repr(self, dataset_dir: Path):
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(dataset_dir)
        r = repr(reader)
        assert "3 samples" in r
        assert "2 languages" in r
