import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest

from bank_marketing.ui.pages import PAGES

ROOT = Path(__file__).resolve().parents[2]
specification = importlib.util.spec_from_file_location("restore_demo", ROOT / "scripts/restore_demo.py")
restore_demo = importlib.util.module_from_spec(specification)
specification.loader.exec_module(restore_demo)

def test_default_test_temp_parent_exists_in_fresh_checkout(tmp_path):
    configure = runpy.run_path(ROOT / "tests/conftest.py")["pytest_configure"]
    configuration = SimpleNamespace(rootpath=tmp_path / "fresh", option=SimpleNamespace(basetemp=None))
    configure(configuration)
    selected = Path(configuration.option.basetemp)
    assert selected.parent == configuration.rootpath / ".pytest_cache"
    assert selected.parent.is_dir()
    selected.mkdir()

def test_explicit_test_temp_path_is_preserved(tmp_path):
    configure = runpy.run_path(ROOT / "tests/conftest.py")["pytest_configure"]
    selected = str(tmp_path / "custom")
    configuration = SimpleNamespace(rootpath=tmp_path / "fresh", option=SimpleNamespace(basetemp=selected))
    configure(configuration)
    assert configuration.option.basetemp == selected
    assert not (configuration.rootpath / ".pytest_cache").exists()


@pytest.fixture
def bundle_factory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "demo").mkdir(parents=True)
    monkeypatch.setattr(restore_demo, "ROOT", project)

    def create(entries):
        archive = project / "demo/deporank_demo_run.zip"
        with zipfile.ZipFile(archive, "w") as writer:
            for name, content in entries:
                member = zipfile.ZipInfo(name) if isinstance(name, str) else name
                if isinstance(name, str):
                    member.filename = name
                writer.writestr(member, content)
        manifest = {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "run_id": "trusted_demo"}
        (project / "demo/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return archive

    return create


def test_english_navigation_contract():
    assert PAGES == ("Overview", "Data and KDD", "Model Comparison", "Single Prediction", "Batch Priorities", "Evaluation and Interpretation", "Documentation")


def test_restore_is_idempotent_and_preserves_bytes(bundle_factory, tmp_path):
    bundle = bundle_factory([("artifacts/active_run.json", b'{"run_id":"trusted_demo"}')])
    target = tmp_path / "restored"
    assert restore_demo.restore(target, bundle)["files_restored"] == 1
    assert restore_demo.restore(target, bundle)["files_restored"] == 0
    assert (target / "artifacts/active_run.json").read_bytes() == b'{"run_id":"trusted_demo"}'


def test_restore_refuses_overwrite_before_any_changes(bundle_factory, tmp_path):
    bundle = bundle_factory([("artifacts/a.json", b"new"), ("artifacts/z.json", b"expected")])
    target = tmp_path / "restored"
    (target / "artifacts").mkdir(parents=True)
    (target / "artifacts/z.json").write_bytes(b"different")
    with pytest.raises(FileExistsError):
        restore_demo.restore(target, bundle)
    assert not (target / "artifacts/a.json").exists()
    assert (target / "artifacts/z.json").read_bytes() == b"different"


def test_restore_rejects_checksum_mismatch(bundle_factory, tmp_path):
    bundle = bundle_factory([("artifacts/run.json", b"safe")])
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        restore_demo.restore(target, bundle)
    assert not target.exists()


@pytest.mark.parametrize("name", ["artifacts/../../escape.json", "/artifacts/absolute.json", "data/not_an_artifact.json", "artifacts\\escape.json", "artifacts/C:escape.json"])
def test_restore_rejects_unsafe_members(bundle_factory, tmp_path, name):
    bundle = bundle_factory([(name, b"unsafe")])
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe"):
        restore_demo.restore(target, bundle)
    assert not list(target.rglob("*.json"))


def test_restore_rejects_symlink_archive_member(bundle_factory, tmp_path):
    member = zipfile.ZipInfo("artifacts/link")
    member.create_system = 3
    member.external_attr = (0o120777 << 16)
    bundle = bundle_factory([(member, "../../escape")])
    with pytest.raises(ValueError, match="Unsafe"):
        restore_demo.restore(tmp_path / "restored", bundle)


def test_restore_rejects_duplicate_members(bundle_factory, tmp_path):
    with pytest.warns(UserWarning, match="Duplicate"):
        bundle = bundle_factory([("artifacts/run.json", b"first"), ("artifacts/run.json", b"second")])
    with pytest.raises(ValueError, match="Unsafe"):
        restore_demo.restore(tmp_path / "restored", bundle)


def test_report_contains_six_required_chapters_and_public_links():
    from pypdf import PdfReader

    report = ROOT / "reports/final_report.pdf"
    if not report.exists():
        pytest.skip("Generate the academic report first")
    reader = PdfReader(report)
    text = "\n".join(page.extract_text() for page in reader.pages)
    for chapter in ("Chapter 1 Introduction", "Chapter 2 Theoretical Basis", "Chapter 3 Methods", "Chapter 4 Implementation and Evaluation", "Chapter 5 Conclusion and Suggestions", "Chapter 6 Appendix"):
        assert chapter in text
    targets = [str(annotation.get_object().get("/A", {}).get("/URI", "")) for page in reader.pages for annotation in page.get("/Annots", [])]
    assert any("presentation/slides.pptx" in target for target in targets)
    assert any("data/raw/bank-additional-full.csv" in target for target in targets)
    assert "https://github.com/DlegendZ/DepoRank" in targets


def test_report_metrics_match_frozen_evidence():
    from pypdf import PdfReader

    report = ROOT / "reports/final_report.pdf"
    if not report.exists():
        pytest.skip("Generate the academic report first")
    text = "\n".join(page.extract_text() for page in PdfReader(report).pages)
    assert "0.5076" in text and "62.14%" in text and "0.2546" in text
    assert "512" in text and "824" in text
    assert "30.83%" in text and "4.81%" in text


def test_deck_contains_editable_tables_charts_and_notes():
    from xml.etree import ElementTree

    deck = ROOT / "presentation/slides.pptx"
    if not deck.exists():
        pytest.skip("Generate the stakeholder deck first")
    with zipfile.ZipFile(deck) as archive:
        slides = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
        assert len(slides) == 16
        for number in (7, 10):
            document = ElementTree.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            assert document.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}tbl") is not None
        notes = [name for name in archive.namelist() if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
        assert len(notes) == 16
        assert len([name for name in archive.namelist() if name.endswith(".xlsx")]) == 4
