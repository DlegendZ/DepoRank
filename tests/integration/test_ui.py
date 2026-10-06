from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bank_marketing.ml.registry import Registry
from bank_marketing.storage import atomic_bytes
from bank_marketing.ui.pages import PAGES

ROOT = Path(__file__).resolve().parents[2]


def _active_app():
    if not (ROOT / "artifacts/active_run.json").exists():
        pytest.skip("Generate trusted academic run to exercise active UI")
    return AppTest.from_file(ROOT / "app.py", default_timeout=60).run()


def _clean(app):
    assert not list(app.exception)
    assert not [e.value for e in app.error if "INTERNAL_ERROR" in e.value]


def test_seven_active_pages():
    app = _active_app()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        _clean(app)


def test_single_and_batch_capacity_without_retraining():
    app = _active_app()
    app.sidebar.radio[0].set_value("Single Prediction").run()
    next(button for button in app.button if button.label == "Predict").click().run()
    _clean(app)
    assert app.session_state["single_result"]["rows"][0]["p_yes"] >= 0
    app.sidebar.radio[0].set_value("Batch Priorities").run()
    app.selectbox[0].set_value("demo_valid.csv").run()
    next(button for button in app.button if button.label == "Score batch").click().run()
    _clean(app)
    batch_id = app.session_state["batch_scores"]["batch_id"]
    assert any("exactly 1 priorities" in value.value for value in app.success)
    app.slider[0].set_value(0.50).run()
    assert app.session_state["batch_scores"]["batch_id"] == batch_id
    assert any("exactly 2 priorities" in value.value for value in app.success)
    app.selectbox[0].set_value("demo_invalid.csv").run()
    assert "batch_scores" not in app.session_state
    assert any("entire batch rejected" in value.value for value in app.error)
    assert not [button for button in app.get("download_button") if getattr(button, "label", "") == "Download priorities.csv"]


def test_empty_registry_all_pages_remain_safe(tmp_path):
    script = f"from pathlib import Path\nfrom bank_marketing.ui.pages import render_app\nrender_app(Path({str(tmp_path)!r}))"
    app = AppTest.from_string(script, default_timeout=60).run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        _clean(app)
        assert any("MODEL_NOT_READY" in e.value for e in app.error)
    app.sidebar.radio[0].set_value("Single Prediction").run()
    assert next(button for button in app.button if button.label == "Predict").disabled


def test_tampered_active_pointer_stops_prediction(tmp_path):
    root = tmp_path / "app"
    atomic_bytes(root / "artifacts/active_run.json", b'{"run_id":"../escape","winner_model_id":"x","selection_lock_sha256":"x"}')
    script = f"from pathlib import Path\nfrom bank_marketing.ui.pages import render_app\nrender_app(Path({str(root)!r}))"
    app = AppTest.from_string(script, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Single Prediction").run()
    _clean(app)
    assert any("ARTIFACT_INTEGRITY_ERROR" in e.value for e in app.error)
    assert next(button for button in app.button if button.label == "Predict").disabled


def test_public_demo_and_random_ranking_na():
    app = _active_app()
    app.sidebar.radio[0].set_value("Batch Priorities").run()
    app.checkbox[0].set_value(True).run()
    app.selectbox[0].set_value("Labeled public subset").run()
    next(button for button in app.button if button.label == "Score batch").click().run()
    _clean(app)
    assert len(app.session_state["batch_scores"]["rows"]) == 100
    assert list(app.json)
    app.sidebar.radio[0].set_value("Evaluation and Interpretation").run()
    app.selectbox[1].set_value("random_ranking").run()
    _clean(app)
    assert any("N/A: random ranking baseline" in i.value for i in app.info)

def test_single_input_change_invalidates_old_score():
    app = _active_app()
    app.sidebar.radio[0].set_value('Single Prediction').run()
    next(b for b in app.button if b.label == 'Predict').click().run()
    assert 'single_result' in app.session_state
    app.number_input[0].set_value(41).run()
    _clean(app)
    assert 'single_result' not in app.session_state
@pytest.mark.parametrize('pointer', [b'{', b'[]', b'{}'])
def test_corrupt_pointer_ui_explains_integrity_failure(tmp_path, pointer):
    atomic_bytes(tmp_path / 'artifacts/active_run.json', pointer)
    script = f'from pathlib import Path\nfrom bank_marketing.ui.pages import render_app\nrender_app(Path({str(tmp_path)!r}))'
    app = AppTest.from_string(script, default_timeout=60).run()
    _clean(app)
    assert any('ARTIFACT_INTEGRITY_ERROR' in e.value for e in app.error)
def test_cleanup_resets_upload_and_demo_inputs():
    app = _active_app()
    app.sidebar.radio[0].set_value('Batch Priorities').run()
    app.selectbox[0].set_value('demo_valid.csv').run()
    next(b for b in app.button if b.label == 'Score batch').click().run()
    assert 'batch_scores' in app.session_state
    app.sidebar.radio[0].set_value('Documentation').run()
    next(b for b in app.button if b.label == 'Clear session data').click().run()
    assert 'batch_scores' not in app.session_state
    app.sidebar.radio[0].set_value('Batch Priorities').run()
    _clean(app)
    assert app.selectbox[0].value == 'No example'
    assert app.get('file_uploader')[0].proto.id.endswith('upload_1')
def test_evaluation_renders_without_report_command(tmp_path):
    if not (ROOT / 'artifacts/active_run.json').exists():
        pytest.skip('Trusted active run required')
    from bank_marketing.storage import read_json, write_json
    registry = Registry(ROOT / 'artifacts')
    ref = registry.active()
    source = registry.run_path(ref.run_id)
    write_json(tmp_path / 'selection_lock.json', read_json(source / 'selection_lock.json'))
    for model in ('decision_tree', 'random_ranking'):
        write_json(tmp_path / f'evaluations/selection_{model}.json', read_json(source / f'evaluations/selection_{model}.json'))
    script = f'from pathlib import Path\nfrom bank_marketing.ml.registry import Registry\nfrom bank_marketing.ui.pages import evaluation\nr=Registry({str(ROOT / "artifacts")!r}).active()\nevaluation(Path({str(ROOT)!r}),Path({str(tmp_path)!r}),r)'
    app = AppTest.from_string(script, default_timeout=60).run()
    _clean(app)
    assert len(app.get('plotly_chart')) == 4
    app.selectbox[1].set_value('random_ranking').run()
    _clean(app)
    assert any('not probabilities' in value.value for value in app.info)
def test_demo_metrics_use_exact_selected_k():
    import json
    app = _active_app()
    app.sidebar.radio[0].set_value('Batch Priorities').run()
    app.checkbox[0].set_value(True).run()
    app.selectbox[0].set_value('Labeled public subset').run()
    app.radio[0].set_value('Count K').run()
    app.number_input[0].set_value(14).run()
    next(b for b in app.button if b.label == 'Score batch').click().run()
    _clean(app)
    assert json.loads(app.json[0].value)['capacity_k'] == 14
