from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_system_page_starts_without_exception() -> None:
    app_path = Path(__file__).parent.parent / "src" / "fine_tuning_studio" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
