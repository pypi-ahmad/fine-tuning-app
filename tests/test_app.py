from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_system_page_starts_without_exception() -> None:
    app = AppTest.from_file(str(Path("src/fine_tuning_studio/app.py")), default_timeout=30).run()
    assert not app.exception
