import json
from pathlib import Path

import pytest

from fine_tuning_studio.evals import EvaluationManifest, compare_reports, write_evaluation_report


def test_evaluation_reports_and_comparison(tmp_path: Path) -> None:
    paths = write_evaluation_report(
        tmp_path, EvaluationManifest(model="org/model", tasks=["mmlu"]), {"mmlu": 0.5}
    )
    assert set(paths) == {"json", "csv", "html"}
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["scores"]["mmlu"] == 0.5
    assert compare_reports([{"mmlu": 0.5}, {"mmlu": 0.75}]) == {"mmlu": [0.5, 0.75]}
    with pytest.raises(ValueError, match="two and four"):
        compare_reports([{"mmlu": 0.5}])
