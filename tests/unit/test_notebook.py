"""The Colab notebook: a valid notebook whose Python and shell cells are free of typos."""

import json
import subprocess
from pathlib import Path
from typing import Any

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "colab_run.ipynb"


def notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text())


def code_cells() -> list[str]:
    cells = notebook()["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_it_is_a_notebook_that_asks_for_a_gpu():
    assert notebook()["nbformat"] == 4
    assert notebook()["metadata"]["accelerator"] == "GPU"


def test_its_python_cells_are_valid_python():
    for source in code_cells():
        if not source.startswith("%%bash"):
            compile(source, str(NOTEBOOK), "exec")  # a syntax error raises here


def test_its_shell_cells_are_valid_bash():
    for source in code_cells():
        if source.startswith("%%bash"):
            body = source.removeprefix("%%bash\n")
            subprocess.run(["bash", "-n"], input=body, text=True, check=True)  # -n: read, don't run


def test_the_live_registry_stays_off_google_drive():
    runs = [source for source in code_cells() if "scalelab run" in source]
    assert runs  # at least one cell runs scalelab
    for run in runs:
        assert "runtime.local_registry=/content/registry.db" in run
