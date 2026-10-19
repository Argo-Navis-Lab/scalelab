"""`scalelab grade`: a finished run's grades, whole and by operation, from its run_id alone."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from scalelab.cli import EXIT_USAGE, main
from scalelab.core.registry import Registry

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"
PARTS = ("", "/*", "/+", "/-")  # an arithmetic eval set's lines: the whole set, then each operation


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """SCALELAB_HOME for this file's tests: a folder holding one finished hello run, made
    once and shared (module scope), because training it takes a few seconds."""
    home = tmp_path_factory.mktemp("home").resolve()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SCALELAB_HOME", str(home))
        assert main(["run", str(HELLO)]) == 0
        yield home


def hello_run(home: Path) -> str:
    (run,) = Registry.sqlite(home / "registry.db").list_runs()
    return run.run_id


def test_grade_prints_every_eval_set_whole_and_by_operation(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["grade", hello_run(home), "--problems", "20"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        f"run:   {hello_run(home)}",
        "model: m1, step 16, trained on arithmetic at difficulties 2, 3, 4, 5",
    ]
    assert lines[2:6] == [  # hello hardly trains: nothing solved yet
        "arithmetic-2     0.0% ± 0.0%  (0 of 20)",
        "arithmetic-2/*   0.0% ± 0.0%  (0 of 6)",
        "arithmetic-2/+   0.0% ± 0.0%  (0 of 8)",
        "arithmetic-2/-   0.0% ± 0.0%  (0 of 6)",
    ]
    names = [line.split()[0] for line in lines[2:]]
    assert names == [f"arithmetic-{d}{part}" for d in (2, 3, 4, 5) for part in PARTS]


def test_grade_never_changes_the_registry(home: Path):
    before = (home / "registry.db").read_bytes()
    main(["grade", hello_run(home), "--problems", "5"])
    assert (home / "registry.db").read_bytes() == before


def test_an_unknown_run_is_refused(home: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grade", "abc-s0-01234567"]) == EXIT_USAGE
    assert capsys.readouterr().err == "scalelab: error: the registry has no run abc-s0-01234567\n"


def test_a_home_without_a_registry_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("SCALELAB_HOME", str(tmp_path))
    assert main(["grade", "abc-s0-01234567"]) == EXIT_USAGE
    assert "no registry at" in capsys.readouterr().err
    assert not (tmp_path / "registry.db").exists()  # and none was created


def test_grading_needs_at_least_one_problem(home: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grade", hello_run(home), "--problems", "0"]) == EXIT_USAGE
    assert "--problems must be at least 1, got 0" in capsys.readouterr().err
