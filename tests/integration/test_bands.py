"""`scalelab bands` (m4s7): finished warm starts graded on each band's multiplications,
smallest model first, and the bands the M4 RL task keeps by its rule."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from scalelab.cli import EXIT_USAGE, main
from scalelab.core.registry import Registry

HELLO = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "hello.yaml"


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """SCALELAB_HOME for this file's tests: two finished hello runs, m1's and m2's, made
    once and shared (module scope), because training them takes a few seconds."""
    home = tmp_path_factory.mktemp("home").resolve()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SCALELAB_HOME", str(home))
        assert main(["run", str(HELLO)]) == 0
        assert main(["run", str(HELLO), "model=ladder_s/m2"]) == 0
        yield home


def hello_runs(home: Path) -> tuple[str, str]:
    """The m1 run's id and the m2 run's, in the order they were made."""
    m1, m2 = (run.run_id for run in Registry.sqlite(home / "registry.db").list_runs())
    return m1, m2


def test_the_smallest_model_comes_first_and_the_rule_picks_the_bands(
    home: Path, capsys: pytest.CaptureFixture[str]
):
    m1, m2 = hello_runs(home)
    assert main(["bands", m2, m1, "--problems", "20"]) == 0  # the largest given first
    assert capsys.readouterr().out.splitlines() == [
        "graded:        greedy pass@1 on each band's * problems (6, 4, 10, 7 of them)",
        "model    band 2   band 3   band 4   band 5  run",
        f"m1         0.0%     0.0%     0.0%     0.0%  {m1}",
        f"m2         0.0%     0.0%     0.0%     0.0%  {m2}",
        "rule:          m1 at least 5% and m2 at most 50%",
        "bands:         none",  # hello hardly trains: m1 solves nothing, no reward to learn from
    ]


def test_bands_never_changes_the_registry(home: Path):
    before = (home / "registry.db").read_bytes()
    main(["bands", *hello_runs(home), "--problems", "5"])
    assert (home / "registry.db").read_bytes() == before


@pytest.mark.parametrize("twice", [False, True])
def test_one_run_is_refused(home: Path, capsys: pytest.CaptureFixture[str], twice: bool):
    m1, _ = hello_runs(home)
    assert main(["bands", m1, *([m1] if twice else [])]) == EXIT_USAGE
    assert "give two or more different runs" in capsys.readouterr().err


def test_an_unknown_run_is_refused(home: Path, capsys: pytest.CaptureFixture[str]):
    m1, _ = hello_runs(home)
    assert main(["bands", m1, "abc-s0-01234567"]) == EXIT_USAGE
    assert capsys.readouterr().err == "scalelab: error: the registry has no run abc-s0-01234567\n"


@pytest.mark.parametrize(("problems", "error"), [(0, "at least 1, got 0"), (1, "hold no *")])
def test_too_few_problems_are_refused(
    home: Path, capsys: pytest.CaptureFixture[str], problems: int, error: str
):
    assert main(["bands", *hello_runs(home), "--problems", str(problems)]) == EXIT_USAGE
    assert error in capsys.readouterr().err
