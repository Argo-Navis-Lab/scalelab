"""The write-ups, the walkthrough and the README link only to files that exist (m3s27)."""

import re
from pathlib import Path

from scalelab.core.config import load_config
from scalelab.core.hashing import config_hash

ROOT = Path(__file__).resolve().parents[2]
WRITEUPS = ROOT / "docs" / "writeups"
WALKTHROUGH = ROOT / "docs" / "walkthrough.md"
LINK = re.compile(r"\]\(([^)#\s]+)")  # a Markdown link's target, up to any #anchor


def pages() -> list[Path]:
    return [ROOT / "README.md", WALKTHROUGH, *sorted(WRITEUPS.glob("*.md"))]


def test_write_up_1_is_published() -> None:
    assert (WRITEUPS / "01-cost-model.md").is_file()


def test_write_up_2_is_published() -> None:
    assert (WRITEUPS / "02-scaling.md").is_file()


def test_every_relative_link_resolves() -> None:
    missing = [
        f"{page.relative_to(ROOT)}: {target}"
        for page in pages()
        for target in LINK.findall(page.read_text(encoding="utf-8"))
        if "://" not in target and not (page.parent / target).exists()
    ]
    assert missing == []


def test_the_walkthrough_names_the_warm_start_its_command_makes() -> None:
    # m4s23: the page builds the warm start's run id from this hash (WS=<hash>-s0-<commit>),
    # so a config change that moves the hash must update the page too.
    command = (
        "scalelab run configs/experiments/warmstart.yaml warm_start.tokens=1000000 "
        "'task.difficulties=[2]'"
    )
    overrides = ["warm_start.tokens=1000000", "task.difficulties=[2]"]
    config = load_config(ROOT / "configs" / "experiments" / "warmstart.yaml", overrides)
    text = WALKTHROUGH.read_text(encoding="utf-8")
    assert command in text
    assert f"WS={config_hash(config)}-s0-" in text
