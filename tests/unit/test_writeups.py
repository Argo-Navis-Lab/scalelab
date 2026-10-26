"""The write-ups and the README link only to files that exist (m3s27)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRITEUPS = ROOT / "docs" / "writeups"
LINK = re.compile(r"\]\(([^)#\s]+)")  # a Markdown link's target, up to any #anchor


def pages() -> list[Path]:
    return [ROOT / "README.md", *sorted(WRITEUPS.glob("*.md"))]


def test_write_up_1_is_published() -> None:
    assert (WRITEUPS / "01-cost-model.md").is_file()


def test_every_relative_link_resolves() -> None:
    missing = [
        f"{page.relative_to(ROOT)}: {target}"
        for page in pages()
        for target in LINK.findall(page.read_text(encoding="utf-8"))
        if "://" not in target and not (page.parent / target).exists()
    ]
    assert missing == []
