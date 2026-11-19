"""The release files agree with each other and with the repo (m4s21)."""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
VERSION = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)  # a release heading
DATED = re.compile(r"^## \[\d+\.\d+\.\d+\] - (.+)$", re.MULTILINE)  # its date (m4s22)


def citation() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def test_the_citation_has_what_github_reads() -> None:
    cff = citation()
    assert cff["cff-version"] == "1.2.0"
    assert {"message", "title", "authors", "version"} <= cff.keys()
    assert all("family-names" in author for author in cff["authors"])


def test_the_citation_names_the_newest_changelog_version() -> None:
    versions = VERSION.findall((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert versions != []
    assert citation()["version"] == versions[0]


def test_the_citation_names_the_repos_license() -> None:
    assert citation()["license"] == "Apache-2.0"
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")[:200]


def test_the_package_version_is_the_citations() -> None:  # m4s22
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == citation()["version"]


def test_the_citation_date_is_the_newest_changelog_date() -> None:  # m4s22
    dates = DATED.findall((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert dates != []
    assert str(citation()["date-released"]) == dates[0]
