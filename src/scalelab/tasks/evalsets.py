"""Frozen eval sets: fixed files of evaluation problems, one per task and difficulty band.

Each file is made once by this module (python -m scalelab.tasks.evalsets), committed to git,
and checked against its recorded sha256 every time it's read.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scalelab.core.artifacts import sha256_hex
from scalelab.core.config import StrictModel
from scalelab.core.storage import atomic_write_bytes
from scalelab.tasks.arithmetic import ArithmeticTask
from scalelab.tasks.base import Task
from scalelab.tasks.countdown import CountdownTask

EVAL_SIZE = 1000  # problems per band: enough to pin a pass rate within about 1.6 points
EVAL_DIR = Path(__file__).resolve().parents[1] / "evalsets"  # src/scalelab/evalsets/

# Every eval stream's seed starts with this number, far beyond the largest run seed
# (2**32 - 1), so training draws its problems from other streams than the eval sets.
EVAL_ENTROPY = 0xBC7A063D513B459A1B0C8C9986528BA

ARITHMETIC_BANDS = range(2, 6)  # digits per number; up to 5, all three operations fit 256 tokens
COUNTDOWN_BANDS = range(3, 7)  # how many numbers

# The frozen files: name -> sha256 of the file's bytes. To change a file on purpose, remake it
# and update its hash here. Any other change makes reading it fail.
EVAL_SETS = {
    "arithmetic-2": "4ef7f5698945e4e9410c5141b9df03f6149cf9619f3938ee76d022f7f87a62ed",
    "arithmetic-3": "ec8798d6cad4e166fd5ca061435636d5ffb71d160737e2f73bc2d01609a02a00",
    "arithmetic-4": "69454440c699870cb91c37eeef6d6c63130fe5cb50beaada7938dfd7cdcc2930",
    "arithmetic-5": "32e688a4475333c3c023a781c5b9af741256e2f0dc58befb8e041ecd29fb58b7",
    "countdown-3": "440200231c71a7f8b61ce54a02af9487b28aaeb37359e3e99cd352e9a2404726",
    "countdown-4": "be6cd922f342e8d44e5b1f7b3fe4e275970ef20c9ade7685c6ccdd89e8c6b29e",
    "countdown-5": "08d166e1611bb9982b85027e3c15887e33511447de5e8bde16ae33c74ec0016d",
    "countdown-6": "951cdf1b2b473f426a86670f2c48d0077a402fa4a6d89bf0838a25cc48c8bd61",
}


class CorruptEvalSetError(Exception):
    """An eval set file's bytes don't match its recorded sha256."""


def eval_set_name(task_name: str, difficulty: int) -> str:
    return f"{task_name}-{difficulty}"


def eval_rng(task_name: str, difficulty: int) -> np.random.Generator:
    """The random generator that makes one eval set."""
    task_number = int.from_bytes(task_name.encode(), "big")  # "arithmetic" as a whole number
    return np.random.default_rng([EVAL_ENTROPY, task_number, difficulty])


def make_eval_set[P](task: Task[P], difficulty: int, n: int = EVAL_SIZE) -> list[P]:
    """The first n problems with different prompts that the band's eval stream gives. Small
    bands repeat problems (2-digit arithmetic has only 24,300), and a repeat would count
    twice, so this draws twice as many as it needs and skips the repeats."""
    unique: dict[tuple[int, ...], P] = {}  # prompt -> its first problem, in drawing order
    for problem in task.sample(eval_rng(task.name, difficulty), 2 * n, difficulty):
        unique.setdefault(tuple(task.format_prompt(problem)), problem)
    if len(unique) < n:
        raise ValueError(
            f"{task.name} at difficulty {difficulty} has fewer than {n} different problems"
        )
    return list(unique.values())[:n]


def to_jsonl(problems: Sequence[StrictModel]) -> bytes:
    """One problem per line, as JSON."""
    return "".join(problem.model_dump_json() + "\n" for problem in problems).encode()


def write_all(directory: Path = EVAL_DIR) -> dict[str, str]:
    """Make every eval set file in `directory`, and return each one's name and sha256."""
    hashes: dict[str, str] = {}
    for difficulty in ARITHMETIC_BANDS:
        hashes.update(_write(ArithmeticTask(), difficulty, directory))
    for difficulty in COUNTDOWN_BANDS:
        hashes.update(_write(CountdownTask(), difficulty, directory))
    return hashes


def read_eval_set[P: StrictModel](
    name: str, problem_type: type[P], directory: Path = EVAL_DIR
) -> list[P]:
    """Load a frozen eval set, refusing it unless its bytes match the recorded sha256."""
    if name not in EVAL_SETS:
        raise ValueError(f"no eval set named {name!r}; there are {sorted(EVAL_SETS)}")
    data = (directory / f"{name}.jsonl").read_bytes()
    if sha256_hex(data) != EVAL_SETS[name]:
        raise CorruptEvalSetError(f"{name}.jsonl doesn't match its recorded sha256")
    return [problem_type.model_validate_json(line) for line in data.decode().splitlines()]


def _write[P: StrictModel](task: Task[P], difficulty: int, directory: Path) -> dict[str, str]:
    name = eval_set_name(task.name, difficulty)
    data = to_jsonl(make_eval_set(task, difficulty))
    atomic_write_bytes(directory / f"{name}.jsonl", data)
    return {name: sha256_hex(data)}


if __name__ == "__main__":
    for name, sha256 in write_all().items():
        print(f'    "{name}": "{sha256}",')
