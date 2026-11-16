"""m4s12: one family of curves across sizes, fit to every run's gradings at once."""

import numpy as np
import pytest
from scipy.special import expit, logit

from scalelab.analysis.scaling import N_REF, Family, Point, SizedRun, fit_family
from scalelab.rl.experiment import geometric_steps

STEPS = geometric_steps(4, 1000)  # graded as an M4 run is: 1 2 3 6 10 18 ... 562 1000
LADDER = {  # M4's sizes (non-embedding N) and their warm starts' R_0 (rounded, m4s10)
    800_000: 0.16,
    4_830_000: 0.11,
    14_160_000: 0.13,
    31_630_000: 0.12,
    59_490_000: 0.07,
}
FLOPS_PER_PARAM = 1.6e6  # a step's FLOPs per parameter: m1 1.3e12 a step, as rl.flops buys
PROBLEMS = 653  # the M4 task's: * at bands 4 and 5 (316 + 337)
TRUE = Family(  # A falls with size, C_mid grows 7.4x a decade, B eases off
    a0=float(logit(0.3)), a1=-0.3, c0=float(np.log(3e15)), c1=2.0, b0=float(np.log(1.2)), b1=-0.2
)


def made_up(family: Family, seeds: int, rng: np.random.Generator | None = None) -> list[SizedRun]:
    """`seeds` runs of every LADDER size, their R exactly the family's (rng None) or that
    many problems solved out of PROBLEMS, as a real grading's are."""
    runs: list[SizedRun] = []
    for params, r0 in LADDER.items():
        curve = family.curve(params, r0)
        compute = [FLOPS_PER_PARAM * params * step for step in STEPS]
        for _ in range(seeds):
            rates = curve(compute)
            if rng is not None:
                rates = rng.binomial(PROBLEMS, rates) / PROBLEMS
            points = [Point(0, 0.0, r0, PROBLEMS)] + [
                Point(step, c, float(rate), PROBLEMS)
                for step, c, rate in zip(STEPS, compute, rates, strict=True)
            ]
            runs.append(SizedRun(params=params, points=tuple(points)))
    return runs


def test_each_parameter_is_a_line_in_log_n():
    at_ref = TRUE.curve(int(N_REF), 0.1)
    assert at_ref.r0 == 0.1
    assert at_ref.a == pytest.approx(0.3)
    assert at_ref.c_mid == pytest.approx(3e15)
    assert at_ref.b == pytest.approx(1.2)
    decade_up = TRUE.curve(int(10 * N_REF), 0.1)
    assert logit(decade_up.a) - logit(at_ref.a) == pytest.approx(TRUE.a1)
    assert decade_up.c_mid / at_ref.c_mid == pytest.approx(np.exp(TRUE.c1))
    assert decade_up.b / at_ref.b == pytest.approx(np.exp(TRUE.b1))
    far = Family(a0=0.0, a1=-50.0, c0=30.0, c1=-50.0, b0=0.0, b1=-50.0).curve(10**12, 0.1)
    assert 0 <= far.a < 1 and far.c_mid > 0 and far.b > 0  # at any size, however far out


@pytest.mark.parametrize(
    "true",
    [
        TRUE,
        Family(a0=0.5, a1=0.4, c0=float(np.log(1e15)), c1=1.0, b0=0.0, b1=0.0),  # A grows
        Family(a0=-1.0, a1=0.0, c0=float(np.log(5e15)), c1=2.5, b0=0.5, b1=0.3),  # sharper
    ],
)
def test_a_family_fit_recovers_the_family_its_runs_came_from(true: Family):
    fit = fit_family(made_up(true, seeds=1))
    found = np.array([getattr(fit.family, name) for name in ("a0", "a1", "c0", "c1", "b0", "b1")])
    wanted = np.array([getattr(true, name) for name in ("a0", "a1", "c0", "c1", "b0", "b1")])
    assert found == pytest.approx(wanted, abs=1e-5)
    assert fit.rmse < 1e-9
    assert fit.points == len(LADDER) * len(STEPS)
    assert fit.sizes == len(LADDER)


def test_a_family_fit_finds_the_family_through_binomial_noise():
    """5 seeds a size, each grading PROBLEMS problems solved or not. Over 200 seeds (m4s12)
    the worst were: A off by 0.067 on the ladder and 0.044 at m7 (202.41M, outside it),
    C_mid by 0.27 decades on the ladder and 0.35 at m7, B 0.68x to 1.45x, the rms 0.0128
    to 0.0159."""
    m7 = 202_410_000
    for seed in range(10):
        fit = fit_family(made_up(TRUE, seeds=5, rng=np.random.default_rng(seed)))
        for params, r0 in [*LADDER.items(), (m7, 0.07)]:
            true, found = TRUE.curve(params, r0), fit.family.curve(params, r0)
            assert found.a == pytest.approx(true.a, abs=0.1)
            assert np.log10(found.c_mid) == pytest.approx(np.log10(true.c_mid), abs=0.5)
            assert true.b / 2 < found.b < true.b * 2
        assert 0.01 < fit.rmse < 0.02
        assert fit.points == 5 * len(LADDER) * len(STEPS)


def test_a_family_fit_refuses_runs_it_cannot_fit():
    runs = made_up(TRUE, seeds=2)
    two_sizes = [run for run in runs if run.params < 10_000_000]
    with pytest.raises(ValueError, match="needs runs of 3 sizes or more, got 2"):
        fit_family(two_sizes)
    late = SizedRun(params=runs[0].params, points=runs[0].points[1:])
    with pytest.raises(ValueError, match="must start at step 0"):
        fit_family([*runs, late])


def test_r_never_leaves_0_and_1_whatever_the_family():
    curve = Family(a0=9.0, a1=5.0, c0=30.0, c1=1.0, b0=0.0, b1=0.0).curve(59_490_000, 0.07)
    assert 0.0 <= float(curve(1e30)) <= 1.0
    assert float(expit(9.0 + 5.0 * np.log10(59_490_000 / N_REF))) == pytest.approx(curve.a)
