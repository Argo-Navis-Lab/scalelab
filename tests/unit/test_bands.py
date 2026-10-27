"""The M4 RL task's bands (m4s7): kept where the smallest model reaches FLOOR and the largest
stays at or under CEILING."""

from scalelab.analysis.bands import CEILING, FLOOR, pick_bands


def test_the_limits_are_the_m4_plans():
    assert (FLOOR, CEILING) == (0.05, 0.50)


def test_a_band_needs_signal_from_the_smallest_and_room_above_the_largest():
    smallest = {2: 0.95, 3: 0.41, 4: 0.085, 5: 0.02}
    largest = {2: 0.99, 3: 0.75, 4: 0.40, 5: 0.30}
    assert pick_bands(smallest, largest) == [4]  # 2 and 3 too easy for the largest, 5 too hard


def test_the_limits_themselves_count():
    assert pick_bands({4: 16 / 320}, {4: 160 / 320}) == [4]  # exactly 5% and exactly 50%


def test_just_past_either_limit_is_left_out():
    assert pick_bands({4: 15 / 320}, {4: 0.3}) == []
    assert pick_bands({4: 0.3}, {4: 161 / 320}) == []


def test_a_band_only_one_model_was_graded_on_is_left_out():
    assert pick_bands({3: 0.2, 4: 0.2}, {4: 0.2, 5: 0.2}) == [4]


def test_the_bands_come_back_in_increasing_order():
    assert pick_bands({5: 0.2, 3: 0.2, 4: 0.2}, {4: 0.2, 5: 0.2, 3: 0.2}) == [3, 4, 5]
