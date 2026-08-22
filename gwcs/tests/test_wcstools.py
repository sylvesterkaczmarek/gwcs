# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose, assert_array_equal

from gwcs.wcstools import grid_from_bounding_box


def test_grid_from_bounding_box_does_not_overshoot():
    for center in (False, True):
        grid = grid_from_bounding_box(((0, 5), (0, 5)), step=2, center=center)
        assert_array_equal(grid[0, 0], [0, 2, 4])
        assert_array_equal(grid[1, :, 0], [0, 2, 4])


def test_grid_from_bounding_box_includes_reachable_upper_bound():
    grid = grid_from_bounding_box((0, 4), step=2, center=False)
    assert_array_equal(grid, [0, 2, 4])


def test_grid_from_bounding_box_includes_fractional_upper_bound():
    grid = grid_from_bounding_box((0, 0.3), step=0.1, center=False)
    assert_allclose(grid, [0, 0.1, 0.2, 0.3])
