# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose


def test_pixel_to_world_values_broadcasts_pixel_inputs(gwcs_3d_spatial_wave):
    pixel_arrays = (np.arange(4.0), 0.0, 0.0)
    broadcast_arrays = np.broadcast_arrays(*pixel_arrays)

    expected = gwcs_3d_spatial_wave.pixel_to_world_values(*broadcast_arrays)
    actual = gwcs_3d_spatial_wave.pixel_to_world_values(*pixel_arrays)

    for actual_axis, expected_axis in zip(actual, expected, strict=True):
        assert_allclose(actual_axis, expected_axis)


def test_pixel_to_world_values_preserves_scalar_outputs(gwcs_3d_spatial_wave):
    result = gwcs_3d_spatial_wave.pixel_to_world_values(0.0, 0.0, 0.0)
    assert all(np.isscalar(axis) for axis in result)
