"""A farther point must not move a visible foreground depth behind its surface."""

import numpy as np
import pytest
import torch

from boxernet.boxernet import sdp_to_patches
from loaders.neon_loader import NeonLoader
from test_neon_loader import make_geometry, recording


def test_neon_uses_nearest_surface_at_overlapping_pixels(recording, tmp_path):
    root, times = recording
    geometry = tmp_path / "map.npz"
    # Camera at (2,3,1) looking along +Y. Both points project at the centre.
    make_geometry(geometry, times, np.array([[2., 5., 1.], [2., 11., 1.]], dtype=np.float32))
    loader = NeonLoader(root, max_n=1, resize=32, geometry_path=geometry)
    try:
        datum = loader.load(0)
        assert datum["nearest_surface_depth"] is True
        args = (datum["sdp_w"][None], datum["cam0"][None], datum["T_world_rig0"][None], 32, 32, 16)
        old = sdp_to_patches(*args)
        new = sdp_to_patches(*args, nearest_surface=True)
        assert old.max().item() == pytest.approx(5.)
        assert new.max().item() == pytest.approx(2.)
        assert (new > 0).sum() == 1
        assert datum["observed_points"] is None
    finally:
        loader.close()


def test_nearest_depth_is_batched_and_empty_input_stays_missing():
    from loaders.base_loader import BaseLoader
    from utils.tw.pose import PoseTW

    camera = BaseLoader.pinhole_from_K(32, 32, 20, 20, 16, 16)
    pose = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    points = torch.tensor([[[0., 0., 2.], [0., 0., 8.], [0., 0., -1.]],
                           [[0., 0., 7.], [0., 0., 3.], [float("nan")] * 3]])
    result = sdp_to_patches(points, camera[None].repeat(2, 1), pose[None].repeat(2, 1),
                            32, 32, 16, nearest_surface=True)
    assert result.flatten(1).max(1).values.tolist() == pytest.approx([2., 3.])
    empty = sdp_to_patches(points[:, :0], camera[None].repeat(2, 1), pose[None].repeat(2, 1),
                          32, 32, 16, nearest_surface=True)
    assert torch.equal(empty, torch.full_like(empty, -1.))
