import torch
from torch import nn as nn


class CopyCat(nn.Module):
    """
    Dummy, no-movement baseline that always outputs the query points as the predicted points.
    """

    def __init__(self):
        super().__init__()
        self.dummy_learnable_param = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            rgbs,
            depths,
            query_points,
            intrs,
            extrs,
            **kwargs,
    ):
        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape
        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        traj_e = query_points[:, None, :, 1:].repeat(1, num_frames, 1, 1)
        vis_e = query_points.new_ones((batch_size, num_frames, num_points))

        results = {
            "traj_e": traj_e,
            "feat_init": None,
            "vis_e": vis_e,
        }
        return results
