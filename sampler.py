import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).view((1,self.n_pts_per_ray,1)).to(ray_bundle.origins.device)
        # TODO (Q1.4): Sample points from z values
        # print("Directions: " + str(ray_bundle.directions.shape))
        sample_points = (ray_bundle.directions.unsqueeze(1) * z_vals)
        # print(sample_points.shape)
        # print("Origins: " + str(ray_bundle.origins.shape))
        assert sample_points.shape == (ray_bundle.directions.shape[-2], self.n_pts_per_ray, 3)
        sample_points += ray_bundle.origins.unsqueeze(1)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}