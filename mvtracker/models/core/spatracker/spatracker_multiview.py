import logging
import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn as nn

from mvtracker.models.core.embeddings import Embedder_Fourier, get_3d_sincos_pos_embed_from_grid, \
    get_1d_sincos_pos_embed_from_grid, get_3d_embedding
from mvtracker.models.core.model_utils import sample_features5d, smart_cat
from mvtracker.models.core.spatracker.blocks import BasicEncoder, EUpdateFormer, CorrBlock
from mvtracker.models.core.spatracker.softsplat import softsplat
from mvtracker.models.core.spatracker.spatracker_monocular import sample_pos_embed
from mvtracker.utils.basic import to_homogeneous, from_homogeneous, time_now


class MultiViewSpaTracker(nn.Module):
    """
    Multi-view Spatial Tracker: A 3D Multi-View Tracker with
    Transformer-based Iterative Flow Updates. This version computes
    local correlation in a global triplane space that is aligned with
    the world coordinate planes. However, this leaves most of the triplane
    space empty since it is difficult to create one plane that covers all the
    relevant areas of interest.
    """

    def __init__(
            self,
            sliding_window_len=8,
            stride=8,
            add_space_attn=True,
            use_3d_pos_embed=True,
            remove_zeromlpflow=True,
            concat_triplane_features=True,
            num_heads=8,
            hidden_size=384,
            space_depth=12,
            time_depth=12,
            fmaps_dim=128,
            triplane_xres=128,
            triplane_yres=128,
            triplane_zres=128,
    ):
        super(MultiViewSpaTracker, self).__init__()

        self.S = sliding_window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = fmaps_dim
        self.flow_embed_dim = 64
        self.b_latent_dim = self.latent_dim // 3
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.use_3d_pos_embed = use_3d_pos_embed
        self.remove_zeromlpflow = remove_zeromlpflow
        self.concat_triplane_features = concat_triplane_features
        self.updateformer_input_dim = (

            # The positional encoding of the 3D flow from t=i to t=0
                + (self.flow_embed_dim + 1) * (3 if self.remove_zeromlpflow else 2)

                # The correlation features (LRR) for the three planes (xy, yz, xz), concatenated
                + 196 * (3 if self.concat_triplane_features else 1)

                # The features of the tracked points, one for each of the three planes
                + self.latent_dim * (3 if self.concat_triplane_features else 1)

                # The visibility mask
                + 1

                # The whether-the-point-is-tracked mask
                + 1

        )
        self.triplane_xres = triplane_xres
        self.triplane_yres = triplane_yres
        self.triplane_zres = triplane_zres

        # Feature encoder
        self.fnet = BasicEncoder(
            input_dim=3,
            output_dim=self.latent_dim,
            norm_fn="instance",
            dropout=0,
            stride=stride,
            Embed3D=False,
        )

        # Convolutional heads for the tri-plane features
        self.headxy = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
        )
        self.headyz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
        )
        self.headxz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
        )

        # Transformer for the iterative flow updates
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1
        self.updateformer = EUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.updateformer_input_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=3 + self.latent_dim * 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            flash=True,
        )

        # Updater of the features of the tracked points
        self.norm_xy = nn.GroupNorm(1, self.latent_dim)
        self.norm_yz = nn.GroupNorm(1, self.latent_dim)
        self.norm_xz = nn.GroupNorm(1, self.latent_dim)
        self.ffeatxy_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatyz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatxz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )

        # Embedders
        self.embed_traj = Embedder_Fourier(input_dim=5, max_freq_log2=5.0, N_freqs=3, include_input=True)
        self.embed3d = Embedder_Fourier(input_dim=3, max_freq_log2=10.0, N_freqs=10, include_input=True)
        self.embedConv = nn.Conv2d(self.latent_dim + 63, self.latent_dim, 3, padding=1)

        # Predictor of the visibility of the tracked points
        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim * (3 if self.concat_triplane_features else 1), 1))
        self.zeroMLPflow = nn.Linear(195, 130)

    def sample_trifeat(self, t, coords, featMapxy, featMapyz, featMapxz):
        """
        Sample the features from the 5D triplane feature map 3*(B S C H W)
        Args:
            t: the time index
            coords: the coordinates of the points B S N 3
            featMapxy: the feature map B S C Hx Wy
            featMapyz: the feature map B S C Hy Wz
            featMapxz: the feature map B S C Hx Wz
        """
        # get xy_t yz_t xz_t
        queried_t = t.reshape(1, 1, -1, 1)
        xy_t = torch.cat(
            [queried_t, coords[..., [0, 1]]],
            dim=-1
        )
        yz_t = torch.cat(
            [queried_t, coords[..., [1, 2]]],
            dim=-1
        )
        xz_t = torch.cat(
            [queried_t, coords[..., [0, 2]]],
            dim=-1
        )
        featxy_init = sample_features5d(featMapxy, xy_t)

        featyz_init = sample_features5d(featMapyz, yz_t)
        featxz_init = sample_features5d(featMapxz, xz_t)

        featxy_init = featxy_init.repeat(1, self.S, 1, 1)
        featyz_init = featyz_init.repeat(1, self.S, 1, 1)
        featxz_init = featxz_init.repeat(1, self.S, 1, 1)

        return featxy_init, featyz_init, featxz_init

    def forward_iteration(
            self,
            fmapXY,
            fmapYZ,
            fmapXZ,
            coords_init,
            vis_init,
            track_mask,
            iters=4,
            feat_init=None,
    ):
        N = coords_init.shape[2]
        B, S, fmap_dim, triplane_H, triplane_W = fmapXY.shape
        triplane_D = fmapXZ.shape[-2]
        device = fmapXY.device

        if coords_init.shape[1] < S:
            coords = torch.cat([coords_init, coords_init[:, -1].repeat(1, S - coords_init.shape[1], 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - coords_init.shape[1], 1, 1)], dim=1)
        else:
            coords = coords_init.clone()

        assert B == 1
        assert fmapXY.shape == (B, S, fmap_dim, triplane_H, triplane_W)
        assert fmapYZ.shape == (B, S, fmap_dim, triplane_D, triplane_H)
        assert fmapXZ.shape == (B, S, fmap_dim, triplane_D, triplane_W)
        assert coords.shape == (B, S, N, 3)
        assert vis_init.shape == (B, S, N, 1)
        assert track_mask.shape == (B, S, N, 1)
        assert feat_init is None or feat_init.shape == (B, S, N, self.latent_dim, 3)

        fcorr_fnXY = CorrBlock(fmapXY, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fnYZ = CorrBlock(fmapYZ, num_levels=self.corr_levels, radius=self.corr_radius)
        fcorr_fnXZ = CorrBlock(fmapXZ, num_levels=self.corr_levels, radius=self.corr_radius)

        ffeats = torch.split(feat_init.clone(), dim=-1, split_size_or_sections=1)
        ffeats = [f.squeeze(-1) for f in ffeats]

        grid_size = coords.new_tensor([triplane_H, triplane_W, triplane_D])
        # @Single-view-difference:
        #     Instead of computing 2D positional embeddings in the XY plane of the single-view triplane
        #     (which is aligned with the monocular view used in the single-view SpatialTracker), I will
        #     compute 3D positional embeddings in the 3D grid of the triplane. This could allow the model
        #     to more easily learn the 3D spatial relationships between the points in the triplane.
        # pos_embed = sample_pos_embed(
        #     grid_size=(H8, W8),
        #     embed_dim=456,
        #     coords=coords[..., :2],
        # )
        embed_dim = self.updateformer_input_dim
        if self.use_3d_pos_embed:
            # Ours
            if embed_dim % 3 != 0:
                # Make sure that the embed_dim is divisible by 3
                embed_dim += 3 - (embed_dim % 3)
            pos_embed = get_3d_sincos_pos_embed_from_grid(
                embed_dim=embed_dim,
                # Normalize the coordinates so that the grid ranges over [-128,128]
                grid=((coords[:, :1, ...] / grid_size) * 2 - 1) * 128,
            ).float()[:, 0, ...].permute(0, 2, 1)
        else:
            # Original
            if embed_dim % 4 != 0:
                # Make sure that the embed_dim is divisible by 4
                embed_dim += 4 - (embed_dim % 4)
            pos_embed = sample_pos_embed(
                grid_size=(triplane_H, triplane_W),
                embed_dim=embed_dim,
                coords=coords[..., :2],
            )
        if embed_dim > self.updateformer_input_dim:
            # If the embed_dim was increased for divisibility, then remove the extra dimensions
            pos_embed = pos_embed[:, :self.updateformer_input_dim, :]
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        embed_dim = self.updateformer_input_dim
        if embed_dim % 2 != 0:
            # Make sure that the embed_dim is divisible by 2
            embed_dim += 2 - (embed_dim % 2)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        if embed_dim > self.updateformer_input_dim:
            # If the embed_dim was increased to be divisible by 2, then remove the extra dimensions
            times_embed = times_embed[:, :, :self.updateformer_input_dim]

        coord_predictions = []
        support_feat = self.support_features

        for _ in range(iters):
            coords = coords.detach()
            fcorrsXY = fcorr_fnXY.corr_sample(ffeats[0], coords[..., :2])
            fcorrsYZ = fcorr_fnYZ.corr_sample(ffeats[1], coords[..., [1, 2]])
            fcorrsXZ = fcorr_fnXZ.corr_sample(ffeats[2], coords[..., [0, 2]])
            # @Single-view-difference:
            #     Instead of summing the correlations for different planes, I will concatenate them so that the model
            #     can learn to differentiate between the correlations of different planes. Summing the correlations up
            #     can make it very difficult for the model to differentiate between the correlations of different
            #     planes unless, e.g., it learns to create the feature maps in a way that they are orthogonal
            #     to each other. But rather than relying on the model to learn this, I believe that it is better
            #     to provide the model with the information that the correlations are from different planes explicitly.
            #     Note that this change will increase the dimension of the correlation features that are given to the
            #     transformer: 196 * 3 = 588, instead of 196.
            # fcorrs = fcorrsXY + fcorrsYZ + fcorrsXZ
            if self.concat_triplane_features:
                # Ours
                fcorrs = torch.cat([fcorrsXY, fcorrsYZ, fcorrsXZ], dim=-1)
            else:
                # Original
                fcorrs = fcorrsXY + fcorrsYZ + fcorrsXZ
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, self.flow_embed_dim, cat_coords=True)
            # @Single-view-difference:
            #     I have removed the zeroMLPflow linear layer which was added to project the flow embedding
            #     from a 195-dimensional vector to a 130-dimensional to have a cleaner architecture.
            #     I believe that the authors have added this layer just to match the 130 that the original
            #     CoTracker implementation had used, but this can introduce confusion in the architecture's design.
            # flows_cat = self.zeroMLPflow(flows_cat)
            if self.remove_zeromlpflow:
                # Ours
                pass
            else:
                # Original
                flows_cat = self.zeroMLPflow(flows_cat)

            ffeats_xy = ffeats[0].permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_yz = ffeats[1].permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_xz = ffeats[2].permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            # @Single-view-difference:
            #     Instead of summing the features for different planes, I will concatenate them so that the model
            #     can learn to differentiate between the features of different planes. Summing the features up
            #     can make it very difficult for the model to differentiate between the features of different
            #     planes. I believe that it is better to provide the model with the information that the features
            #     are from different planes explicitly. Note that this change will increase the dimension of the
            #     feature embeddings that are given to the transformer: 128 * 3 = 384, instead of 128.
            # ffeats_ = ffeats_xy + ffeats_yz + ffeats_xz
            if self.concat_triplane_features:
                # Ours
                ffeats_ = torch.cat([ffeats_xy, ffeats_yz, ffeats_xz], dim=-1)
            else:
                # Original
                ffeats_ = ffeats_xy + ffeats_yz + ffeats_xz

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat([
                    track_mask,
                    torch.zeros_like(track_mask[:, 0]).repeat(1, vis_init.shape[1] - track_mask.shape[1], 1, 1),
                ], dim=1)
            track_mask_and_vis = torch.cat([track_mask, vis_init], dim=2).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, track_mask_and_vis], dim=2)
            assert transformer_input.shape[-1] == pos_embed.shape[-1]

            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta, delta_se3F = self.updateformer(x, support_feat)
            support_feat = support_feat + delta_se3F[0] / 100
            delta = rearrange(delta, " b n t d -> (b n) t d")
            d_coord = delta[:, :, :3]
            d_feats_xy = delta[:, :, 3:self.latent_dim + 3]
            d_feats_yz = delta[:, :, self.latent_dim + 3:self.latent_dim * 2 + 3]
            d_feats_xz = delta[:, :, self.latent_dim * 2 + 3:]
            d_feats_xy_norm = self.norm_xy(d_feats_xy.view(-1, self.latent_dim))
            d_feats_yz_norm = self.norm_yz(d_feats_yz.view(-1, self.latent_dim))
            d_feats_xz_norm = self.norm_xz(d_feats_xz.view(-1, self.latent_dim))
            ffeats_xy = ffeats_xy.reshape(-1, self.latent_dim) + self.ffeatxy_updater(d_feats_xy_norm)
            ffeats_yz = ffeats_yz.reshape(-1, self.latent_dim) + self.ffeatyz_updater(d_feats_yz_norm)
            ffeats_xz = ffeats_xz.reshape(-1, self.latent_dim) + self.ffeatxz_updater(d_feats_xz_norm)
            ffeats[0] = ffeats_xy.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
            ffeats[1] = ffeats_yz.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
            ffeats[2] = ffeats_xz.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
            coords = coords + d_coord.reshape(B, N, S, 3).permute(0, 2, 1, 3)
            if torch.isnan(coords).any():
                logging.error("Got NaN values in coords, perhaps the training exploded")
                import ipdb;
                ipdb.set_trace()

            coord_predictions.append(coords.clone())

        # @Single-view-difference:
        #     Instead of summing the features for different planes,
        #     I will concatenate before inputting them to the shallow visibility predictor.
        # ffeats_f = ffeats[0] + ffeats[1] + ffeats[2]
        if self.concat_triplane_features:
            ffeats_f = torch.cat(ffeats, dim=-1)
            vis_e = self.vis_predictor(ffeats_f.reshape(B * S * N, self.latent_dim * 3)).reshape(B, S, N)
        else:
            ffeats_f = ffeats[0] + ffeats[1] + ffeats[2]
            vis_e = self.vis_predictor(ffeats_f.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)

        self.support_features = support_feat.detach()

        return coord_predictions, vis_e, feat_init

    def forward(
            self,
            rgbs,
            depths,
            query_points,
            intrs,
            extrs,
            iters=4,
            feat_init=None,
            is_train=False,
            save_debug_logs=False,
            debug_logs_path="",
            **kwargs,
    ):
        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        if feat_init is not None:
            raise NotImplementedError("feat_init is not supported yet")

        if save_debug_logs:
            os.makedirs(debug_logs_path, exist_ok=True)
            if kwargs:
                warnings.warn(f"Received unexpected kwargs: {kwargs.keys()}")

        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1
        self.is_train = is_train

        # Unpack the query points
        query_points_t = query_points[:, :, :1].long()
        query_points_xyz_worldspace = query_points[:, :, 1:]

        # Invert intrinsics and extrinsics
        intrs_inv = torch.inverse(intrs.float())
        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(batch_size, num_views, num_frames, 1, 1)
        extrs_square[:, :, :, :3, :] = extrs
        extrs_inv = torch.inverse(extrs_square.float())

        # Interpolate the rgbs and depthmaps to the stride of the SpaTracker
        strided_height = height // self.stride
        strided_width = width // self.stride

        strided_depths = nn.functional.interpolate(
            input=depths.reshape(-1, 1, height, width),
            scale_factor=1.0 / self.stride,
            mode="nearest",
        ).reshape(batch_size, num_views, num_frames, 1, strided_height, strided_width)

        strided_rgbs = nn.functional.interpolate(
            input=rgbs.reshape(-1, 3, height, width),
            scale_factor=1.0 / self.stride,
            mode="bilinear",
        ).reshape(batch_size, num_views, num_frames, 3, strided_height, strided_width)

        # Un-project strided depthmaps back to world coordinates
        pixel_xy = torch.stack(torch.meshgrid(
            (torch.arange(0, height / self.stride) + 0.5) * self.stride - 0.5,
            (torch.arange(0, width / self.stride) + 0.5) * self.stride - 0.5,
            indexing="ij",
        )[::-1], dim=-1)
        pixel_xy = pixel_xy.to(device=rgbs.device, dtype=rgbs.dtype)
        pixel_xy_homo = to_homogeneous(pixel_xy)
        depthmap_camera_xyz = torch.einsum('BVTij,HWj->BVTHWi', intrs_inv, pixel_xy_homo)
        depthmap_camera_xyz = depthmap_camera_xyz * strided_depths[..., 0, :, :, None]
        depthmap_camera_xyz_homo = to_homogeneous(depthmap_camera_xyz)
        depthmap_world_xyz_homo = torch.einsum('BVTij,BVTHWj->BVTHWi', extrs_inv, depthmap_camera_xyz_homo)
        depthmap_world_xyz = from_homogeneous(depthmap_world_xyz_homo)

        if save_debug_logs:
            t = 0
            n_skip = 4
            xyz = depthmap_world_xyz[0, :, t, ::n_skip, ::n_skip, :].reshape(-1, 3).cpu().numpy()
            c = strided_rgbs.permute(0, 1, 2, 4, 5, 3)[0, :, t, ::n_skip, ::n_skip].reshape(-1, 3).cpu().numpy() / 255
            filename = time_now() + "__rgbd_with_queries"
            qp = query_points_xyz_worldspace[0].cpu().numpy()
            qc = np.array([[1, 0, 0]] * query_points_xyz_worldspace.shape[1])
            self._plot_pointcloud(debug_logs_path, filename, xyz, c, qp, qc, show=False)

        # Put the three planes along the YX, ZX, and ZY axes
        # TODO: Hardcode the xyz ranges for the triplanes,
        #       as taking the whole range would make the
        #       central object of interest very tiny and
        #       the grid would be wasted in representing
        #       wast background.
        x_range = [-14, 14]
        y_range = [-14, 14]
        z_range = [-1, 10]
        query_points_outside_of_triplane_range = (
                (query_points_xyz_worldspace[..., 0].flatten() < x_range[0]) |
                (query_points_xyz_worldspace[..., 0].flatten() > x_range[1]) |
                (query_points_xyz_worldspace[..., 1].flatten() < y_range[0]) |
                (query_points_xyz_worldspace[..., 1].flatten() > y_range[1]) |
                (query_points_xyz_worldspace[..., 2].flatten() < z_range[0]) |
                (query_points_xyz_worldspace[..., 2].flatten() > z_range[1])
        )
        if query_points_outside_of_triplane_range.any():
            warnings.warn(f"Some Query points are outside of the triplane range. "
                          f"x_range={x_range}, y_range={y_range}, z_range={z_range}. "
                          f"query_points_xyz_worldspace={query_points_xyz_worldspace[:, query_points_outside_of_triplane_range]}")

        kwargs = {"device": depthmap_world_xyz.device, "dtype": depthmap_world_xyz.dtype}
        triplane_xyz_min = torch.tensor([x_range[0], y_range[0], z_range[0]], **kwargs)
        triplane_xyz_max = torch.tensor([x_range[1], y_range[1], z_range[1]], **kwargs)
        triplane_grid_dims = torch.tensor([self.triplane_xres, self.triplane_yres, self.triplane_zres], **kwargs)

        if save_debug_logs:
            t = 0
            n_skip = 1
            xyz = depthmap_world_xyz[0, :, t, ::n_skip, ::n_skip, :].reshape(-1, 3).cpu().numpy()
            c = strided_rgbs.permute(0, 1, 2, 4, 5, 3)[0, :, t, ::n_skip, ::n_skip, :].reshape(-1,
                                                                                               3).cpu().numpy() / 255
            mask = (
                    (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1]) &
                    (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1]) &
                    (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
            )
            xyz_in_range = xyz[mask]
            c_in_range = c[mask]

            qp = query_points_xyz_worldspace[0].cpu().numpy()
            qc = np.array([[1, 0, 0]] * query_points_xyz_worldspace.shape[1])
            mask = (
                    (qp[:, 0] >= x_range[0]) & (qp[:, 0] <= x_range[1]) &
                    (qp[:, 1] >= y_range[0]) & (qp[:, 1] <= y_range[1]) &
                    (qp[:, 2] >= z_range[0]) & (qp[:, 2] <= z_range[1])
            )
            qp_in_range = qp[mask]
            qc_in_range = qc[mask]

            filename = time_now() + "__rgbd_with_queries_within_triplane_range"
            self._plot_pointcloud(debug_logs_path, filename, xyz_in_range, c_in_range, qp_in_range, qc_in_range,
                                  show=False)

        # Pre-compute the per-view feature maps
        rgbs_normalized = 2 * (rgbs / 255.0) - 1.0
        fnet_fmaps = self.fnet(rgbs_normalized.reshape(-1, 3, height, width))
        fnet_fmaps = fnet_fmaps.reshape(
            batch_size, num_views, num_frames, self.latent_dim, strided_height, strided_width,
        )

        # Add Positional 3D Embeddings/Encodings
        def world_to_triplane(points, inverse=False):
            assert points.shape[-1] == 3
            if inverse:
                return points * (triplane_xyz_max - triplane_xyz_min) / (triplane_grid_dims - 1) + triplane_xyz_min
            else:
                return (points - triplane_xyz_min) / (triplane_xyz_max - triplane_xyz_min) * (triplane_grid_dims - 1)

        depthmap_world_xyz_normalized = (depthmap_world_xyz - triplane_xyz_min) / (triplane_xyz_max - triplane_xyz_min)
        positional_encoding_3d = self.embed3d(2 * depthmap_world_xyz_normalized.reshape(-1, 3) - 1)
        positional_encoding_3d = (
            positional_encoding_3d
            .reshape(batch_size, num_views, num_frames, strided_height, strided_width, -1)
            .permute(0, 1, 2, 5, 3, 4)  # HWC --> CHW
        )
        fmaps = torch.cat([fnet_fmaps, positional_encoding_3d], dim=-3)
        fmaps = fmaps.reshape(-1, self.latent_dim + self.embed3d.out_dim, strided_height, strided_width)
        fmaps = self.embedConv(fmaps)
        fmaps = fmaps.reshape(batch_size, num_views, num_frames, self.latent_dim, strided_height, strided_width)

        # Compute the flows from each depthmap to the triplane
        # The flows are needed to splat the features from the depthmap to the triplane
        # The flow defines how one 2D plane is transformed to another 2D plane
        # In our case, the first plane will be of ... TODO describe the planes more
        depthmap_world_xyz_normalized_to_triplane_grid = depthmap_world_xyz_normalized * (triplane_grid_dims - 1)
        depthmap_world_xyz_reproduced = world_to_triplane(
            points=depthmap_world_xyz_normalized_to_triplane_grid,
            inverse=True,
        )
        if not depthmap_world_xyz_reproduced.allclose(depthmap_world_xyz, atol=0.72):
            logging.info("depthmap_world_xyz_reproduced", depthmap_world_xyz_reproduced)
            logging.info("depthmap_world_xyz", depthmap_world_xyz)
            warnings.warn(f"Applying the inverse of world_to_triplane did not reproduce depthmap_world_xyz... "
                          f"The maximum difference is {torch.max(torch.abs(depthmap_world_xyz_reproduced - depthmap_world_xyz))}")

        flow_pointcloud_to_xy = depthmap_world_xyz_normalized_to_triplane_grid[..., [0, 1]]
        flow_pointcloud_to_yz = depthmap_world_xyz_normalized_to_triplane_grid[..., [1, 2]]
        flow_pointcloud_to_xz = depthmap_world_xyz_normalized_to_triplane_grid[..., [0, 2]]
        flow_pointcloud_to_xy = (
            flow_pointcloud_to_xy
            .permute(0, 2, 5, 3, 1, 4)
            .reshape(batch_size * num_frames, 2, strided_height, num_views * strided_width)
        )
        flow_pointcloud_to_yz = (
            flow_pointcloud_to_yz
            .permute(0, 2, 5, 3, 1, 4)
            .reshape(batch_size * num_frames, 2, strided_height, num_views * strided_width)
        )
        flow_pointcloud_to_xz = (
            flow_pointcloud_to_xz
            .permute(0, 2, 5, 3, 1, 4)
            .reshape(batch_size * num_frames, 2, strided_height, num_views * strided_width)
        )

        # Compute the triplane features by splatting the per-view features following the flows
        def splat_fmaps(fmaps, flow_xy, flow_yz, flow_xz, out_shape):
            dtype = fmaps.dtype
            fmaps = fmaps.float()
            flow_xy = flow_xy.float()
            flow_yz = flow_yz.float()
            flow_xz = flow_xz.float()
            fmap_xy, fmap_xy_norm = softsplat(
                tenIn=fmaps,
                tenFlow=flow_xy,
                tenMetric=None,
                strMode="avg",
                tenoutH=out_shape[1],
                tenoutW=out_shape[0],
                use_pointcloud_splatting=True,
                return_normalization_tensor=True,
            )
            fmap_yz, fmap_yz_norm = softsplat(
                tenIn=fmaps,
                tenFlow=flow_yz,
                tenMetric=None,
                strMode="avg",
                tenoutH=out_shape[2],
                tenoutW=out_shape[1],
                use_pointcloud_splatting=True,
                return_normalization_tensor=True,
            )
            fmap_xz, fmap_xz_norm = softsplat(
                tenIn=fmaps,
                tenFlow=flow_xz,
                tenMetric=None,
                strMode="avg",
                tenoutH=out_shape[2],
                tenoutW=out_shape[0],
                use_pointcloud_splatting=True,
                return_normalization_tensor=True,
            )
            if dtype != fmaps.dtype:
                fmap_xy = fmap_xy.to(dtype)
                fmap_yz = fmap_yz.to(dtype)
                fmap_xz = fmap_xz.to(dtype)
                fmap_xy_norm = fmap_xy_norm.to(dtype)
                fmap_yz_norm = fmap_yz_norm.to(dtype)
                fmap_xz_norm = fmap_xz_norm.to(dtype)
            return fmap_xy, fmap_yz, fmap_xz, fmap_xy_norm, fmap_yz_norm, fmap_xz_norm

        fmaps = (
            fmaps
            .permute(0, 2, 3, 4, 1, 5)
            .reshape(batch_size * num_frames, self.latent_dim, strided_height, num_views * strided_width)
        )
        fmap_xy, fmap_yz, fmap_xz, fmap_xy_norm, fmap_yz_norm, fmap_xz_norm = splat_fmaps(
            fmaps=fmaps,
            flow_xy=flow_pointcloud_to_xy,
            flow_yz=flow_pointcloud_to_yz,
            flow_xz=flow_pointcloud_to_xz,
            out_shape=(self.triplane_xres, self.triplane_yres, self.triplane_zres),
        )

        if save_debug_logs and (self.triplane_xres == self.triplane_yres == self.triplane_zres):
            # Visualize how the splatting would look like if the strided_rgbs would be directly splatted instead of feature maps
            rgbs_fmaps = (
                strided_rgbs
                .permute(0, 2, 3, 4, 1, 5)
                .reshape(batch_size * num_frames, 3, strided_height, num_views * strided_width)
            )
            rgbs_fmap_xy, rgbs_fmap_yz, rgbs_fmap_xz, rgbs_fmap_xy_norm, rgbs_fmap_yz_norm, rgbs_fmap_xz_norm = splat_fmaps(
                fmaps=rgbs_fmaps,
                flow_xy=flow_pointcloud_to_xy,
                flow_yz=flow_pointcloud_to_yz,
                flow_xz=flow_pointcloud_to_xz,
                out_shape=(self.triplane_xres, self.triplane_yres, self.triplane_zres),
            )
            rgbs_fmap_xy_yz_xz_concat = torch.concat([rgbs_fmap_xy, rgbs_fmap_yz, rgbs_fmap_xz], -1)
            rgbs_fmap_norm_xy_yz_xz_concat = torch.concat([rgbs_fmap_xy_norm, rgbs_fmap_yz_norm, rgbs_fmap_xz_norm], -1)
            self._plot_featuremaps(
                logs_path=debug_logs_path,
                filename=time_now() + "__splatted_rgbs",
                fmaps_before_splatting=rgbs_fmaps,
                splatted_fmaps=rgbs_fmap_xy_yz_xz_concat,
                splat_normalization=rgbs_fmap_norm_xy_yz_xz_concat,
                chosen_channels=(0, 1, 2),
            )

        if save_debug_logs and (self.triplane_xres == self.triplane_yres == self.triplane_zres):
            # Also splat only the first view RGBs to see how the splatting would look like
            rgbs_fmaps = strided_rgbs[0, 0]
            rgbs_fmap_xy, rgbs_fmap_yz, rgbs_fmap_xz, rgbs_fmap_xy_norm, rgbs_fmap_yz_norm, rgbs_fmap_xz_norm = splat_fmaps(
                fmaps=rgbs_fmaps,
                flow_xy=flow_pointcloud_to_xy[:, :, :, :strided_width],
                flow_yz=flow_pointcloud_to_yz[:, :, :, :strided_width],
                flow_xz=flow_pointcloud_to_xz[:, :, :, :strided_width],
                out_shape=(self.triplane_xres, self.triplane_yres, self.triplane_zres),
            )
            rgbs_fmap_xy_yz_xz_concat = torch.concat([rgbs_fmap_xy, rgbs_fmap_yz, rgbs_fmap_xz], -1)
            rgbs_fmap_norm_xy_yz_xz_concat = torch.concat([rgbs_fmap_xy_norm, rgbs_fmap_yz_norm, rgbs_fmap_xz_norm],
                                                          -1)
            self._plot_featuremaps(
                logs_path=debug_logs_path,
                filename=time_now() + "__splatted_rgbs_first_view_only",
                fmaps_before_splatting=rgbs_fmaps,
                splatted_fmaps=rgbs_fmap_xy_yz_xz_concat,
                splat_normalization=rgbs_fmap_norm_xy_yz_xz_concat,
                chosen_channels=(0, 1, 2),
            )
            xyz = to_homogeneous(
                flow_pointcloud_to_xy[0, :, :, :strided_width].permute(1, 2, 0).reshape(-1, 2)).cpu().numpy()
            c = strided_rgbs[0, 0, 0, :, :].permute(1, 2, 0).reshape(-1, 3).cpu().numpy() / 255
            self._plot_pointcloud(debug_logs_path, time_now() + "__flow_xy_debug", xyz, c, show=False)

        if save_debug_logs and (self.triplane_xres == self.triplane_yres == self.triplane_zres):
            if not (self.triplane_xres == self.triplane_yres == self.triplane_zres):
                raise NotImplementedError("Current implementation assumed these, otherwise needs some padding/interp.")
            fmap_xy_yz_xz_concat = torch.concat([fmap_xy, fmap_yz, fmap_xz], dim=-1)
            fmap_norm_xy_yz_xz_concat = torch.concat([fmap_xy_norm, fmap_yz_norm, fmap_xz_norm], dim=-1)
            self._plot_featuremaps(
                logs_path=debug_logs_path,
                filename=time_now() + "__fmaps",
                fmaps_before_splatting=fmaps,
                splatted_fmaps=fmap_xy_yz_xz_concat,
                splat_normalization=fmap_norm_xy_yz_xz_concat,
                chosen_channels=(0, 1, 2),
            )

        fmap_xy = self.headxy(fmap_xy)
        fmap_yz = self.headyz(fmap_yz)
        fmap_xz = self.headxz(fmap_xz)

        fmap_xy = fmap_xy.reshape(batch_size, num_frames, self.latent_dim, self.triplane_yres, self.triplane_xres)
        fmap_yz = fmap_yz.reshape(batch_size, num_frames, self.latent_dim, self.triplane_zres, self.triplane_yres)
        fmap_xz = fmap_xz.reshape(batch_size, num_frames, self.latent_dim, self.triplane_zres, self.triplane_xres)

        if save_debug_logs and (self.triplane_xres == self.triplane_yres == self.triplane_zres):
            if not (self.triplane_xres == self.triplane_yres == self.triplane_zres):
                raise NotImplementedError("Current implementation assumed these, otherwise needs some padding/interp.")
            fmap_xy_yz_xz_concat = torch.concat([fmap_xy[0], fmap_yz[0], fmap_xz[0]], dim=-1)
            fmap_norm_xy_yz_xz_concat = torch.concat([fmap_xy_norm, fmap_yz_norm, fmap_xz_norm], dim=-1)
            self._plot_featuremaps(
                logs_path=debug_logs_path,
                filename=time_now() + "__fmaps_after_head",
                fmaps_before_splatting=fmaps,
                splatted_fmaps=fmap_xy_yz_xz_concat,
                splat_normalization=fmap_norm_xy_yz_xz_concat,
                chosen_channels=(-3, -2, -1),
            )

        # Filter the points that never appear during 1 - T
        assert batch_size == 1, "Batch size > 1 is not supported yet"
        query_points_t = query_points_t.squeeze(0).squeeze(-1)  # BN1 --> N
        ind_array = torch.arange(num_frames, device=query_points.device)
        ind_array = ind_array[None, :, None].repeat(batch_size, 1, num_points)
        track_mask = (ind_array >= query_points_t[None, None, :]).unsqueeze(-1)

        # Prepare the initial coordinates and visibility
        coords_init = query_points_xyz_worldspace.unsqueeze(1).repeat(1, self.S, 1, 1)
        coords_init = world_to_triplane(coords_init)
        vis_init = query_points.new_ones((batch_size, self.S, num_points, 1)) * 10

        # Sort the queries via their first appeared time
        _, sort_inds = torch.sort(query_points_t, dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        assert torch.allclose(query_points_t, query_points_t[sort_inds][inv_sort_inds])

        query_points_t_ = query_points_t[sort_inds]
        coords_init_ = coords_init[..., sort_inds, :].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()
        track_mask_ = track_mask[:, :, sort_inds].clone()

        # Placeholders for the results (for the sorted points)
        traj_e_ = query_points.new_zeros((batch_size, num_frames, num_points, 3))
        vis_e_ = query_points.new_zeros((batch_size, num_frames, num_points))

        # Perform the iterative forward pass of the SpaTracker as usual,
        # but make sure to use the pre-computed triplane features
        w_idx_start = 0
        p_idx_start = 0
        vis_predictions = []
        coord_predictions = []
        p_idx_end_list = []
        while w_idx_start < num_frames - self.S // 2:
            curr_wind_points = torch.nonzero(query_points_t_ < w_idx_start + self.S)
            if curr_wind_points.shape[0] == 0:
                w_idx_start = w_idx_start + self.S // 2
                continue
            p_idx_end = curr_wind_points[-1] + 1
            p_idx_end_list.append(p_idx_end)

            # TODO: Is cloning necessary here – I don't think so?
            fmap_xy_seq = fmap_xy[:, w_idx_start:w_idx_start + self.S].clone()
            fmap_yz_seq = fmap_yz[:, w_idx_start:w_idx_start + self.S].clone()
            fmap_xz_seq = fmap_xz[:, w_idx_start:w_idx_start + self.S].clone()

            # the number of frames may not be divisible by self.S
            S_local = fmap_xy_seq.shape[1]
            if S_local < self.S:
                fmap_xy_seq = torch.cat([fmap_xy_seq, fmap_xy_seq[:, -1, None].repeat(1, self.S - S_local, 1, 1, 1)], 1)
                fmap_yz_seq = torch.cat([fmap_yz_seq, fmap_yz_seq[:, -1, None].repeat(1, self.S - S_local, 1, 1, 1)], 1)
                fmap_xz_seq = torch.cat([fmap_xz_seq, fmap_xz_seq[:, -1, None].repeat(1, self.S - S_local, 1, 1, 1)], 1)

            if p_idx_end - p_idx_start > 0:
                queried_t = (query_points_t_[p_idx_start:p_idx_end] - w_idx_start)
                featxy_init, featyz_init, featxz_init = self.sample_trifeat(
                    t=queried_t,
                    featMapxy=fmap_xy_seq,
                    featMapyz=fmap_yz_seq,
                    featMapxz=fmap_xz_seq,
                    coords=coords_init_[:, :1, p_idx_start:p_idx_end],
                )
                feat_init_curr = torch.stack([featxy_init, featyz_init, featxz_init], dim=-1)
                feat_init = smart_cat(feat_init, feat_init_curr, dim=2)

            # Update the initial coordinates and visibility for non-first windows
            if p_idx_start > 0:
                last_coords = coords[-1][:, self.S // 2:].clone()  # Take the predicted coords from the last window
                coords_init_[:, : self.S // 2, :p_idx_start] = last_coords
                coords_init_[:, self.S // 2:, :p_idx_start] = last_coords[:, -1].repeat(1, self.S // 2, 1, 1)

                last_vis = vis[:, self.S // 2:][..., None]
                vis_init_[:, : self.S // 2, :p_idx_start] = last_vis
                vis_init_[:, self.S // 2:, :p_idx_start] = last_vis[:, -1].repeat(1, self.S // 2, 1, 1)

            track_mask_current = track_mask_[:, w_idx_start: w_idx_start + self.S, :p_idx_end]
            if S_local < self.S:
                track_mask_current = torch.cat([
                    track_mask_current,
                    track_mask_current[:, -1:].repeat(1, self.S - S_local, 1, 1),
                ], 1)

            coords, vis, _ = self.forward_iteration(
                fmapXY=fmap_xy_seq,
                fmapYZ=fmap_yz_seq,
                fmapXZ=fmap_xz_seq,
                coords_init=coords_init_[:, :, :p_idx_end],
                feat_init=feat_init[:, :, :p_idx_end],
                vis_init=vis_init_[:, :, :p_idx_end],
                track_mask=track_mask_current,
                iters=iters,
            )

            coords_in_worldspace = [world_to_triplane(coord, inverse=True) for coord in coords]

            if is_train:
                coord_predictions.append([coord[:, :S_local] for coord in coords_in_worldspace])
                vis_predictions.append(vis[:, :S_local])

            traj_e_[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = coords_in_worldspace[-1][:, :S_local]
            vis_e_[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = torch.sigmoid(vis[:, :S_local])

            track_mask_[:, : w_idx_start + self.S, :p_idx_end] = 0.0
            w_idx_start = w_idx_start + self.S // 2

            p_idx_start = p_idx_end

        traj_e = traj_e_[:, :, inv_sort_inds]
        vis_e = vis_e_[:, :, inv_sort_inds]

        results = {
            "traj_e": traj_e,
            "feat_init": feat_init,
            "vis_e": vis_e,
        }
        if self.is_train:
            results["train_data"] = {
                "vis_predictions": vis_predictions,
                "coord_predictions": coord_predictions,
                "attn_predictions": None,
                "p_idx_end_list": p_idx_end_list,
                "sort_inds": sort_inds,
                "Rigid_ln_total": None,
            }
        return results

    @staticmethod
    def _plot_pointcloud(logs_path, filename, xyz, c, q_xyz=None, q_c=None,
                         elevations=(0, 30, 90), azimuths=(0, 45, 90), show=False):
        fig = plt.figure(figsize=(len(azimuths) * 4.8, len(elevations) * 4.8))
        fig.suptitle(filename)
        for i, elev_ in enumerate(elevations):
            for j, azim in enumerate(azimuths):
                ax = fig.add_subplot(len(elevations), len(azimuths), i * len(azimuths) + j + 1, projection='3d')
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=1, marker=".", label="RGBD pointcloud")
                if q_xyz is not None:
                    ax.scatter(q_xyz[:, 0], q_xyz[:, 1], q_xyz[:, 2], c=q_c, s=3, marker="^", label="Query Points")
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend()
                ax.view_init(elev=elev_, azim=azim)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(logs_path, f"{filename}.png"))
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def _plot_featuremaps(
            logs_path,
            filename,
            fmaps_before_splatting,
            splatted_fmaps,
            splat_normalization,
            chosen_channels=(-3, -2, -1),
    ):
        num_frames, n_channels, height_before, width_before = fmaps_before_splatting.shape
        _, _, height_after, width_after = splatted_fmaps.shape

        assert fmaps_before_splatting.shape == (num_frames, n_channels, height_before, width_before)
        assert splatted_fmaps.shape == (num_frames, n_channels, height_after, width_after)
        assert splat_normalization.shape == (num_frames, 1, height_after, width_after)

        fmaps_before_splatting = fmaps_before_splatting.detach().cpu().float().numpy()
        splatted_fmaps = splatted_fmaps.detach().cpu().float().numpy()
        splat_normalization = splat_normalization.detach().cpu().float().numpy()

        # Extract the chosen channels and normalize them
        fmaps_before_splatting = fmaps_before_splatting[:, chosen_channels, :, :]
        splatted_fmaps = splatted_fmaps[:, chosen_channels, :, :]

        ch_min = fmaps_before_splatting.min(axis=(0, 2, 3), keepdims=True)
        ch_max = fmaps_before_splatting.max(axis=(0, 2, 3), keepdims=True)
        fmaps_before_splatting = (fmaps_before_splatting - ch_min) / (ch_max - ch_min)
        splatted_fmaps = (splatted_fmaps - ch_min) / (ch_max - ch_min)

        # Normalize the normalization as well ( ͡° ͜ʖ ͡°)
        splat_normalization = splat_normalization / splat_normalization.max()

        # Pad the shorter side to match the longer side
        if width_before != width_after:
            if width_after > width_before:
                fmaps_before_splatting = np.pad(
                    fmaps_before_splatting,
                    ((0, 0), (0, 0), (0, 0), (0, width_after - width_before)),
                    mode='constant',
                    constant_values=0
                )
            else:
                splatted_fmaps = np.pad(
                    splatted_fmaps,
                    ((0, 0), (0, 0), (0, 0), (0, width_before - width_after)),
                    mode='constant',
                    constant_values=0
                )
                splat_normalization = np.pad(
                    splat_normalization,
                    ((0, 0), (0, 0), (0, 0), (0, width_before - width_after)),
                    mode='constant',
                    constant_values=0
                )

        # Concatenate images along the height dimension
        splat_normalization = np.repeat(splat_normalization, 3, axis=1)
        imgs = [
            np.concatenate([
                fmaps_before_splatting[t],
                splatted_fmaps[t],
                splat_normalization[t]
            ], axis=1).transpose(1, 2, 0)[..., ::-1]
            for t in range(num_frames)
        ]

        video = cv2.VideoWriter(
            os.path.join(logs_path, f"{filename}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12,
            (imgs[0].shape[1], imgs[0].shape[0]),
        )

        for img in imgs:
            video.write((img * 255).astype(np.uint8))

        video.release()
        logging.info(f"Saved the featuremap video to {os.path.abspath(os.path.join(logs_path, f'{filename}.mp4'))}")
