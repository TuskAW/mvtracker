# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn as nn

from mvtracker.models.core.embeddings import (
    get_3d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed_from_grid,
    Embedder_Fourier,
)
from mvtracker.models.core.model_utils import (
    bilinear_sample2d, smart_cat, sample_features5d, pixel_xy_and_camera_z_to_world_space
)
from mvtracker.models.core.spatracker.blocks import (
    BasicEncoder,
    CorrBlock,
    EUpdateFormer,
    pix2cam,
    cam2pix
)
from mvtracker.models.core.spatracker.softsplat import softsplat


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def sample_pos_embed(grid_size, embed_dim, coords):
    if coords.shape[-1] == 2:
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim,
                                            grid_size=grid_size)
        pos_embed = (
            torch.from_numpy(pos_embed)
            .reshape(grid_size[0], grid_size[1], embed_dim)
            .float()
            .unsqueeze(0)
            .to(coords.device)
        )
        sampled_pos_embed = bilinear_sample2d(
            pos_embed.permute(0, 3, 1, 2),
            coords[:, 0, :, 0], coords[:, 0, :, 1]
        )
    elif coords.shape[-1] == 3:
        sampled_pos_embed = get_3d_sincos_pos_embed_from_grid(
            embed_dim, coords[:, :1, ...]
        ).float()[:, 0, ...].permute(0, 2, 1)

    return sampled_pos_embed


class SpaTracker(nn.Module):
    def __init__(
            self,
            sliding_window_len=8,
            stride=8,
            add_space_attn=True,
            num_heads=8,
            hidden_size=384,
            space_depth=12,
            time_depth=12,
            triplane_zres=128,
    ):
        super(SpaTracker, self).__init__()

        self.S = sliding_window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = latent_dim = 128
        self.b_latent_dim = self.latent_dim // 3
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.triplane_zres = triplane_zres

        # @Encoder
        self.fnet = BasicEncoder(input_dim=3,
                                 output_dim=self.latent_dim, norm_fn="instance", dropout=0,
                                 stride=stride, Embed3D=False
                                 )

        # conv head for the tri-plane features
        self.headyz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))

        self.headxz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))

        # @UpdateFormer
        self.updateformer = EUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=456,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=latent_dim + 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            flash=True
        )
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1

        self.norm = nn.GroupNorm(1, self.latent_dim)

        self.ffeat_updater = nn.Sequential(
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

        # TODO @NeuralArap: optimize the arap
        self.embed_traj = Embedder_Fourier(
            input_dim=5, max_freq_log2=5.0, N_freqs=3, include_input=True
        )
        self.embed3d = Embedder_Fourier(
            input_dim=3, max_freq_log2=10.0, N_freqs=10, include_input=True
        )
        self.embedConv = nn.Conv2d(self.latent_dim + 63,
                                   self.latent_dim, 3, padding=1)

        # @Vis_predictor
        self.vis_predictor = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.embedProj = nn.Linear(63, 456)
        self.zeroMLPflow = nn.Linear(195, 130)

    def prepare_track(self, rgbds, queries):
        """
        NOTE:
        Normalized the rgbs and sorted the queries via their first appeared time
        Args:
            rgbds: the input rgbd images (B T 4 H W)
            queries: the input queries (B N 4)
        Return:
            rgbds: the normalized rgbds (B T 4 H W)
            queries: the sorted queries (B N 4)
            track_mask:
        """
        assert (rgbds.shape[2] == 4) and (queries.shape[2] == 4)
        # Step1: normalize the rgbs input
        device = rgbds.device
        rgbds[:, :, :3, ...] = 2 * (rgbds[:, :, :3, ...] / 255.0) - 1.0
        B, T, C, H, W = rgbds.shape
        B, N, __ = queries.shape
        self.traj_e = torch.zeros((B, T, N, 3), device=device)
        self.vis_e = torch.zeros((B, T, N), device=device)

        # Step2: sort the points via their first appeared time
        first_positive_inds = queries[0, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds, dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[sort_inds]
        # check if can be inverse
        assert torch.allclose(
            first_positive_inds, first_positive_inds[sort_inds][inv_sort_inds]
        )

        # filter those points never appear points during 1 - T
        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >=
                      first_positive_inds[None, None, :]).unsqueeze(-1)

        # scale the coords_init
        coords_init = queries[:, :, 1:].reshape(B, 1, N, 3).repeat(
            1, self.S, 1, 1
        )
        coords_init[..., :2] /= float(self.stride)

        # Step3: initial the regular grid
        gridx = torch.linspace(0, W // self.stride - 1, W // self.stride)
        gridy = torch.linspace(0, H // self.stride - 1, H // self.stride)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing="ij")
        gridxy = torch.stack([gridx, gridy], dim=-1).to(rgbds.device).permute(
            2, 1, 0
        )
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        # Step4: initial traj for neural arap
        T_series = torch.linspace(0, 5, T).reshape(1, T, 1, 1).cuda()  # 1 T 1 1
        T_series = T_series.repeat(B, 1, N, 1)
        # get the 3d traj in the camera coordinates
        intr_init = self.intrs[:, queries[0, :, 0].long()]
        Traj_series = pix2cam(queries[:, :, None, 1:].double(), intr_init.double())
        # torch.inverse(intr_init.double())@queries[:,:,1:,None].double() # B N 3 1
        Traj_series = Traj_series.repeat(1, 1, T, 1).permute(0, 2, 1, 3).float()
        Traj_series = torch.cat([T_series, Traj_series], dim=-1)
        # get the indicator for the neural arap
        Traj_mask = -1e2 * torch.ones_like(T_series)
        Traj_series = torch.cat([Traj_series, Traj_mask], dim=-1)

        return (
            rgbds,
            first_positive_inds,
            first_positive_sorted_inds,
            sort_inds, inv_sort_inds,
            track_mask, gridxy, coords_init[..., sort_inds, :].clone(),
            vis_init, Traj_series[..., sort_inds, :].clone()
        )

    def sample_trifeat(self, t,
                       coords,
                       featMapxy,
                       featMapyz,
                       featMapxz):
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

    def neural_arap(self, coords, Traj_arap, intrs_S, T_mark):
        """ calculate the ARAP embedding and offset
        Args:
            coords: the coordinates of the current points   1 S N' 3
            Traj_arap: the trajectory of the points   1 T N' 5
            intrs_S: the camera intrinsics B S 3 3

        """
        coords_out = coords.clone()
        coords_out[..., :2] *= float(self.stride)
        coords_out[..., 2] = coords_out[..., 2] / self.Dz
        coords_out[..., 2] = coords_out[..., 2] * (self.d_far - self.d_near) + self.d_near
        intrs_S = intrs_S[:, :, None, ...].repeat(1, 1, coords_out.shape[2], 1, 1)
        B, S, N, D = coords_out.shape
        if S != intrs_S.shape[1]:
            intrs_S = torch.cat(
                [intrs_S, intrs_S[:, -1:].repeat(1, S - intrs_S.shape[1], 1, 1, 1)], dim=1
            )
            T_mark = torch.cat(
                [T_mark, T_mark[:, -1:].repeat(1, S - T_mark.shape[1], 1)], dim=1
            )
        xyz_ = pix2cam(coords_out.double(), intrs_S.double()[:, :, 0])
        xyz_ = xyz_.float()
        xyz_embed = torch.cat([T_mark[..., None], xyz_,
                               torch.zeros_like(T_mark[..., None])], dim=-1)

        xyz_embed = self.embed_traj(xyz_embed)
        Traj_arap_embed = self.embed_traj(Traj_arap)
        d_xyz, traj_feat = self.arapFormer(xyz_embed, Traj_arap_embed)
        # update in camera coordinate
        xyz_ = xyz_ + d_xyz.clamp(-5, 5)
        # project back to the image plane
        coords_out = cam2pix(xyz_.double(), intrs_S[:, :, 0].double()).float()
        # resize back
        coords_out[..., :2] /= float(self.stride)
        coords_out[..., 2] = (coords_out[..., 2] - self.d_near) / (self.d_far - self.d_near)
        coords_out[..., 2] *= self.Dz

        return xyz_, coords_out, traj_feat

    def gradient_arap(self, coords, aff_avg=None, aff_std=None, aff_f_sg=None,
                      iter=0, iter_num=4, neigh_idx=None, intr=None, msk_track=None):
        with torch.enable_grad():
            coords.requires_grad_(True)
            y = self.ARAP_ln(coords, aff_f_sg=aff_f_sg, neigh_idx=neigh_idx,
                             iter=iter, iter_num=iter_num, intr=intr, msk_track=msk_track)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=coords,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]

        return gradients.detach()

    def forward_iteration(
            self,
            fmapXY,
            fmapYZ,
            fmapXZ,
            coords_init,
            feat_init=None,
            vis_init=None,
            track_mask=None,
            iters=4,
            intrs_S=None,
    ):
        B, S_init, N, D = coords_init.shape
        assert D == 3
        assert B == 1
        B, S, __, H8, W8 = fmapXY.shape
        device = fmapXY.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)],
                dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            intrs_S = torch.cat(
                [intrs_S, intrs_S[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

        fcorr_fnXY = CorrBlock(
            fmapXY, num_levels=self.corr_levels, radius=self.corr_radius
        )
        fcorr_fnYZ = CorrBlock(
            fmapYZ, num_levels=self.corr_levels, radius=self.corr_radius
        )
        fcorr_fnXZ = CorrBlock(
            fmapXZ, num_levels=self.corr_levels, radius=self.corr_radius
        )

        ffeats = torch.split(feat_init.clone(), dim=-1, split_size_or_sections=1)
        ffeats = [f.squeeze(-1) for f in ffeats]

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=456,
            coords=coords[..., :2],
        )
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)

        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        coord_predictions = []
        attn_predictions = []
        Rot_ln = 0
        support_feat = self.support_features

        for __ in range(iters):
            coords = coords.detach()
            # if self.args.if_ARAP == True:
            #     # refine the track with arap
            #     xyz_pred, coords, flows_cat0 = self.neural_arap(coords.detach(),
            #                                                    Traj_arap.detach(),
            #                                                    intrs_S, T_mark)
            fcorrsXY = fcorr_fnXY.corr_sample(ffeats[0], coords[..., :2])
            fcorrsYZ = fcorr_fnYZ.corr_sample(ffeats[1], coords[..., [1, 2]])
            fcorrsXZ = fcorr_fnXZ.corr_sample(ffeats[2], coords[..., [0, 2]])
            # fcorrs = fcorrsXY
            fcorrs = fcorrsXY + fcorrsYZ + fcorrsXZ
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, 64, cat_coords=True)
            flows_cat = self.zeroMLPflow(flows_cat)

            ffeats_xy = ffeats[0].permute(0,
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_yz = ffeats[1].permute(0,
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_xz = ffeats[2].permute(0,
                                          2, 1, 3).reshape(B * N, S, self.latent_dim)
            ffeats_ = ffeats_xy + ffeats_yz + ffeats_xz

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, 2)
            )

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)

            if transformer_input.shape[-1] < pos_embed.shape[-1]:
                # padding the transformer_input to the same dimension as pos_embed
                transformer_input = F.pad(
                    transformer_input, (0, pos_embed.shape[-1] - transformer_input.shape[-1]),
                    "constant", 0
                )

            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta, delta_se3F = self.updateformer(x, support_feat)
            support_feat = support_feat + delta_se3F[0] / 100
            delta = rearrange(delta, " b n t d -> (b n) t d")
            d_coord = delta[:, :, :3]
            d_feats = delta[:, :, 3:]

            ffeats_xy = self.ffeat_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_xy.reshape(-1,
                                                                                                             self.latent_dim)
            ffeats_yz = self.ffeatyz_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_yz.reshape(-1,
                                                                                                               self.latent_dim)
            ffeats_xz = self.ffeatxz_updater(self.norm(d_feats.view(-1, self.latent_dim))) + ffeats_xz.reshape(-1,
                                                                                                               self.latent_dim)
            ffeats[0] = ffeats_xy.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            ffeats[1] = ffeats_yz.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            ffeats[2] = ffeats_xz.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C
            coords = coords + d_coord.reshape(B, N, S, 3).permute(0, 2, 1, 3)
            if torch.isnan(coords).any():
                # import ipdb;
                # ipdb.set_trace()
                logging.error("nan in coords")

            coords_out = coords.clone()
            coords_out[..., :2] *= float(self.stride)

            coords_out[..., 2] = coords_out[..., 2] / self.Dz
            coords_out[..., 2] = coords_out[..., 2] * (self.d_far - self.d_near) + self.d_near

            coord_predictions.append(coords_out)

        ffeats_f = ffeats[0] + ffeats[1] + ffeats[2]
        vis_e = self.vis_predictor(ffeats_f.reshape(B * S * N, self.latent_dim)).reshape(
            B, S, N
        )
        self.support_features = support_feat.detach()
        return coord_predictions, attn_predictions, vis_e, feat_init, Rot_ln

    def forward(self, rgbds, queries, iters=4, feat_init=None, is_train=False, intrs=None):
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1
        self.is_train = is_train
        B, T, C, H, W = rgbds.shape
        # set the intrinsic or simply initialized
        if intrs is None:
            intrs = torch.from_numpy(np.array([[W, 0.0, W // 2],
                                               [0.0, W, H // 2],
                                               [0.0, 0.0, 1.0]]))
            intrs = intrs[None,
            None, ...].repeat(B, T, 1, 1).float().to(rgbds.device)
        self.intrs = intrs

        # prepare the input for tracking
        (
            rgbds,
            first_positive_inds,
            first_positive_sorted_inds, sort_inds,
            inv_sort_inds, timestep_should_be_estimated_mask, gridxy,
            coords_init, vis_init, Traj_arap
        ) = self.prepare_track(rgbds.clone(), queries)
        coords_init_ = coords_init.clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()

        depth_all = rgbds[:, :, 3, ...]
        d_near = self.d_near = depth_all[depth_all > 0.01].min().item()
        d_far = self.d_far = depth_all[depth_all > 0.01].max().item()

        B, N, __ = queries.shape
        self.Dz = Dz = self.triplane_zres
        w_idx_start = 0
        p_idx_end = 0
        p_idx_start = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        attn_predictions = []
        p_idx_end_list = []
        Rigid_ln_total = 0
        while w_idx_start < T - self.S // 2:
            curr_wind_points = torch.nonzero(
                first_positive_sorted_inds < w_idx_start + self.S)
            if curr_wind_points.shape[0] == 0:
                w_idx_start = w_idx_start + self.S // 2
                logging.info(f"No points in window {w_idx_start}-{w_idx_start + self.S}; adding empty results to list")
                p_idx_end_list.append(torch.zeros((1,), dtype=torch.int64, device=first_positive_sorted_inds.device))
                if is_train:
                    vis_predictions.append(torch.zeros((B, self.S, 0), device=rgbds.device))
                    coord_predictions.append(
                        [torch.zeros((B, self.S, 0, 3), device=rgbds.device) for _ in range(iters)])
                    attn_predictions.append([-1 for _ in range(iters)])
                continue
            p_idx_end = curr_wind_points[-1] + 1
            p_idx_end_list.append(p_idx_end)
            # the T may not be divided by self.S
            rgbds_seq = rgbds[:, w_idx_start:w_idx_start + self.S].clone()
            S = S_local = rgbds_seq.shape[1]
            if S < self.S:
                rgbds_seq = torch.cat(
                    [rgbds_seq,
                     rgbds_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbds_seq.shape[1]

            rgbs_ = rgbds_seq.reshape(B * S, C, H, W)[:, :3]
            depths = rgbds_seq.reshape(B * S, C, H, W)[:, 3:].clone()
            # open the mask
            # Traj_arap[:, w_idx_start:w_idx_start + self.S, :p_idx_end, -1] = 0
            # step1: normalize the depth map

            depths = (depths - d_near) / (d_far - d_near)
            depths_dn = nn.functional.interpolate(
                depths, scale_factor=1.0 / self.stride, mode="nearest")
            depths_dnG = depths_dn * Dz

            # step2: normalize the coordinate
            coords_init_[:, :, p_idx_start:p_idx_end, 2] = (
                                                                   coords_init[:, :, p_idx_start:p_idx_end, 2] - d_near
                                                           ) / (d_far - d_near)
            coords_init_[:, :, p_idx_start:p_idx_end, 2] *= Dz

            # efficient triplane splatting
            gridxyz = torch.cat([gridxy[None, ...].repeat(
                depths_dn.shape[0], 1, 1, 1), depths_dnG], dim=1)
            Fxy2yz = gridxyz[:, [1, 2], ...] - gridxyz[:, :2]
            Fxy2xz = gridxyz[:, [0, 2], ...] - gridxyz[:, :2]
            gridxyz_nm = gridxyz.clone()
            gridxyz_nm[:, 0, ...] = (gridxyz_nm[:, 0, ...] - gridxyz_nm[:, 0, ...].min()) / (
                    gridxyz_nm[:, 0, ...].max() - gridxyz_nm[:, 0, ...].min())
            gridxyz_nm[:, 1, ...] = (gridxyz_nm[:, 1, ...] - gridxyz_nm[:, 1, ...].min()) / (
                    gridxyz_nm[:, 1, ...].max() - gridxyz_nm[:, 1, ...].min())
            gridxyz_nm[:, 2, ...] = (gridxyz_nm[:, 2, ...] - gridxyz_nm[:, 2, ...].min()) / (
                    gridxyz_nm[:, 2, ...].max() - gridxyz_nm[:, 2, ...].min())
            gridxyz_nm = 2 * (gridxyz_nm - 0.5)
            _, _, h4, w4 = gridxyz_nm.shape
            gridxyz_nm = gridxyz_nm.permute(0, 2, 3, 1).reshape(S * h4 * w4, 3)
            featPE = self.embed3d(gridxyz_nm).view(S, h4, w4, -1).permute(0, 3, 1, 2)
            if fmaps_ is None:
                fmaps_ = torch.cat([self.fnet(rgbs_), featPE], dim=1)
                fmaps_ = self.embedConv(fmaps_)
            else:
                fmaps_new = torch.cat([self.fnet(rgbs_[self.S // 2:]), featPE[self.S // 2:]], dim=1)
                fmaps_new = self.embedConv(fmaps_new)
                fmaps_ = torch.cat(
                    [fmaps_[self.S // 2:], fmaps_new], dim=0
                )

            fmapXY = fmaps_[:, :self.latent_dim].reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )

            fmapYZ = softsplat(fmapXY[0], Fxy2yz, None,
                               strMode="avg", tenoutH=self.Dz, tenoutW=H // self.stride)
            fmapXZ = softsplat(fmapXY[0], Fxy2xz, None,
                               strMode="avg", tenoutH=self.Dz, tenoutW=W // self.stride)

            fmapYZ = self.headyz(fmapYZ)[None, ...]
            fmapXZ = self.headxz(fmapXZ)[None, ...]

            if p_idx_end - p_idx_start > 0:
                queried_t = (first_positive_sorted_inds[p_idx_start:p_idx_end]
                             - w_idx_start)
                (featxy_init,
                 featyz_init,
                 featxz_init) = self.sample_trifeat(
                    t=queried_t, featMapxy=fmapXY,
                    featMapyz=fmapYZ, featMapxz=fmapXZ,
                    coords=coords_init_[:, :1, p_idx_start:p_idx_end]
                )
                # T, S, N, C, 3
                feat_init_curr = torch.stack([featxy_init,
                                              featyz_init, featxz_init], dim=-1)
                feat_init = smart_cat(feat_init, feat_init_curr, dim=2)

            if p_idx_start > 0:
                # preprocess the coordinates of last windows
                last_coords = coords[-1][:, self.S // 2:].clone()
                last_coords[..., :2] /= float(self.stride)
                last_coords[..., 2:] = (last_coords[..., 2:] - d_near) / (d_far - d_near)
                last_coords[..., 2:] = last_coords[..., 2:] * Dz

                coords_init_[:, : self.S // 2, :p_idx_start] = last_coords
                coords_init_[:, self.S // 2:, :p_idx_start] = last_coords[
                                                              :, -1
                                                              ].repeat(1, self.S // 2, 1, 1)

                last_vis = vis[:, self.S // 2:].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :p_idx_start] = last_vis
                vis_init_[:, self.S // 2:, :p_idx_start] = last_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            coords, attns, vis, __, Rigid_ln = self.forward_iteration(
                fmapXY=fmapXY,
                fmapYZ=fmapYZ,
                fmapXZ=fmapXZ,
                coords_init=coords_init_[:, :, :p_idx_end],
                feat_init=feat_init[:, :, :p_idx_end],
                vis_init=vis_init_[:, :, :p_idx_end],
                track_mask=timestep_should_be_estimated_mask[:, w_idx_start: w_idx_start + self.S, :p_idx_end],
                iters=iters,
                intrs_S=self.intrs[:, w_idx_start: w_idx_start + self.S],
            )

            Rigid_ln_total += Rigid_ln

            if is_train:
                vis_predictions.append(vis[:, :S_local])
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                attn_predictions.append(attns)

            self.traj_e[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = coords[-1][:, :S_local]
            self.vis_e[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = vis[:, :S_local]

            timestep_should_be_estimated_mask[:, : w_idx_start + self.S, :p_idx_end] = 0.0
            w_idx_start = w_idx_start + self.S // 2

            p_idx_start = p_idx_end

        self.traj_e = self.traj_e[:, :, inv_sort_inds]
        self.vis_e = self.vis_e[:, :, inv_sort_inds]

        self.vis_e = torch.sigmoid(self.vis_e)
        train_data = (
            (vis_predictions, coord_predictions, attn_predictions,
             p_idx_end_list, sort_inds, Rigid_ln_total)
        )
        if self.is_train:
            return self.traj_e, feat_init, self.vis_e, train_data
        else:
            return self.traj_e, feat_init, self.vis_e


class SpaTrackerMultiViewAdapter(nn.Module):
    def __init__(self, **kwargs):
        super(SpaTrackerMultiViewAdapter, self).__init__()
        self.spatracker = SpaTracker(**kwargs)

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
            query_points_view=None,
            **kwargs,
    ):
        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape

        depths = depths.clamp(max=36.0)

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        if feat_init is not None:
            raise NotImplementedError("feat_init is not supported yet")

        # Project the queries to each view
        query_points_t = query_points[:, :, :1].long()
        query_points_xyz_worldspace = query_points[:, :, 1:]

        query_points_xy_pixelspace_per_view = query_points.new_zeros((batch_size, num_views, num_points, 2))
        query_points_z_cameraspace_per_view = query_points.new_zeros((batch_size, num_views, num_points, 1))
        for batch_idx in range(batch_size):
            for t in query_points_t[batch_idx].unique():
                query_points_t_mask = query_points_t[batch_idx].squeeze(-1) == t
                point_3d_world = query_points_xyz_worldspace[batch_idx][query_points_t_mask]

                # World to camera space
                point_4d_world_homo = torch.cat(
                    [point_3d_world, point_3d_world.new_ones(point_3d_world[..., :1].shape)], -1)
                point_3d_camera = torch.einsum('Aij,Bj->ABi', extrs[batch_idx, :, t, :, :], point_4d_world_homo[:, :])

                # Camera to pixel space
                point_2d_pixel_homo = torch.einsum('Aij,ABj->ABi', intrs[batch_idx, :, t, :, :], point_3d_camera[:, :])
                point_2d_pixel = point_2d_pixel_homo[..., :2] / point_2d_pixel_homo[..., 2:]

                query_points_xy_pixelspace_per_view[batch_idx, :, query_points_t_mask] = point_2d_pixel
                query_points_z_cameraspace_per_view[batch_idx, :, query_points_t_mask] = point_3d_camera[..., -1:]

        # Estimate occlusion mask in each view based on depth maps
        query_points_depth_in_view = query_points.new_zeros((batch_size, num_views, num_points, 1))
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                for t in query_points_t[batch_idx].unique():
                    query_points_t_mask = query_points_t[batch_idx].squeeze(-1) == t
                    interpolated_depth = bilinear_sample2d(
                        im=depths[batch_idx, view_idx, t][None],
                        x=query_points_xy_pixelspace_per_view[batch_idx, view_idx, query_points_t_mask, 0][None],
                        y=query_points_xy_pixelspace_per_view[batch_idx, view_idx, query_points_t_mask, 1][None],
                    )[0].permute(1, 0).type(query_points.dtype)
                    query_points_depth_in_view[batch_idx, view_idx, query_points_t_mask] = interpolated_depth

        query_points_depth_in_view_masked = query_points_depth_in_view.clone()
        query_points_outside_of_view_box = (
                (query_points_xy_pixelspace_per_view[..., 0] < 0) |
                (query_points_xy_pixelspace_per_view[..., 0] >= width) |
                (query_points_xy_pixelspace_per_view[..., 1] < 0) |
                (query_points_xy_pixelspace_per_view[..., 1] >= height) |
                (query_points_z_cameraspace_per_view[..., 0] < 0)
        )
        if query_points_outside_of_view_box.all(1).any():
            warnings.warn(f"There are some query points that are outside of the frame of every view: "
                          f"{query_points_xy_pixelspace_per_view[query_points_outside_of_view_box.all(1)[:, None, :].repeat(1, num_views, 1)].reshape(num_views, -1, 2).permute(1, 0, 2)}")
        query_points_depth_in_view_masked[query_points_outside_of_view_box] = -1e4
        # query_points_occluded_by_depthmap = (query_points_depth_in_view * 1.1 < query_points_z_cameraspace_per_view)
        # query_points_depth_in_view_masked[query_points_occluded_by_depthmap] = -1e3

        query_points_best_visibility_view = (
                query_points_depth_in_view_masked - query_points_z_cameraspace_per_view).argmax(1)
        query_points_best_visibility_view = query_points_best_visibility_view.squeeze(-1)

        if query_points_view is not None:
            query_points_best_visibility_view = query_points_view
            logging.info(f"Using the provided query_points_view instead of the estimated one")

        assert batch_size == 1, "Batch size > 1 is not supported yet"
        batch_idx = 0

        results = {}

        # Call the SpaTracker for each view
        traj_e_per_view = {}
        feat_init_per_view = {}
        vis_e_per_view = {}
        train_data_per_view = {}
        for view_idx in range(num_views):
            track_mask = query_points_best_visibility_view[batch_idx] == view_idx
            if track_mask.sum() == 0:
                continue

            view_query_points = torch.concat([
                query_points_t[batch_idx, :, :][track_mask],
                query_points_xy_pixelspace_per_view[batch_idx, view_idx, :, :][track_mask],
                query_points_z_cameraspace_per_view[batch_idx, view_idx, :, :][track_mask],
            ], dim=-1)

            view_rgbds = torch.concat([rgbs[batch_idx, view_idx], depths[batch_idx, view_idx]], dim=1)
            view_intrs = intrs[batch_idx, view_idx]
            view_extrs = extrs[batch_idx, view_idx]

            output_tuple = self.spatracker(
                rgbds=view_rgbds[None],
                queries=view_query_points[None],
                intrs=view_intrs[None],
                iters=iters,
                feat_init=None,
                is_train=is_train,
            )
            if is_train:
                view_traj_e, view_feat_init, view_vis_e, view_train_data = output_tuple
            else:
                view_traj_e, view_feat_init, view_vis_e = output_tuple

            # Project points to the world space
            intrs_inv = torch.inverse(view_intrs.float())
            view_extrs_square = torch.eye(4).to(view_extrs.device)[None].repeat(num_frames, 1, 1)
            view_extrs_square[:, :3, :] = view_extrs
            extrs_inv = torch.inverse(view_extrs_square.float())
            view_traj_e = pixel_xy_and_camera_z_to_world_space(
                pixel_xy=view_traj_e[0, ..., :-1].float(),
                camera_z=view_traj_e[0, ..., -1:].float(),
                intrs_inv=intrs_inv,
                extrs_inv=extrs_inv,
            )[None]
            if is_train:
                num_windows = len(view_train_data[1])
                num_iterations = len(view_train_data[1][0])
                coord_predictions = view_train_data[1]
                window_start_t = 0
                while window_start_t < num_frames - self.spatracker.S // 2:
                    window_idx = window_start_t // (self.spatracker.S // 2)
                    for iteration_idx in range(num_iterations):
                        coord_predictions[window_idx][iteration_idx] = pixel_xy_and_camera_z_to_world_space(
                            pixel_xy=coord_predictions[window_idx][iteration_idx][0, ..., :-1].float(),
                            camera_z=coord_predictions[window_idx][iteration_idx][0, ..., -1:].float(),
                            intrs_inv=intrs_inv[window_start_t:window_start_t + self.spatracker.S],
                            extrs_inv=extrs_inv[window_start_t:window_start_t + self.spatracker.S],
                        )[None]
                    window_start_t = window_start_t + (self.spatracker.S // 2)
                assert window_idx == num_windows - 1, "The last window should be the last one"
                assert view_train_data[1] == coord_predictions, "The view_train_data[1] should be updated in-place"

            # Set the trajectory to (0,0,0) for the timesteps before the query timestep
            for point_idx, t in enumerate(query_points_t[batch_idx, :, :].squeeze(-1)[track_mask]):
                view_traj_e[0, :t, point_idx, :] = 0.0

            traj_e_per_view[view_idx] = view_traj_e
            feat_init_per_view[view_idx] = view_feat_init
            vis_e_per_view[view_idx] = view_vis_e
            if is_train:
                train_data_per_view[view_idx] = view_train_data

        # Merging the results from all views
        views_to_keep = list(traj_e_per_view.keys())
        traj_e = torch.cat([traj_e_per_view[view_idx] for view_idx in views_to_keep], dim=2)
        vis_e = torch.cat([vis_e_per_view[view_idx] for view_idx in views_to_keep], dim=2)
        feat_init = torch.cat([feat_init_per_view[view_idx] for view_idx in views_to_keep], dim=2)

        # Sort the traj_e and vis_e based on the original indices, since concatenating the results from all views
        # will first put the results from the first view, then the results from the second view, and so on.
        # But we want to keep the trajectories order to match the original query points order.
        sort_inds = []
        for view_idx in views_to_keep:
            track_mask = query_points_best_visibility_view[batch_idx] == view_idx
            if track_mask.sum() == 0:
                continue
            global_indices = torch.nonzero(track_mask).squeeze(-1)
            sort_inds += [global_indices]
        sort_inds = torch.cat(sort_inds, dim=0)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)

        # Use the inv_sort_inds to sort the traj_e and vis_e
        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        feat_init = None  # Not supported yet, correct sorting needs to be implemented

        # Delete the intermediate variables to avoid confusion with the later variables
        del sort_inds, inv_sort_inds

        # # Sanity check that the sorted traj_e have about similar values for the query points
        # # The forward pass is expected to tweak the values a bit, but they would probably stay close
        # pred_xyz_for_query = traj_e[0][query_points_t[batch_idx].squeeze(-1), torch.arange(num_points)]
        # pred_xyz_for_query = pred_xyz_for_query.type(query_points_xyz_worldspace.dtype)
        # assert torch.allclose(pred_xyz_for_query, query_points_xyz_worldspace[batch_idx], atol=1)
        # # But, an untrained model might not be able to predict the query points exactly

        # # Also check that the query points are visible
        # pred_visibility_for_query = vis_e[0][query_points_t[batch_idx].squeeze(-1), torch.arange(num_points)]
        # assert torch.all(pred_visibility_for_query > 0.5)
        # # But, for some points the model might predict the query points to be occluded

        if not is_train:
            if torch.isnan(traj_e).any():
                warnings.warn(
                    f"Found {torch.isnan(traj_e).sum()}/{traj_e.numel()} NaN values in traj_e. Setting them to 0.")
                traj_e[traj_e.isnan()] = 0
            if torch.isnan(vis_e).any():
                warnings.warn(
                    f"Found {torch.isnan(vis_e).sum()}/{vis_e.numel()} NaN values in visibilities. Setting them to 1.")
                vis_e[vis_e.isnan()] = 1

        # Save to results
        results["traj_e"] = traj_e
        results["feat_init"] = feat_init
        results["vis_e"] = vis_e

        # If training mode, we need to merge the results from all views.
        # Those merged results are used in the backward pass to compute the loss.
        # train_data is a tuple of (vis_pred, coord_pred, attn_pred, p_idx_end_list, sort_inds, Rigid_ln_total)
        if is_train:
            # SpaTracker is using sliding windows, and for each window, it is using multiple iterations.
            num_windows = len(train_data_per_view[views_to_keep[0]][0])
            num_iterations = len(train_data_per_view[views_to_keep[0]][1][0])

            sort_inds = []
            vis_predictions = [[] for _ in range(num_windows)]
            coord_predictions = [[[] for _ in range(num_iterations)] for _ in range(num_windows)]
            for window_idx in range(num_windows):
                for view_idx in views_to_keep:
                    # What points will be tracked in this view
                    track_mask = query_points_best_visibility_view[batch_idx] == view_idx
                    if track_mask.sum() == 0:
                        # This view does not track any points at all
                        continue

                    # Get the indices of points that appeared in this window (from the points tracked in this view)
                    try:
                        start_idx = 0 if window_idx == 0 else train_data_per_view[view_idx][3][window_idx - 1].item()
                        end_idx = train_data_per_view[view_idx][3][window_idx].item()
                        if end_idx == 0:
                            # No points from this view were tracked in this window
                            continue
                    except Exception as e:
                        logging.error(f"Error: {e}")
                        logging.error(f"view_idx: {view_idx}, window_idx: {window_idx}")
                        logging.error(f"train_data_per_view[view_idx][3]: {train_data_per_view[view_idx][3]}")
                        raise e

                    # Convert the view-specific sorted indices to "global" indices
                    # that say which trajectory/query the point originally belonged to
                    indices_in_view = train_data_per_view[view_idx][4][start_idx:end_idx]
                    global_indices = torch.nonzero(track_mask).squeeze(-1)[indices_in_view]

                    # Sorted indices are saying how the original trajectories were reordered/sorted
                    # in the return results. This is because in the forward passes, we want to group
                    # the points that will appear in the same window together. The points that haven't
                    # appeared in a window will not be used in the forward pass for that window.
                    # For each new window, points can only be added, not removed, and they will be added
                    # if they have just appeared in that window. Since we are merging the results from
                    # all views, we will first take all the points that appeared in the first window from
                    # all views, then all the points that appeared in the second window from all views,
                    # and so on. This is why we do a for loop over the windows first, then over the views
                    # and merge the indices in the next line:
                    sort_inds.append(global_indices)
                    # The indices are now sorted in the order that they will appear in the merged results.
                    # This can be illustrated as follows:
                    #   Final sorted indices for the merged results: [
                    #     view 1 new points from window 1
                    #     view 2 new points from window 1
                    #     view ... new points from window 1
                    #     view 1 new points from window 2
                    #     view 2 new points from window 2
                    #     view ... new points from window 2
                    #     ...
                    #   ]

                    # This also means that the results from each view need to be carefully merged to match
                    # the expected ordering/sorting. To illustrate this, the merged results for the vis_predictions
                    # and coord_predictions will look like this:
                    #   Window 1 results: [
                    #     view 1 new points from window 1
                    #     view 2 new points from window 1
                    #     view ... new points from window 1
                    #   ]
                    #   Window 2 results: [
                    #     view 1 new points from window 1
                    #     view 2 new points from window 1
                    #     view ... new points from window 1
                    #     view 1 new points from window 2
                    #     view 2 new points from window 2
                    #     view ... new points from window 2
                    #   ]
                    #   Window ...

                    # Below we will merge the results from all views for each window as illustrated above
                    for window_idx_inner in range(num_windows):
                        vis_predictions[window_idx_inner].append(
                            train_data_per_view[view_idx][0][window_idx_inner][:, :, start_idx:end_idx]
                        )
                        for iteration_idx in range(num_iterations):
                            coord_predictions[window_idx_inner][iteration_idx].append(
                                train_data_per_view[view_idx][1][window_idx_inner][iteration_idx][
                                :, :, start_idx:end_idx, :]
                            )

            # Concatenate the merged results correctly
            sort_inds = torch.cat(sort_inds, dim=0)
            vis_predictions = [
                torch.cat(vis_predictions[window_idx], dim=2)
                for window_idx in range(num_windows)
            ]
            coord_predictions = [
                [
                    torch.cat(coord_predictions[window_idx][iteration_idx], dim=2)
                    for iteration_idx in range(num_iterations)
                ]
                for window_idx in range(num_windows)
            ]

            # Compute the p_idx_end_list for each window, it is the sum of the number of points
            # that appeared in each view for that window as this is the way we have merged the results.
            p_idx_end_list = [
                torch.stack([
                    train_data_per_view[view_idx][3][window_idx]
                    for view_idx in views_to_keep
                ], dim=1).sum(dim=1)
                for window_idx in range(num_windows)
            ]

            # Compute the attn_predictions and Rigid_ln_total
            attn_predictions = None  # Not supported yet
            Rigid_ln_total = None  # Not supported yet

            # Sanity check that using the computed sort_inds gives the same results as the merged traj_e and vis_e
            traj_e_reproduced = traj_e.new_zeros(traj_e.shape)
            vis_e_reproduced = vis_e.new_zeros(vis_e.shape)
            window_start_t = 0
            while window_start_t < num_frames - self.spatracker.S // 2:
                window_idx = window_start_t // (self.spatracker.S // 2)
                p_idx_end = p_idx_end_list[window_idx]
                if p_idx_end == 0:
                    continue
                wind_coords = coord_predictions[window_idx][-1]
                wind_vis = vis_predictions[window_idx]
                traj_e_reproduced[:, window_start_t:window_start_t + self.spatracker.S, :p_idx_end] = wind_coords
                vis_e_reproduced[:, window_start_t:window_start_t + self.spatracker.S, :p_idx_end] = wind_vis
                window_start_t = window_start_t + (self.spatracker.S // 2)
            inv_sort_inds = torch.argsort(sort_inds, dim=0)
            traj_e_reproduced = traj_e_reproduced[:, :, inv_sort_inds]
            vis_e_reproduced = torch.sigmoid(vis_e_reproduced[:, :, inv_sort_inds])

            # Set the trajectory to (0,0,0) for the timesteps before the query timestep
            for point_idx, t in enumerate(query_points_t[batch_idx, :, :].squeeze(-1)):
                traj_e_reproduced[0, :t, point_idx, :] = 0.0

            assert torch.allclose(traj_e, traj_e_reproduced, atol=1e-3)
            assert torch.allclose(vis_e, vis_e_reproduced, atol=1e-3)

            # Save to results
            results["train_data"] = {
                "vis_predictions": vis_predictions,
                "coord_predictions": coord_predictions,
                "attn_predictions": attn_predictions,
                "p_idx_end_list": p_idx_end_list,
                "sort_inds": sort_inds,
                "Rigid_ln_total": Rigid_ln_total,
            }

        return results
