"""
Before running the script, you need to install the toolkit and other
dependencies, as well as download the data and necessary MANO checkpoints/models.

Install the toolkit and dependencies:
```sh
# Create a new conda environment
conda create -n dexycb python=3.9
conda activate dexycb
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c iopath iopath
pip install --upgrade setuptools wheel
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu121_pyt241/download.html
conda install ninja scipy matplotlib -c conda-forge
pip install numpy==1.21.6 matplotlib==3.6 pandas==2.0 scikit-image scipy==1.11 rerun-sdk pyembree rtree --no-deps

# Install dex-ycb-toolkit
cd /home/frrajic/xode/03-macos/
git clone --recursive git@github.com:NVlabs/dex-ycb-toolkit.git
cd dex-ycb-toolkit
pip install -e .

# Install bop_toolkit dependencies
cd bop_toolkit
pip install -r requirements.txt
cd ..

# Install manopth
cd manopth
pip install -e .
cd ..

# Make sure numpy version is not too high (so that np.bool is not deprecated)
pip install numpy==1.21.6 matplotlib==3.6 pandas==2.0 scikit-image scipy==1.11 rerun-sdk pyembree rtree --no-deps
```

Download the DexYCB dataset from the [project site](https://dex-ycb.github.io):
```sh
export DEX_YCB_DIR=/home/frrajic/xode/00-data/dex-january-2025
cd $DEX_YCB_DIR

#  20200709-subject-01.tar.gz (12G)
#  20200813-subject-02.tar.gz (12G)
#  20200820-subject-03.tar.gz (12G)
#  20200903-subject-04.tar.gz (12G)
#  20200908-subject-05.tar.gz (12G)
#  20200918-subject-06.tar.gz (12G)
#  20200928-subject-07.tar.gz (12G)
#  20201002-subject-08.tar.gz (12G)
#  20201015-subject-09.tar.gz (12G)
#  20201022-subject-10.tar.gz (12G)
gdown --fuzzy https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-/view?usp=sharing &

#  bop.tar.gz (1.2G)
#  calibration.tar.gz (16K)
#  models.tar.gz (1.4G)
gdown --fuzzy https://drive.google.com/file/d/1CPqLjsaYNjE3xSJbuWmqaMsGvyGIxiKL/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ/view?usp=sharing &
gdown --fuzzy https://drive.google.com/file/d/1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM/view?usp=sharing &

tar xvf 20200709-subject-01.tar.gz &
tar xvf 20200813-subject-02.tar.gz &
tar xvf 20200820-subject-03.tar.gz &
tar xvf 20200903-subject-04.tar.gz &
tar xvf 20200908-subject-05.tar.gz &
tar xvf 20200918-subject-06.tar.gz &
tar xvf 20200928-subject-07.tar.gz &
tar xvf 20201002-subject-08.tar.gz &
tar xvf 20201015-subject-09.tar.gz &
tar xvf 20201022-subject-10.tar.gz &
tar xvf bop.tar.gz &
tar xvf calibration.tar.gz &
tar xvf models.tar.gz &

rm 20200709-subject-01.tar.gz
rm 20200813-subject-02.tar.gz
rm 20200820-subject-03.tar.gz
rm 20200903-subject-04.tar.gz
rm 20200908-subject-05.tar.gz
rm 20200918-subject-06.tar.gz
rm 20200928-subject-07.tar.gz
rm 20201002-subject-08.tar.gz
rm 20201015-subject-09.tar.gz
rm 20201022-subject-10.tar.gz
rm bop.tar.gz
rm calibration.tar.gz
rm models.tar.gz
```

The structure of the dataset should look like this:
```sh
tree -L 1 $DEX_YCB_DIR
# /home/frrajic/xode/00-data/dex-january-2025
# ├── 20200709-subject-01
# ├── 20200813-subject-02
# ├── 20200820-subject-03
# ├── 20200903-subject-04
# ├── 20200908-subject-05
# ├── 20200918-subject-06
# ├── 20200928-subject-07
# ├── 20201002-subject-08
# ├── 20201015-subject-09
# ├── 20201022-subject-10
# ├── bop
# ├── calibration
# └── models

du -sch $DEX_YCB_DIR/*
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200709-subject-01
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200813-subject-02
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200820-subject-03
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200903-subject-04
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200908-subject-05
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200918-subject-06
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20200928-subject-07
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20201002-subject-08
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20201015-subject-09
# 13G     /home/frrajic/xode/00-data/dex-january-2025/20201022-subject-10
# 24G     /home/frrajic/xode/00-data/dex-january-2025/bop
# 200K    /home/frrajic/xode/00-data/dex-january-2025/calibration
# 3.5G    /home/frrajic/xode/00-data/dex-january-2025/models
# 154G    total
```

Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de)
and place the file under `manopath`. Unzip the file and create symlink:
```sh
cd /home/frrajic/xode/03-macos/dex-ycb-toolkit

cd manopth
unzip mano_v1_2.zip
cd mano
ln -s ../mano_v1_2/models models
cd ../..
```

Finally, run the script:
```sh
conda activate dexycb
export DEX_YCB_DIR=/home/frrajic/xode/00-data/dex-january-2025
cd /home/frrajic/xode/03-macos/dex-ycb-toolkit
python /home/frrajic/xode/03-macos/spatialtracker/scripts/dex_ycb_to_neus_format.py
```
"""

import os

import cv2
import imageio
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization as vis
import rerun as rr
import torch
import trimesh
import yaml
from dex_ycb_toolkit.layers.mano_group_layer import MANOGroupLayer
from dex_ycb_toolkit.layers.ycb_group_layer import YCBGroupLayer
from dex_ycb_toolkit.layers.ycb_layer import dcm2rv, rv2dcm
from matplotlib import cm
from matplotlib.cm import get_cmap
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRendererWithFragments,
    RasterizationSettings,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm


def sample_surface(mesh: trimesh.Trimesh, count, face_weight=None, seed=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Adapted from:
    https://github.com/mikedh/trimesh/blob/a47b66d2d18404bc044aa9fcb983a80b1287919a/trimesh/sample.py#L23

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    face_weight : None or len(mesh.faces) float
      Weight faces by a factor other than face area.
      If None will be the same as face_weight=mesh.area
    seed : None or int
      If passed as an integer will provide deterministic results
      otherwise pulls the seed from operating system entropy.

    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    colors : (count, 4) float
      Colors of each sampled point
      Returns only when the sample_color is True
    """

    if face_weight is None:
        # len(mesh.faces) float, array of the areas
        # of each face of the mesh
        face_weight = mesh.area_faces

    # cumulative sum of weights (len(mesh.faces))
    # cumulative sum of weights (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)

    # seed the random number generator as requested
    default_rng = np.random.default_rng
    random = default_rng(seed).random

    # last value of cumulative sum is total summed weight/area
    face_pick = random(count) * weight_cum[-1]
    # get the index of the selected faces
    picked_faces = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[picked_faces]
    tri_vectors = tri_vectors[picked_faces]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    picked_weights = random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    outside_triangle = picked_weights.sum(axis=1).reshape(-1) > 1.0
    picked_weights[outside_triangle] -= 1.0
    picked_weights = np.abs(picked_weights)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * picked_weights).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    picked_points = sample_vector + tri_origins

    return picked_faces, picked_weights, picked_points


def pick_points_from_mesh(mesh, picked_faces, picked_weights, reference_mesh):
    if reference_mesh is not None:
        # Number of vertices must match, but the 3D location of vertices can change
        assert reference_mesh.vertices.shape == mesh.vertices.shape, "Number of vertices must match"

        # The faces must be the same
        assert np.allclose(reference_mesh.faces, mesh.faces), "Faces must be the same"

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[picked_faces]
    tri_vectors = tri_vectors[picked_faces]

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * picked_weights).sum(axis=1)

    picked_points = sample_vector + tri_origins

    return picked_points


class SequenceLoader():
    """DexYCB sequence loader."""

    def __init__(
            self,
            name,
            device='cuda:0',
            preload=True,
            app='viewer',
            **kwargs,
    ):
        """Constructor.

        Args:
          name: Sequence name.
          device: A torch.device string argument. The specified device is used only
            for certain data loading computations, but not storing the loaded data.
            Currently the loaded data is always stored as numpy arrays on CPU.
          preload: Whether to preload the point cloud or load it online.
          app: 'viewer' or 'renderer'.
        """
        assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
        assert app in ('viewer', 'renderer', 'convert_to_neus')
        self._name = name
        self._device = torch.device(device)
        self._preload = preload
        self._app = app

        assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
        self._dex_ycb_dir = os.environ['DEX_YCB_DIR']

        # Load meta.
        meta_file = self._dex_ycb_dir + '/' + self._name + "/meta.yml"
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        self._serials = meta['serials']
        self._h = 480
        self._w = 640
        self._num_cameras = len(self._serials)
        self._data_dir = [
            self._dex_ycb_dir + '/' + self._name + '/' + s for s in self._serials
        ]
        self._color_prefix = "color_"
        self._depth_prefix = "aligned_depth_to_color_"
        self._label_prefix = "labels_"
        self._num_frames = meta['num_frames']
        self._ycb_ids = meta['ycb_ids']
        self._mano_sides = meta['mano_sides']

        # Load intrinsics.
        def intr_to_K(x):
            return torch.tensor(
                [[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self._device)

        self._K = []
        for s in self._serials:
            intr_file = self._dex_ycb_dir + "/calibration/intrinsics/" + s + '_' + str(
                self._w) + 'x' + str(self._h) + ".yml"
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            K = intr_to_K(intr['color'])
            self._K.append(K)
        self._K_inv = [torch.inverse(k) for k in self._K]

        # Load extrinsics.
        extr_file = self._dex_ycb_dir + "/calibration/extrinsics_" + meta[
            'extrinsics'] + "/extrinsics.yml"
        with open(extr_file, 'r') as f:
            extr = yaml.load(f, Loader=yaml.FullLoader)
        T = extr['extrinsics']
        T = {
            s: torch.tensor(T[s], dtype=torch.float32,
                            device=self._device).view(3, 4) for s in T
        }
        self._R = [T[s][:, :3] for s in self._serials]
        self._t = [T[s][:, 3] for s in self._serials]
        self._R_inv = [torch.inverse(r) for r in self._R]
        self._t_inv = [torch.mv(r, -t) for r, t in zip(self._R_inv, self._t)]
        self._master_intrinsics = self._K[[
            i for i, s in enumerate(self._serials) if s == extr['master']
        ][0]].cpu().numpy()
        self._tag_R = T['apriltag'][:, :3]
        self._tag_t = T['apriltag'][:, 3]
        self._tag_R_inv = torch.inverse(self._tag_R)
        self._tag_t_inv = torch.mv(self._tag_R_inv, -self._tag_t)
        self._tag_lim = [-0.00, +1.20, -0.10, +0.70, -0.10, +0.70]

        # Compute texture coordinates.
        y, x = torch.meshgrid(torch.arange(self._h), torch.arange(self._w), indexing="ij")
        x = x.float()
        y = y.float()
        s = torch.stack((x / (self._w - 1), y / (self._h - 1)), dim=2)
        self._pcd_tex_coord = [s.numpy()] * self._num_cameras

        # Compute rays.
        self._p = []
        ones = torch.ones((self._h, self._w), dtype=torch.float32)
        xy1s = torch.stack((x, y, ones), dim=2).view(self._w * self._h, 3).t()
        xy1s = xy1s.to(self._device)
        for c in range(self._num_cameras):
            p = torch.mm(self._K_inv[c], xy1s)
            self._p.append(p)

        # Load point cloud.
        if self._preload:
            print('Preloading point cloud')
            self._color = []
            self._depth = []
            for c in range(self._num_cameras):
                color = []
                depth = []
                for i in range(self._num_frames):
                    rgb, d = self._load_frame_rgbd(c, i)
                    color.append(rgb)
                    depth.append(d)
                self._color.append(color)
                self._depth.append(depth)
            self._color = np.array(self._color, dtype=np.uint8)
            self._depth = np.array(self._depth, dtype=np.uint16)
            self._pcd_rgb = [x for x in self._color]
            self._pcd_vert = []
            self._pcd_mask = []
            for c in range(self._num_cameras):
                p, m = self._deproject_depth_and_filter_points(self._depth[c], c)
                self._pcd_vert.append(p)
                self._pcd_mask.append(m)
        else:
            print('Loading point cloud online')
            self._pcd_rgb = [
                np.zeros((self._h, self._w, 3), dtype=np.uint8)
                for _ in range(self._num_cameras)
            ]
            self._pcd_vert = [
                np.zeros((self._h, self._w, 3), dtype=np.float32)
                for _ in range(self._num_cameras)
            ]
            self._pcd_mask = [
                np.zeros((self._h, self._w), dtype=np.bool)
                for _ in range(self._num_cameras)
            ]

        # Create YCB group layer.
        self._ycb_group_layer = YCBGroupLayer(self._ycb_ids).to(self._device)

        self._ycb_model_dir = self._dex_ycb_dir + "/models"
        self._ycb_count = self._ycb_group_layer.count
        self._ycb_material = self._ycb_group_layer.material
        self._ycb_tex_coords = self._ycb_group_layer.tex_coords

        # Create MANO group layer.
        mano_betas = []
        for m in meta['mano_calib']:
            mano_calib_file = self._dex_ycb_dir + "/calibration/mano_" + m + "/mano.yml"
            with open(mano_calib_file, 'r') as f:
                mano_calib = yaml.load(f, Loader=yaml.FullLoader)
            betas = np.array(mano_calib['betas'], dtype=np.float32)
            mano_betas.append(betas)

        self._mano_group_layer = MANOGroupLayer(self._mano_sides,
                                                mano_betas).to(self._device)

        # Prepare data for viewer.
        if app == 'viewer':
            s = np.cumsum([0] + self._ycb_group_layer.count[:-1])
            e = np.cumsum(self._ycb_group_layer.count)
            self._ycb_seg = list(zip(s, e))

            ycb_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
            data = np.load(ycb_file)
            ycb_pose = data['pose_y']
            i = np.any(ycb_pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=2)
            pose = ycb_pose.reshape(-1, 7)
            v, n = self.transform_ycb(pose)
            self._ycb_vert = [
                np.zeros((self._num_frames, n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            self._ycb_norm = [
                np.zeros((self._num_frames, n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            for o in range(self._ycb_group_layer.num_obj):
                io = i[:, o]
                self._ycb_vert[o][io] = v[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
                self._ycb_norm[o][io] = n[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]

            mano_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
            data = np.load(mano_file)
            mano_pose = data['pose_m']
            i = np.any(mano_pose != 0.0, axis=2)
            pose = torch.from_numpy(mano_pose).to(self._device)
            pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
            verts, _ = self._mano_group_layer(pose)
            # Numpy array is faster than PyTorch Tensor here.
            verts = verts.cpu().numpy()
            f = self._mano_group_layer.f.cpu().numpy()
            v = verts[:, f.ravel()]
            n = np.cross(v[:, 1::3, :] - v[:, 0::3, :], v[:, 2::3, :] - v[:, 1::3, :])
            n = np.repeat(n, 3, axis=1)
            l = verts[:, f[:, [0, 1, 1, 2, 2, 0]].ravel(), :]
            self._mano_vert = [
                np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            self._mano_norm = [
                np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            self._mano_line = [
                np.zeros((self._num_frames, 9228, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            for o in range(self._mano_group_layer.num_obj):
                io = i[:, o]
                self._mano_vert[o][io] = v[io, 4614 * o:4614 * (o + 1), :]
                self._mano_norm[o][io] = n[io, 4614 * o:4614 * (o + 1), :]
                self._mano_line[o][io] = l[io, 9228 * o:9228 * (o + 1), :]

        # Prepare data for renderer.
        if app == 'renderer':
            self._ycb_pose = []
            self._mano_vert = []
            self._mano_joint_3d = []

            for c in range(self._num_cameras):
                ycb_pose = []
                mano_pose = []
                mano_joint_3d = []
                for i in range(self._num_frames):
                    label_file = self._data_dir[
                                     c] + '/' + self._label_prefix + "{:06d}.npz".format(i)
                    label = np.load(label_file)
                    pose_y = np.hstack((label['pose_y'],
                                        np.array([[[0, 0, 0, 1]]] * len(label['pose_y']),
                                                 dtype=np.float32)))
                    pose_m = label['pose_m']
                    joint_3d = label['joint_3d']
                    ycb_pose.append(pose_y)
                    mano_pose.append(pose_m)
                    mano_joint_3d.append(joint_3d)
                ycb_pose = np.array(ycb_pose, dtype=np.float32)
                mano_pose = np.array(mano_pose, dtype=np.float32)
                mano_joint_3d = np.array(mano_joint_3d, dtype=np.float32)
                self._ycb_pose.append(ycb_pose)
                self._mano_joint_3d.append(mano_joint_3d)

                i = np.any(mano_pose != 0.0, axis=2)
                pose = torch.from_numpy(mano_pose).to(self._device)
                pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
                verts, _ = self._mano_group_layer(pose)
                verts = verts.cpu().numpy()
                mano_vert = [
                    np.zeros((self._num_frames, 778, 3), dtype=np.float32)
                    for _ in range(self._mano_group_layer.num_obj)
                ]
                for o in range(self._mano_group_layer.num_obj):
                    io = i[:, o]
                    mano_vert[o][io] = verts[io, 778 * o:778 * (o + 1), :]
                self._mano_vert.append(mano_vert)

        # Convert to Neus format.
        if app == "convert_to_neus":
            output_dataset_path = kwargs.get("output_dataset_path", "output_dataset")
            downscaling_factor = kwargs.get("downscaling_factor", 1)
            n_points = kwargs.get("n_points", 3_600)
            n_subsample = kwargs.get("n_subsample", 1)
            seed = kwargs.get("seed", 72)
            stream_rerun_viz = kwargs.get("stream_rerun_viz", False)
            save_rerun_viz = kwargs.get("save_rerun_viz", False)

            np.random.seed(seed)
            torch.manual_seed(seed)

            # Save camera centers as a .ply pointcloud, for debugging purposes.
            t_centered = torch.stack(self._t) - torch.tensor([0., 0., 1.])  # Move along z axis by -1
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, self._num_cameras))[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(t_centered.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd_file = os.path.join(output_dataset_path, f"camera_center__{c:02d}_cameras.ply")
            o3d.io.write_point_cloud(pcd_file, pcd)

            # Create the view folders.
            for c in range(self._num_cameras):
                view_folder = os.path.join(output_dataset_path, f"view_{c:02d}")
                os.makedirs(view_folder, exist_ok=True)

                # Save the intrinsics.txt file.
                intrinsics_file = os.path.join(view_folder, "intrinsics.txt")
                intrinsics = np.zeros((4, 4), dtype=np.float32)
                intrinsics[:3, :3] = self._K[c].cpu().numpy()
                intrinsics[3, 3] = 1
                intrinsics_str = '\n'.join([' '.join([str(x) for x in row]) for row in intrinsics])
                with open(intrinsics_file, "w") as f:
                    f.write(intrinsics_str)

                # Save the cameras_sphere.npz file.
                R = self._R
                t = self._t
                t_centered = [t_ - torch.tensor([0., 0., 1.]) for t_ in t]  # Move along z axis by -1
                R_inv = [torch.inverse(r) for r in R]
                t_centered_inv = [torch.mv(r, -t) for r, t in zip(R_inv, t_centered)]
                extrinsics = np.zeros((4, 4), dtype=np.float32)
                extrinsics[:3, :3] = R_inv[c].cpu().numpy()
                extrinsics[:3, 3] = t_centered_inv[c].cpu().numpy()
                extrinsics[3, 3] = 1
                cameras_sphere_file = os.path.join(view_folder, "cameras_sphere.npz")
                cameras_sphere = {
                    **{f'world_mat_{output_frame_id}': intrinsics @ extrinsics
                       for output_frame_id in range(math.ceil(self._num_frames / n_subsample))},
                    **{f'scale_mat_{output_frame_id}': np.diag(
                        [downscaling_factor, downscaling_factor, downscaling_factor, 1.0])
                        for output_frame_id in range(math.ceil(self._num_frames / n_subsample))}
                }
                np.savez_compressed(cameras_sphere_file, **cameras_sphere)

                # Also, save the intrinsics and extrinsics directly into a .npz file.
                camera_params_path = os.path.join(view_folder, "intrinsics_extrinsics.npz")
                np.savez_compressed(camera_params_path, intrinsics=intrinsics, extrinsics=extrinsics)

                # Save the rgb and depth images. And dummy masks.
                rgb_folder = os.path.join(view_folder, "rgb")
                depth_folder = os.path.join(view_folder, "depth")
                mask_folder = os.path.join(view_folder, "mask")
                rgb_with_valid_depth_folder = os.path.join(view_folder, "rgb_with_valid_depth")
                os.makedirs(rgb_folder, exist_ok=True)
                os.makedirs(depth_folder, exist_ok=True)
                os.makedirs(mask_folder, exist_ok=True)
                os.makedirs(rgb_with_valid_depth_folder, exist_ok=True)
                for output_frame_id in range(math.ceil(self._num_frames / n_subsample)):
                    input_frame_id = output_frame_id * n_subsample
                    rgb = self._color[c][input_frame_id][:, :, ::-1]
                    rgb_file = os.path.join(rgb_folder, f"{output_frame_id:05d}.png")
                    cv2.imwrite(rgb_file, rgb)

                    depth = self._depth[c][input_frame_id]
                    depth_file = os.path.join(depth_folder, f"{output_frame_id:05d}.png")
                    cv2.imwrite(depth_file, depth)

                    rgb_plot = rgb.copy()
                    rgb_plot[depth == 0] = 255
                    cv2.imwrite(os.path.join(rgb_with_valid_depth_folder, f"{output_frame_id:05d}.png"), rgb_plot)

                    label_file = self._data_dir[c] + '/' + self._label_prefix + "{:06d}.npz".format(input_frame_id)
                    label = np.load(label_file)
                    seg_mask = label["seg"]
                    mask = seg_mask != 0  # Everything that is not background
                    mask = mask[:, :, None].astype(np.uint8).repeat(3, 2) * 255
                    # dummy_mask = np.ones((self._h, self._w, 3)).astype(np.uint8) * 255
                    # mask = dummy_mask
                    mask_file = os.path.join(mask_folder, f"{output_frame_id:05d}.png")
                    imageio.imwrite(mask_file, mask)

                    # Backproject the depth image to 3D points for visualization purposes.
                    if output_frame_id in [0, math.ceil(self._num_frames / n_subsample) - 1] and c in range(
                            self._num_cameras):
                        d = self._depth[c][input_frame_id]
                        d = d.astype(np.float32) / 1000
                        d = torch.from_numpy(d).to(self._device)

                        p = torch.mul(
                            d.view(1, -1, self._w * self._h).expand(3, -1, -1),
                            self._p[c].unsqueeze(1))
                        p = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p.view(3, -1))
                        p = p.t().view(self._h, self._w, 3)
                        p = p.cpu().numpy()

                        m = d > 0
                        p = p[m]
                        colors = self._color[c][input_frame_id][m] / 255

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(p)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        pcd_file = os.path.join(view_folder, f"pcd_for_t{output_frame_id:03d}.ply")
                        o3d.io.write_point_cloud(pcd_file, pcd)

            # Compute meshes for each frame.
            s = np.cumsum([0] + self._ycb_group_layer.count[:-1])
            e = np.cumsum(self._ycb_group_layer.count)
            self._ycb_seg = list(zip(s, e))

            ycb_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
            data = np.load(ycb_file)
            ycb_pose = data['pose_y'][::n_subsample]
            i = np.any(ycb_pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=2)
            pose = ycb_pose.reshape(-1, 7)
            v, n = self.transform_ycb(pose)
            self._ycb_vert = [
                np.zeros((math.ceil(self._num_frames / n_subsample), n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            self._ycb_norm = [
                np.zeros((math.ceil(self._num_frames / n_subsample), n, 3), dtype=np.float32)
                for n in self._ycb_count
            ]
            for o in range(self._ycb_group_layer.num_obj):
                io = i[:, o]
                self._ycb_vert[o][io] = v[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
                self._ycb_norm[o][io] = n[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
            self._ycb_faces = [
                np.arange(n).reshape(-1, 3)
                for n in self._ycb_count
            ]

            mano_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
            data = np.load(mano_file)
            mano_pose = data['pose_m'][::n_subsample]
            i = np.any(mano_pose != 0.0, axis=2)
            pose = torch.from_numpy(mano_pose).to(self._device)
            pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
            verts, _ = self._mano_group_layer(pose)
            # Numpy array is faster than PyTorch Tensor here.
            verts = verts.cpu().numpy()
            f = self._mano_group_layer.f.cpu().numpy()
            v = verts[:, f.ravel()]
            n = np.cross(v[:, 1::3, :] - v[:, 0::3, :], v[:, 2::3, :] - v[:, 1::3, :])
            n = np.repeat(n, 3, axis=1)
            l = verts[:, f[:, [0, 1, 1, 2, 2, 0]].ravel(), :]
            self._mano_vert = [
                np.zeros((math.ceil(self._num_frames / n_subsample), 4614, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            self._mano_norm = [
                np.zeros((math.ceil(self._num_frames / n_subsample), 4614, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            self._mano_line = [
                np.zeros((math.ceil(self._num_frames / n_subsample), 9228, 3), dtype=np.float32)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            self._mano_faces = [
                np.arange(4614).reshape(-1, 3)
                for _ in range(self._mano_group_layer.num_obj)
            ]
            for o in range(self._mano_group_layer.num_obj):
                io = i[:, o]
                self._mano_vert[o][io] = v[io, 4614 * o:4614 * (o + 1), :]
                self._mano_norm[o][io] = n[io, 4614 * o:4614 * (o + 1), :]
                self._mano_line[o][io] = l[io, 9228 * o:9228 * (o + 1), :]

            vert = []
            vert += self._ycb_vert
            vert += self._mano_vert

            norm = []
            norm += self._ycb_norm
            norm += self._mano_norm

            ids = []
            ids += self._ycb_group_layer._ids
            ids += [255 for _ in self._mano_group_layer._sides]

            names = []
            names += ["ycb-" + layer._class_name for layer in self._ycb_group_layer._layers]
            names += [f"mano-{side}-hand" for side in self._mano_group_layer._sides]

            faces = []
            faces += self._ycb_faces
            faces += self._mano_faces

            print(f"Number of meshes: {len(vert)}")
            assert len(vert) == len(norm) == len(faces) == len(ids) == len(names)

            print(f"Mesh names: {names}")
            print(f"Mesh IDS: {ids}")

            all_vertices = np.concatenate(vert, axis=1)
            all_normals = np.concatenate(norm, axis=1)
            all_faces = np.concatenate([
                f + np.sum([v.shape[1] for v in vert[:i]]).astype(np.uint32)
                for i, f in enumerate(faces)
            ])
            all_ids = np.concatenate([np.full(v.shape[1], i) for i, v in enumerate(vert)])
            assert all_vertices.shape[0] == all_normals.shape[0]
            assert all_vertices.shape[1] == all_normals.shape[1] == all_faces.shape[0] * 3 == all_ids.shape[0]
            assert all_faces.max() + 1 == all_vertices.shape[1]
            print(f"all_vertices.shape: {all_vertices.shape}")
            print(f"all_normals.shape: {all_normals.shape}")
            print(f"all_faces.shape: {all_faces.shape}")
            print(f"all_ids.shape: {all_ids.shape}")

            n_frames = all_vertices.shape[0]
            meshes = [
                trimesh.Trimesh(
                    vertices=all_vertices[frame_idx],
                    faces=all_faces,
                    vertex_normals=all_normals[frame_idx],
                    process=False,
                )
                for frame_idx in range(n_frames)
            ]

            # Put the query points onto the frame where the hand is first visible
            hands_visible = np.any(mano_pose != 0.0, axis=2).all(axis=1)
            assert np.any(hands_visible), "Hands must be visible in at least one frame"
            t0 = np.argmax(hands_visible, axis=0)

            objects_visible = np.any(ycb_pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=2).all(axis=1)
            assert objects_visible[t0], "Objects must be visible in the first frame where the hands are visible"

            picked_faces, picked_weights, picked_points = sample_surface(meshes[t0], n_points, seed=seed)
            assert np.allclose(picked_points,
                               pick_points_from_mesh(meshes[t0], picked_faces, picked_weights, meshes[t0]))
            picked_vertices = meshes[t0].faces[:, 0][picked_faces]
            picked_ids = all_ids[picked_vertices]

            # Track the points
            tracks_3d = []
            for frame_idx in range(n_frames):
                points = pick_points_from_mesh(meshes[frame_idx], picked_faces, picked_weights, meshes[t0])
                tracks_3d.append(points)
                if frame_idx == t0:
                    assert np.allclose(points, picked_points)
            tracks_3d = np.stack(tracks_3d)  # (n_frames, n_points, 3)

            # Project the points to the camera
            tracks_2d = []
            tracks_2d_z = []
            for c in range(self._num_cameras):
                p = torch.from_numpy(tracks_3d).to(self._device).T.reshape(3, -1)
                p = self._R_inv[c].double() @ p + self._t_inv[c][:, None]
                p = self._K[c].double() @ p
                z = p[2]
                p = p[:2] / z

                p = p.cpu().numpy().reshape(2, n_points, math.ceil(self._num_frames / n_subsample)).T
                z = z.cpu().numpy().reshape(n_points, math.ceil(self._num_frames / n_subsample)).T

                tracks_2d.append(p)
                tracks_2d_z.append(z)

            tracks_2d = np.stack(tracks_2d)
            tracks_2d_z = np.stack(tracks_2d_z)

            # --- Estimate occlusion
            rendered_depth = []
            for c in range(self._num_cameras):
                rendered_depth_camera = []
                for frame_idx in range(n_frames):
                    rgb = self._color[c][0]
                    depth = (self._depth[c][0] / 1000).clip(0, 2)
                    h, w = self._h, self._w
                    K = self._K[c].cpu().numpy()
                    w2c = np.eye(4, dtype=float)
                    w2c[:3, :3] = self._R_inv[c].cpu().numpy()
                    w2c[:3, 3] = self._t_inv[c].cpu().numpy()
                    c2w = np.linalg.inv(w2c)

                    # Render depth
                    device = "cuda"
                    vertices = torch.tensor(all_vertices[frame_idx], dtype=torch.float32).to(device)
                    faces = torch.tensor(all_faces, dtype=torch.int64).to(device)
                    vertex_colors = torch.ones_like(vertices).unsqueeze(0).to(device)
                    textures = TexturesVertex(verts_features=vertex_colors)
                    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
                    intrinsics = torch.eye(4, dtype=torch.float32).to(device)
                    intrinsics[:3, :3] = torch.from_numpy(K)
                    cameras = cameras_from_opencv_projection(
                        R=torch.from_numpy(w2c[:3, :3]).to(device)[None].float(),
                        tvec=torch.from_numpy(w2c[:3, 3]).to(device)[None].float(),
                        camera_matrix=self._K[c].to(device)[None].float(),
                        image_size=torch.tensor([self._h, self._w], dtype=torch.int32).to(device)[None].float(),
                    )
                    raster_settings = RasterizationSettings(
                        image_size=(self._h, self._w),
                        blur_radius=0.0,
                        faces_per_pixel=1,
                        bin_size=0,
                    )
                    renderer = MeshRendererWithFragments(
                        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                        shader=SoftPhongShader(device=device, cameras=cameras, lights=PointLights(device=device)),
                    )
                    images, fragments = renderer(mesh)
                    depth_map = fragments.zbuf
                    rendered_depth_camera.append(depth_map.cpu().numpy()[0, :, :, 0])
                rendered_depth.append(rendered_depth_camera)
            rendered_depth = np.stack(rendered_depth)
            assert rendered_depth.shape == (self._num_cameras, n_frames, self._h, self._w)

            seg_masks = []
            for c in range(self._num_cameras):
                seg_masks_camera = []
                for frame_idx in range(n_frames):
                    input_frame_id = frame_idx * n_subsample
                    label_file = self._data_dir[c] + '/' + self._label_prefix + "{:06d}.npz".format(input_frame_id)
                    label = np.load(label_file)
                    seg_masks_camera.append(label["seg"])
                seg_masks.append(seg_masks_camera)
            seg_masks = np.stack(seg_masks)
            assert seg_masks.shape == (self._num_cameras, n_frames, self._h, self._w)

            seg_unique = np.unique(seg_masks)
            cmap = get_cmap("tab10")
            seg_masks_rgb = np.zeros((*seg_masks.shape, 3), dtype=np.uint8)
            for idx, val in enumerate(seg_unique):
                seg_masks_rgb[seg_masks == val] = (np.array(cmap(idx / len(seg_unique))[:3]) * 255).astype(np.uint8)
            assert seg_masks_rgb.shape == (self._num_cameras, n_frames, self._h, self._w, 3)

            def estimate_occlusion_by_depth_and_segment(
                    depth_map,
                    x,
                    y,
                    num_frames,
                    thresh,
                    seg_id=None,
                    segments=None,
                    min_or_max_reduce="max",
                    convert_to_pixel_coords=True,
                    occlude_if_depth_larger_than_xxx=None,
            ):
                # need to convert from raster to pixel coordinates
                if convert_to_pixel_coords:
                    x = x - 0.5
                    y = y - 0.5

                x0 = np.floor(x).astype(np.int32)
                x1 = x0 + 1
                y0 = np.floor(y).astype(np.int32)
                y1 = y0 + 1

                shp = depth_map.shape
                assert len(depth_map.shape) == 3
                x0 = np.clip(x0, 0, shp[2] - 1)
                x1 = np.clip(x1, 0, shp[2] - 1)
                y0 = np.clip(y0, 0, shp[1] - 1)
                y1 = np.clip(y1, 0, shp[1] - 1)

                depth_map = depth_map.reshape(-1)
                rng = np.arange(num_frames)[:, np.newaxis]
                assert x.shape[0] == y.shape[0] == num_frames
                i1 = np.take(depth_map, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
                i2 = np.take(depth_map, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
                i3 = np.take(depth_map, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
                i4 = np.take(depth_map, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

                if min_or_max_reduce == "max":
                    depth = np.maximum(np.maximum(np.maximum(i1, i2), i3), i4)
                elif min_or_max_reduce == "min":
                    depth = np.minimum(np.minimum(np.minimum(i1, i2), i3), i4)
                else:
                    raise ValueError(f"Unknown min_or_max_reduce: {min_or_max_reduce}")
                if occlude_if_depth_larger_than_xxx is not None:
                    depth[depth >= occlude_if_depth_larger_than_xxx] = 0
                depth_occluded = depth < thresh
                print("┌ Depth occlusion: ", depth_occluded.sum(), "/", depth_occluded.size)

                occluded = depth_occluded
                if segments is not None:
                    segments = segments.reshape(-1)
                    i1 = np.take(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
                    i2 = np.take(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
                    i3 = np.take(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
                    i4 = np.take(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x1)
                    seg_occluded = np.ones_like(depth_occluded, dtype=bool)
                    for i in [i1, i2, i3, i4]:
                        i = i.astype(np.int32)
                        seg_occluded = np.logical_and(seg_occluded, seg_id != i)
                    print("| Segmentation occlusion: ", seg_occluded.sum(), "/", seg_occluded.size)
                    occluded = np.logical_or(occluded, seg_occluded)

                return occluded

            tracks_2d_visibilities = []
            for c in range(self._num_cameras):
                occlusion = np.zeros((tracks_2d[c].shape[0], tracks_2d[c].shape[1]), dtype=bool)
                print(f"N occluded: {occlusion.sum()} / {occlusion.size}")
                occlusion = np.logical_or(occlusion, (tracks_2d_z[c] <= 0) | (tracks_2d_z[c] >= (65535 / 1000)))
                print(f"N occluded (after Z): {occlusion.sum()} / {occlusion.size}")
                occlusion = np.logical_or(occlusion, tracks_2d[c][:, :, 0] <= 0)
                occlusion = np.logical_or(occlusion, tracks_2d[c][:, :, 1] <= 0)
                occlusion = np.logical_or(occlusion, tracks_2d[c][:, :, 0] >= self._w - 1)
                occlusion = np.logical_or(occlusion, tracks_2d[c][:, :, 1] >= self._h - 1)
                print(f"N occluded (& out-of-frame): {occlusion.sum()} / {occlusion.size}")

                # # V1: Use the depth map to estimate occlusion
                # depth_map_for_occlusion = self._depth[c][::n_subsample].copy()
                # depth_map_for_occlusion[depth_map_for_occlusion == 0] = 65535
                # depth_map_for_occlusion = depth_map_for_occlusion / 1000.0

                # # V2: Make the depth for occlussion be the depth from projected predicted points, taking the minimum z over all points at a pixel
                # depth_map_for_occlusion = np.ones((tracks_2d_z.shape[1], self._h, self._w),
                #                                   dtype=np.float32) * 65535 / 1000
                # for frame_idx in range(math.ceil(self._num_frames / n_subsample)):
                #     for point_idx in range(n_points):
                #         if np.isnan(tracks_2d[c][frame_idx, point_idx]).any():
                #             continue
                #         x = int(tracks_2d[c][frame_idx, point_idx, 0])
                #         y = int(tracks_2d[c][frame_idx, point_idx, 1])
                #         z = tracks_2d_z[c][frame_idx, point_idx]
                #         if 0 <= x < self._w and 0 <= y < self._h:
                #             depth_map_for_occlusion[frame_idx, y - 3:y + 3, x - 3:x + 3] = np.minimum(
                #                 depth_map_for_occlusion[frame_idx, y - 3:y + 3, x - 3:x + 3],
                #                 z,
                #             )
                # # Visualize it side by side with GT depth
                # if False:
                #     for frame_idx in range(math.ceil(self._num_frames / n_subsample)):
                #         if frame_idx not in [0, math.ceil(self._num_frames / n_subsample) - 1]:
                #             continue
                #         d1 = self._depth[c][frame_idx * n_subsample] / 1000
                #         d2 = depth_map_for_occlusion[frame_idx]
                #         d12 = np.concatenate([d1, d2], axis=1)
                #         plt.figure(dpi=150, figsize=(d12.shape[1] / 100, d12.shape[0] / 100))
                #         plt.title(f"Depth GT (left) vs Depth used for occlusion (right), frame {frame_idx}")
                #         plt.imshow(d12.clip(0.5, 1))
                #         plt.axis('off')
                #         plt.tight_layout(pad=0)
                #         plt.savefig(os.path.join(output_dataset_path,
                #                                  f"depth_used_for_occlussion_view_{c:02d}_frame_{frame_idx:05d}.png"))
                #         # plt.show()
                #
                # seg_mask = []
                # for output_frame_id in range(math.ceil(self._num_frames / n_subsample)):
                #     input_frame_id = output_frame_id * n_subsample
                #     label_file = self._data_dir[c] + '/' + self._label_prefix + "{:06d}.npz".format(input_frame_id)
                #     label = np.load(label_file)
                #     seg_mask.append(label["seg"])
                # seg_mask = np.stack(seg_mask)
                # depth_or_segment_occluded = estimate_occlusion_by_depth_and_segment(
                #     depth_map=depth_map_for_occlusion,
                #     segments=seg_mask,
                #     x=tracks_2d[c][:, :, 0],
                #     y=tracks_2d[c][:, :, 1],
                #     num_frames=tracks_2d[c].shape[0],
                #     thresh=tracks_2d_z[c] * 0.995,
                #     seg_id=np.array(ids)[picked_ids],
                # )
                # occlusion = np.logical_or(occlusion, depth_or_segment_occluded)
                # print(f"N occluded (& obscured by other objects): {occlusion.sum()} / {occlusion.size}")
                # print()
                # tracks_2d_visibilities.append(~occlusion)

                # # V3.a: Neither the GT depth nor the segmentation mask are reliable for occlusion estimation.
                # #       Instead, we will use the rendered depth map, with a little help from the GT depth.
                # #       First, the rendered depth needs to match the point depth, if not, the point is occluded.
                # #       This will work perfectly for all the objects that have a full 3D mesh over time. So all
                # #       the objects on the table, plus the MANO hand (but not the arm). This is susceptible to
                # #       errors in estimating the mesh location, but it should be less problematic than the other
                # #       segmentation mask and GT depth in that it will be less noisy and more consistent over time.
                # rendered_depth_for_occlusion = rendered_depth[c].copy()
                # rendered_depth_for_occlusion[rendered_depth_for_occlusion <= 0] = 65535 / 1000
                # depth_or_segment_occluded = estimate_occlusion_by_depth_and_segment(
                #     depth_map=rendered_depth_for_occlusion,
                #     x=tracks_2d[c, :, :, 0],
                #     y=tracks_2d[c, :, :, 1],
                #     num_frames=n_frames,
                #     thresh=tracks_2d_z[c, :, :] - 0.01,
                #     min_or_max_reduce="min",
                #     convert_to_pixel_coords=False,
                #     occlude_if_depth_larger_than_xxx=65535 / 1000,
                # )
                # occlusion = np.logical_or(occlusion, depth_or_segment_occluded)
                # print(f"N occluded (& obscured in rendered depth): {occlusion.sum()} / {occlusion.size}")
                # print()
                #
                # # # V3.b: Second, to avoid occlusion by the arm, we will use the GT depth map but with a high threshold.
                # # #       This will avoid the arm occluding the points as the arm is not in the rendered depth map.
                # # depth_or_segment_occluded = estimate_occlusion_by_depth_and_segment(
                # #     depth_map=self._depth[c][::n_subsample] / 1000,
                # #     x=tracks_2d[c, :, :, 0],
                # #     y=tracks_2d[c, :, :, 1],
                # #     num_frames=n_frames,
                # #     thresh=tracks_2d_z[c, :, :] * 0.995,
                # # )
                # # occlusion = np.logical_or(occlusion, depth_or_segment_occluded)
                # # print(f"N occluded (& obscured in GT depth): {occlusion.sum()} / {occlusion.size}")
                # # print()

                # V4.a: Forget the rendered depths, it's still difficult because the depth map is pixelized.
                #       Instead, let's shoot rays from the camera onto the scene mesh and see where they intersect.
                #       It is very very slow but most accurate.
                camera_center = self._t[c].cpu().numpy()
                for frame_idx in range(n_frames):
                    for track_idx in tqdm(range(n_points), desc=f"Ray casting for camera {c} frame {frame_idx}"):
                        if occlusion[frame_idx, track_idx]:
                            continue
                        ray_direction = tracks_3d[frame_idx, track_idx] - camera_center
                        ray_direction /= np.linalg.norm(ray_direction)
                        intersections = meshes[frame_idx].ray.intersects_location(camera_center[None],
                                                                                  ray_direction[None])
                        if len(intersections[0]) == 0:
                            occlusion[frame_idx, track_idx] = True
                            continue
                        intersection_depth = np.inf
                        for intersection in intersections[0]:
                            intersection_depth = min(intersection_depth, np.linalg.norm(intersection - camera_center))
                        track_depth = np.linalg.norm(tracks_3d[frame_idx, track_idx] - camera_center)
                        occlusion[frame_idx, track_idx] = not np.isclose(intersection_depth, track_depth, atol=0.001)
                print(f"N occluded (& obscured in scene mesh): {occlusion.sum()} / {occlusion.size}")
                print()

                # V4.b: The arm is not in the scene mesh and it is causing problems for 1/2 cameras. Let's use the
                #       GT depths to figure out if the arm is occluding the points. Unfortunately, this will also not
                #       work perfectly because the GT depths are missing around silhouette edges.
                depth_map_for_occlusion = self._depth[c][::n_subsample].copy()
                depth_map_for_occlusion[depth_map_for_occlusion <= 0] = 65535
                depth_map_for_occlusion = depth_map_for_occlusion / 1000.0
                depth_or_segment_occluded = estimate_occlusion_by_depth_and_segment(
                    depth_map=depth_map_for_occlusion,
                    x=tracks_2d[c, :, :, 0],
                    y=tracks_2d[c, :, :, 1],
                    num_frames=n_frames,
                    thresh=tracks_2d_z[c, :, :] - 0.12,
                    min_or_max_reduce="min",
                    convert_to_pixel_coords=False,
                )
                occlusion = np.logical_or(occlusion, depth_or_segment_occluded)
                print(f"N occluded (& obscured in GT depth): {occlusion.sum()} / {occlusion.size}")
                print()

                # Idea for V5: Do V4 and additionally try looking at if the RGB changed a lot for the point.
                #              If it did, then it is likely that the point is occluded by the arm/person.
                #              However, this might suffer from the same problem as in V3: edges would be noisy
                #              and might quickly jump from visible to occluded to visible again.
                ...

                tracks_2d_visibilities.append(~occlusion)

            tracks_2d_visibilities = np.stack(tracks_2d_visibilities)
            tracks_3d_visibilities = tracks_2d_visibilities.any(axis=0)

            if stream_rerun_viz or save_rerun_viz:
                assert not (stream_rerun_viz and save_rerun_viz), ("Stream and save rerun at the same time not "
                                                                   "supported. But you can save what was streamed "
                                                                   "within the rerun viewer. Or run again. Or impl it.")

                rr.init("dexycb_preprocessing", recording_id="v0.1")
                if stream_rerun_viz:
                    rr.connect_tcp()
                rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
                rr.set_time_seconds("frame", 0)
                rr.log(
                    "world/xyz",
                    rr.Arrows3D(
                        vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ),
                )
                entity_prefix = f"{os.path.basename(output_dataset_path)}/"
                radii_scale = 0.1
                for t in range(n_frames):
                    t_input = t * n_subsample
                    rr.set_time_seconds("frame", t / 12)
                    rr.log(f"{entity_prefix}mesh", rr.Mesh3D(
                        vertex_positions=np.asarray(meshes[frame_idx].vertices),
                        triangle_indices=np.asarray(meshes[frame_idx].faces),
                    ))
                    for c in range(self._num_cameras):
                        rgb = self._color[c, t_input]
                        depth = (self._depth[c, t_input] / 1000).clip(0, 2)
                        rend_depth = rendered_depth[c, t].clip(0, 2)
                        seg_mask = seg_masks_rgb[c, t]
                        seg_rgb = seg_masks_rgb[c, t]
                        h, w = self._h, self._w
                        K = self._K[c].cpu().numpy()
                        K_inv = np.linalg.inv(K)
                        w2c = np.eye(4, dtype=float)
                        w2c[:3, :3] = self._R_inv[c].cpu().numpy()
                        w2c[:3, 3] = self._t_inv[c].cpu().numpy()
                        c2w = np.linalg.inv(w2c)

                        cam_pinhole = rr.Pinhole(image_from_camera=K, width=w, height=h)
                        cam_transform = rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3])
                        for name, archetype in [
                            ("rgb", rr.Image(rgb)),
                            ("seg", rr.Image(seg_rgb)),
                            ("depth-gt", rr.DepthImage(depth, point_fill_ratio=0.2)),
                            ("depth-rendered", rr.DepthImage(rend_depth, point_fill_ratio=0.2)),
                        ]:
                            rr.log(f"{entity_prefix}/image/{name}/view-{c:02d}", cam_pinhole)
                            rr.log(f"{entity_prefix}/image/{name}/view-{c:02d}", cam_transform)
                            rr.log(f"{entity_prefix}/image/{name}/view-{c:02d}/{name}", archetype)

                        # Compute 3D points from GT depth map
                        y, x = np.indices((self._h, self._w))
                        homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                        cam_coords = (K_inv @ homo_pixel_coords) * depth.ravel()
                        cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                        world_coords = (c2w @ cam_coords)[:3].T
                        valid_mask = depth.ravel() > 0
                        rr.log(f"{entity_prefix}point_cloud/rgb-gt/view-{c}", rr.Points3D(
                            positions=world_coords[valid_mask],
                            colors=rgb.reshape(-1, 3)[valid_mask].astype(np.uint8),
                            radii=0.01 * radii_scale,
                        ))
                        rr.log(f"{entity_prefix}point_cloud/seg-gt/view-{c}", rr.Points3D(
                            positions=world_coords[valid_mask],
                            colors=seg_rgb.reshape(-1, 3)[valid_mask].astype(np.uint8),
                            radii=0.01 * radii_scale,
                        ))

                        # Compute 3D points from GT depth map
                        y, x = np.indices((self._h, self._w))
                        homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                        cam_coords = (K_inv @ homo_pixel_coords) * rend_depth.ravel()
                        cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                        world_coords = (c2w @ cam_coords)[:3].T
                        valid_mask = rend_depth.ravel() > 0
                        rr.log(f"{entity_prefix}point_cloud/rgb-rend/view-{c}", rr.Points3D(
                            positions=world_coords[valid_mask],
                            colors=rgb.reshape(-1, 3)[valid_mask].astype(np.uint8),
                            radii=0.01 * radii_scale,
                        ))
                        rr.log(f"{entity_prefix}point_cloud/seg-rend/view-{c}", rr.Points3D(
                            positions=world_coords[valid_mask],
                            colors=seg_rgb.reshape(-1, 3)[valid_mask].astype(np.uint8),
                            radii=0.01 * radii_scale,
                        ))

                def log_tracks(
                        tracks: np.ndarray,
                        visibles: np.ndarray,
                        query_timestep: np.ndarray,
                        colors: np.ndarray,

                        entity_format_str="{}",

                        log_points=True,
                        points_radii=0.03 * radii_scale,
                        invisible_color=[0., 0., 0.],

                        log_line_strips=True,
                        max_strip_length_past=12,
                        max_strip_length_future=1,
                        strips_radii=0.0042 * radii_scale,

                        log_error_lines=False,
                        error_lines_radii=0.0072 * radii_scale,
                        error_lines_color=[1., 0., 0.],
                        gt_for_error_lines=None,
                ) -> None:
                    """
                    Log tracks to Rerun.

                    Parameters:
                        tracks: Shape (T, N, 3), the 3D trajectories of points.
                        visibles: Shape (T, N), boolean visibility mask for each point at each timestep.
                        query_timestep: Shape (T, N), the frame index after which the tracks start.
                        colors: Shape (N, 4), RGBA colors for each point.
                        entity_prefix: String prefix for entity hierarchy in Rerun.
                        entity_suffix: String suffix for entity hierarchy in Rerun.
                    """

                    T, N, _ = tracks.shape
                    assert tracks.shape == (T, N, 3)
                    assert visibles.shape == (T, N)
                    assert query_timestep.shape == (N,)
                    assert query_timestep.min() >= 0
                    assert query_timestep.max() < T
                    assert colors.shape == (N, 4)

                    for n in range(N):
                        rr.log(entity_format_str.format(f"track-{n}"), rr.Clear(recursive=True))
                        for t in range(query_timestep[n], T):
                            rr.set_time_seconds("frame", t / 12)

                            # Log the point (special handling for invisible points)
                            if log_points:
                                rr.log(
                                    entity_format_str.format(f"track-{n}/point"),
                                    rr.Points3D(
                                        positions=[tracks[t, n]],
                                        colors=[colors[n, :3]] if visibles[t, n] else [invisible_color],
                                        radii=points_radii,
                                    ),
                                )

                            # Log line segments for visible tracks
                            if log_line_strips and t > query_timestep[n]:
                                strip_t_start = max(t - max_strip_length_past, query_timestep[n].item())
                                strip_t_end = min(t + max_strip_length_future, T - 1)

                                strips = np.stack([
                                    tracks[strip_t_start:strip_t_end, n],
                                    tracks[strip_t_start + 1:strip_t_end + 1, n],
                                ], axis=-2)
                                strips_visibility = visibles[strip_t_start + 1:strip_t_end + 1, n]
                                strips_colors = np.where(
                                    strips_visibility[:, None],
                                    colors[None, n, :3],
                                    [invisible_color],
                                )

                                rr.log(
                                    entity_format_str.format(f"track-{n}/line"),
                                    rr.LineStrips3D(strips=strips, colors=strips_colors, radii=strips_radii),
                                )

                            if log_error_lines:
                                assert gt_for_error_lines is not None
                                strips = np.stack([
                                    tracks[t, n],
                                    gt_for_error_lines[t, n],
                                ], axis=-2)
                                rr.log(
                                    entity_format_str.format(f"track-{n}/error"),
                                    rr.LineStrips3D(strips=strips, colors=error_lines_color, radii=error_lines_radii),
                                )

                # Log the tracks
                cmap = matplotlib.colormaps["gist_rainbow"]
                norm = matplotlib.colors.Normalize(vmin=tracks_3d[..., 0].min(), vmax=tracks_3d[..., 0].max())
                track_color = cmap(norm(tracks_3d[-1, :, 0]))
                track_color = track_color * 0 + 1  # Just make all tracks white

                N = 800
                B = 200
                for tracks_batch_start in range(0, N, B):
                    tracks_batch_end = min(tracks_batch_start + B, N)
                    for name, visibles in [
                        ("tracks/c01234567-visibility",
                         tracks_2d_visibilities.any(0)[:, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c0123-visibility",
                         tracks_2d_visibilities.any(0)[:, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c2345-visibility",
                         tracks_2d_visibilities.any(0)[:, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c0-visibility", tracks_2d_visibilities[0, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c1-visibility", tracks_2d_visibilities[1, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c2-visibility", tracks_2d_visibilities[2, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c3-visibility", tracks_2d_visibilities[3, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c4-visibility", tracks_2d_visibilities[4, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c5-visibility", tracks_2d_visibilities[5, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c6-visibility", tracks_2d_visibilities[6, :, tracks_batch_start:tracks_batch_end]),
                        ("tracks/c7-visibility", tracks_2d_visibilities[7, :, tracks_batch_start:tracks_batch_end]),
                    ]:
                        log_tracks(
                            tracks=tracks_3d[:, tracks_batch_start:tracks_batch_end],
                            visibles=visibles,
                            query_timestep=visibles.argmax(axis=0),
                            colors=track_color[tracks_batch_start:tracks_batch_end],
                            entity_format_str=f"{entity_prefix}/{name}/{tracks_batch_start:02d}-{tracks_batch_end:02d}/{{}}",
                            max_strip_length_future=0,
                        )

                if save_rerun_viz:
                    rr_rrd_path = os.path.join(output_dataset_path, f"rerun_viz.rrd")
                    rr.save(rr_rrd_path)
                    print(f"Saved Rerun recording to: {os.path.abspath(rr_rrd_path)}")

            # import pydevd_pycharm
            # pydevd_pycharm.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)

            # Save the tracks
            tracks_3d_file = os.path.join(output_dataset_path, "tracks_3d.npz")
            np.savez(
                tracks_3d_file,
                tracks_3d=(tracks_3d - np.array([0., 0., 1.])) / DOWNSCALING_FACTOR,
                tracks_3d_visibilities=tracks_3d_visibilities,
                object_ids=np.array(ids)[picked_ids],
                object_id_to_name={i: name for i, name in zip(ids, names)},
                tracks_2d=tracks_2d,
                tracks_2d_z=tracks_2d_z,
                tracks_2d_visibilities=tracks_2d_visibilities,
            )

            # Save some .ply files of the trajectories for debugging
            colors = plt.cm.viridis(tracks_3d[t0, :, 2] / tracks_3d[t0, :, 2].max())[:, :3]
            for frame_idx in [0, t0, t0 + 1, n_frames // 3, (2 * n_frames) // 3, n_frames - 1]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(tracks_3d[frame_idx])
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd_file = os.path.join(output_dataset_path, f"tracks_3d_{frame_idx}.ply")
                o3d.io.write_point_cloud(pcd_file, pcd)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(tracks_3d[frame_idx][tracks_3d_visibilities[frame_idx]])
                pcd.colors = o3d.utility.Vector3dVector(colors[tracks_3d_visibilities[frame_idx]])
                pcd_file = os.path.join(output_dataset_path, f"tracks_3d_{frame_idx}_visible.ply")
                o3d.io.write_point_cloud(pcd_file, pcd)

            # Also save the first frame trimesh as a mesh
            meshes[0].export(os.path.join(output_dataset_path, "first_frame_mesh.obj"))

        self._frame = -1

    def _load_frame_rgbd(self, c, i):
        """Loads an RGB-D frame.

        Args:
          c: Camera index.
          i: Frame index.

        Returns:
          color: A unit8 numpy array of shape [H, W, 3] containing the color image.
          depth: A uint16 numpy array of shape [H, W] containing the depth image.
        """
        color_file = self._data_dir[
                         c] + '/' + self._color_prefix + "{:06d}.jpg".format(i)
        color = cv2.imread(color_file)
        color = color[:, :, ::-1]
        depth_file = self._data_dir[
                         c] + '/' + self._depth_prefix + "{:06d}.png".format(i)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return color, depth

    def _deproject_depth_and_filter_points(self, d, c):
        """Deprojects a depth image to point cloud and filters points.

        Args:
          d: A uint16 numpy array of shape [F, H, W] or [H, W] containing the depth
            image in millimeters.
          c: Camera index.

        Returns:
          p: A float32 numpy array of shape [F, H, W, 3] or [H, W, 3] containing the
            point cloud.
          m: A bool numpy array of shape [F, H, W] or [H, W] containing the mask for
            points within the tag cooridnate limit.
        """
        nd = d.ndim
        d = d.astype(np.float32) / 1000
        d = torch.from_numpy(d).to(self._device)
        p = torch.mul(
            d.view(1, -1, self._w * self._h).expand(3, -1, -1),
            self._p[c].unsqueeze(1))
        p = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p.view(3, -1))
        p_tag = torch.addmm(self._tag_t_inv.unsqueeze(1), self._tag_R_inv, p)
        mx1 = p_tag[0, :] > self._tag_lim[0]
        mx2 = p_tag[0, :] < self._tag_lim[1]
        my1 = p_tag[1, :] > self._tag_lim[2]
        my2 = p_tag[1, :] < self._tag_lim[3]
        mz1 = p_tag[2, :] > self._tag_lim[4]
        mz2 = p_tag[2, :] < self._tag_lim[5]
        m = mx1 & mx2 & my1 & my2 & mz1 & mz2
        p = p.t().view(-1, self._h, self._w, 3)
        m = m.view(-1, self._h, self._w)
        if nd == 2:
            p = p.squeeze(0)
            m = m.squeeze(0)
        p = p.cpu().numpy()
        m = m.cpu().numpy()
        return p, m

    def transform_ycb(self,
                      pose,
                      c=None,
                      camera_to_world=True,
                      run_ycb_group_layer=True,
                      return_trans_mat=False):
        """Transforms poses in SE3 between world and camera frames.

        Args:
          pose: A float32 numpy array of shape [N, 7] or [N, 6] containing the
            poses. Each row contains one pose represented by rotation in quaternion
            (x, y, z, w) or rotation vector and translation.
          c: Camera index.
          camera_to_world: Whether from camera to world or from world to camera.
          run_ycb_group_layer: Whether to return vertices and normals by running the
            YCB group layer or to return poses.
          return_trans_mat: Whether to return poses in transformation matrices.

        Returns:
          If run_ycb_group_layer is True:
            v: A float32 numpy array of shape [F, V, 3] containing the vertices.
            n: A float32 numpy array of shape [F, V, 3] containing the normals.
          else:
            A float32 numpy array of shape [N, 6] containing the transformed poses.
        """
        if pose.shape[1] == 7:
            q = pose[:, :4]
            t = pose[:, 4:]
            R = Rot.from_quat(q).as_matrix().astype(np.float32)
            R = torch.from_numpy(R).to(self._device)
            t = torch.from_numpy(t).to(self._device)
        if pose.shape[1] == 6:
            r = pose[:, :3]
            t = pose[:, 3:]
            r = torch.from_numpy(r).to(self._device)
            t = torch.from_numpy(t).to(self._device)
            R = rv2dcm(r)
        if c is not None:
            if camera_to_world:
                R_c = self._R[c]
                t_c = self._t[c]
            else:
                R_c = self._R_inv[c]
                t_c = self._t_inv[c]
            R = torch.bmm(R_c.expand(R.size(0), -1, -1), R)
            t = torch.addmm(t_c, t, R_c.t())
        if run_ycb_group_layer or not return_trans_mat:
            r = dcm2rv(R)
            p = torch.cat([r, t], dim=1)
        else:
            p = torch.cat([R, t.unsqueeze(2)], dim=2)
            p = torch.cat([
                p,
                torch.tensor([[[0, 0, 0, 1]]] * R.size(0),
                             dtype=torch.float32,
                             device=self._device)
            ],
                dim=1)
        if run_ycb_group_layer:
            p = p.view(-1, self._ycb_group_layer.num_obj * 6)
            v, n = self._ycb_group_layer(p)
            v = v[:, self._ycb_group_layer.f.view(-1)]
            n = n[:, self._ycb_group_layer.f.view(-1)]
            v = v.cpu().numpy()
            n = n.cpu().numpy()
            return v, n
        else:
            p = p.cpu().numpy()
            return p

    @property
    def serials(self):
        return self._serials

    @property
    def num_cameras(self):
        return self._num_cameras

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def dimensions(self):
        return self._w, self._h

    @property
    def ycb_ids(self):
        return self._ycb_ids

    @property
    def K(self):
        return self._K

    @property
    def master_intrinsics(self):
        return self._master_intrinsics

    def step(self):
        """Steps the frame."""
        self._frame = (self._frame + 1) % self._num_frames
        if not self._preload:
            self._update_pcd()

    def _update_pcd(self):
        """Updates the point cloud."""
        for c in range(self._num_cameras):
            rgb, d = self._load_frame_rgbd(c, self._frame)
            p, m = self._deproject_depth_and_filter_points(d, c)
            self._pcd_rgb[c][:] = rgb
            self._pcd_vert[c][:] = p
            self._pcd_mask[c][:] = m

    @property
    def pcd_rgb(self):
        if self._preload:
            return [x[self._frame] for x in self._pcd_rgb]
        else:
            return self._pcd_rgb

    @property
    def pcd_vert(self):
        if self._preload:
            return [x[self._frame] for x in self._pcd_vert]
        else:
            return self._pcd_vert

    @property
    def pcd_tex_coord(self):
        return self._pcd_tex_coord

    @property
    def pcd_mask(self):
        if self._preload:
            return [x[self._frame] for x in self._pcd_mask]
        else:
            return self._pcd_mask

    @property
    def ycb_group_layer(self):
        return self._ycb_group_layer

    @property
    def num_ycb(self):
        return self._ycb_group_layer.num_obj

    @property
    def ycb_model_dir(self):
        return self._ycb_model_dir

    @property
    def ycb_count(self):
        return self._ycb_count

    @property
    def ycb_material(self):
        return self._ycb_material

    @property
    def ycb_pose(self):
        if self._app == 'viewer':
            return None
        if self._app == 'renderer':
            return [x[self._frame] for x in self._ycb_pose]

    @property
    def ycb_vert(self):
        if self._app == 'viewer':
            return [x[self._frame] for x in self._ycb_vert]
        if self._app == 'renderer':
            return None

    @property
    def ycb_norm(self):
        if self._app == 'viewer':
            return [x[self._frame] for x in self._ycb_norm]
        if self._app == 'renderer':
            return None

    @property
    def ycb_tex_coords(self):
        return self._ycb_tex_coords

    @property
    def mano_group_layer(self):
        return self._mano_group_layer

    @property
    def num_mano(self):
        return self._mano_group_layer.num_obj

    @property
    def mano_vert(self):
        if self._app == 'viewer':
            return [x[self._frame] for x in self._mano_vert]
        if self._app == 'renderer':
            return [[y[self._frame] for y in x] for x in self._mano_vert]

    @property
    def mano_norm(self):
        if self._app == 'viewer':
            return [x[self._frame] for x in self._mano_norm]
        if self._app == 'renderer':
            return None

    @property
    def mano_line(self):
        if self._app == 'viewer':
            return [x[self._frame] for x in self._mano_line]
        if self._app == 'renderer':
            return None

    @property
    def mano_joint_3d(self):
        if self._app == 'viewer':
            return None
        if self._app == 'renderer':
            return [x[self._frame] for x in self._mano_joint_3d]


# Some hacking with global variables to make the visualization work
first_frame_seen = False
ready_to_close = False


def visualize_3dpt_tracks(tracks_path, output_video_path):
    global first_frame_seen, ready_to_close
    print(f"Visualizing 3D point tracks from {tracks_path} to {output_video_path}...")

    tracks = np.load(tracks_path)["tracks_3d"] + np.array([0, 0, 1])
    n_frames, n_points, _ = tracks.shape

    frames_path = f"{output_video_path}__frames"
    os.makedirs(frames_path, exist_ok=True)
    first_frame_seen = False
    ready_to_close = False

    # images = [imageio.imread(f"{frames_path}/frame_{i:04d}.png") for i in range(n_frames)]
    # video_writer = imageio.get_writer(output_video_path, fps=10)
    # for img in images:
    #     video_writer.append_data(img)
    # video_writer.close()
    # ready_to_close = True
    # return

    z = tracks[2 * n_frames // 3, :, 2]

    point_colors = np.zeros((n_points, 3))
    point_colors[:, 0] = np.sin(z)
    point_colors[:, 1] = np.sin(z + 2 * np.pi / 3)
    point_colors[:, 2] = np.sin(z + 4 * np.pi / 3)

    point_colors = cm.jet(z / np.percentile(z, 99.9))[:, :3]

    print("Preparing clouds...")
    pointclouds = []
    for frame_idx in tqdm(range(n_frames)):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(tracks[frame_idx])
        pc.colors = o3d.utility.Vector3dVector(point_colors)
        pointclouds += [{
            "name": f"cloud t={frame_idx}",
            "geometry": pc,
            "time": frame_idx / 4,
        }]

    def start_animation(w: o3d.cpu.pybind.visualization.O3DVisualizer) -> None:
        w.is_animating = True

    frames_path = f"{output_video_path}__frames"
    os.makedirs(frames_path, exist_ok=True)
    first_frame_seen = False
    ready_to_close = False

    def create_video(w: o3d.cpu.pybind.visualization.O3DVisualizer, t: float) -> None:
        global first_frame_seen, ready_to_close
        if ready_to_close:
            print("Please close the window to finish the video export.")
            return
        if t == 0 and not first_frame_seen:
            first_frame_seen = True
        elif t == 0 and first_frame_seen:
            images = [imageio.imread(f"{frames_path}/frame_{i:04d}.png") for i in range(n_frames)]
            video_writer = imageio.get_writer(output_video_path, fps=10)
            for img in images:
                video_writer.append_data(img)
            video_writer.close()
            ready_to_close = True
            return
        w.export_current_image(f"{frames_path}/frame_{int(t * 4):04d}.png")

    vis.draw(
        title=tracks_path,
        width=1920,
        height=1080,
        point_size=4,
        geometry=pointclouds,
        animation_time_step=1 / 4,
        # ibl="crossroads",
        eye=np.array([0, 0, 0]),
        lookat=np.array([0, 0, 1]),
        up=np.array([0, -1, 0]),
        field_of_view=60.0,
        on_init=start_animation,
        on_animation_frame=create_video,
        on_animation_tick=None,
    )


DOWNSCALING_FACTOR = 1.0
SEQUENCES = [
    # Each sequence has a different target object and a different (human) subject performing an action.
    "20200709-subject-01/20200709_141754",
    "20200813-subject-02/20200813_145653",
    "20200820-subject-03/20200820_135841",
    "20200903-subject-04/20200903_104428",
    "20200908-subject-05/20200908_144409",
    "20200918-subject-06/20200918_114117",
    "20200928-subject-07/20200928_144906",
    "20201002-subject-08/20201002_110227",
    "20201015-subject-09/20201015_144721",
    "20201022-subject-10/20201022_112651",
]


def main():
    assert os.environ['DEX_YCB_DIR']
    for n_subsample in [3]:
        for sequence in tqdm(SEQUENCES):
            print(f"Processing sequence: {sequence}")
            SequenceLoader(
                sequence,
                device="cpu",
                preload=True,
                app="convert_to_neus",
                output_dataset_path=os.path.join(os.environ['DEX_YCB_DIR'],
                                                 f"neus_nsubsample-{n_subsample}/{sequence.replace('/', '__')}"),
                downscaling_factor=DOWNSCALING_FACTOR,
                n_subsample=n_subsample,
                seed=72,
                stream_rerun_viz=False,
                save_rerun_viz=True,
            )
    print("Done converting the dataset.")


if __name__ == '__main__':
    main()
