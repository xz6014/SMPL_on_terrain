import time
import torch
import trimesh
from lib.smpl_layer import SMPL_Layer

import os

bm_norm = SMPL_Layer(gender='male', hands=False, normalise=True)
import numpy as np
from copy import deepcopy
import pickle as pkl
from scenic_dataset.utils import process_heights
import argparse
from lib.meshviewer import Mesh, MeshViewer, colors, Floor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", default=3, type=int)
    parser.add_argument("--npy_path", default="SCENIC_dataset/pkl/transition_jumping_003_184.pkl", type=str)
    args = parser.parse_args()

    mv = MeshViewer(offscreen=False)
    save_scene_path = "scenes"
    os.makedirs(save_scene_path, exist_ok=True)

    name = args.npy_path.split('/')[-1].split('.pkl')[0]
    gender, trans, betas, pose = pkl.load(open(args.npy_path, 'rb'))[:4]
    assert 60 >= len(pose) >= 20

    trans = torch.FloatTensor(trans)
    betas = torch.FloatTensor(betas).flatten()[:10][None].repeat(trans.shape[0], 1)
    pose = torch.FloatTensor(pose)
    if pose.shape[1] != 72:
        pose = torch.FloatTensor(pose.reshape(-1, 52, 3)[:, list(np.arange(0, 22)) + [22, 37]]).reshape(-1, 72)

    v, J = bm_norm(pose=pose, trans=trans, betas=betas, scale=1)[:2]
    pose = pose[:, :66]
    ground = deepcopy(v[:, :, 1].min())
    J[:, :, 1] -= ground
    trans[:, 1] -= ground
    v[:, :, 1] -= ground

    terrain_meshes = process_heights(J.numpy(), name=name, nsamples=args.n_samples)

    for t, terrain_mesh in enumerate(terrain_meshes):
        for i in range(0, v.shape[0]):
            meshes = []
            meshes += [Mesh(vertices=v[i], faces=bm_norm.faces, smooth=True, vc=colors['lightblue'])]
            if terrain_mesh:
                meshes += [terrain_mesh]
            else:
                meshes += [Floor(scale=(5, 5), y_up=True, vc=colors['brown'])]
            mv.set_static_meshes(meshes)
            time.sleep(0.1)
