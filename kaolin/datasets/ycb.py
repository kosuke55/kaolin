# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ipdb

import sys
import os
from pathlib import Path
import torch
import torch.utils.data as data
import warnings
import urllib.request
import zipfile
import json
import re
from collections import OrderedDict
from glob import glob
import numpy as np
import random

from tqdm import tqdm
import scipy.sparse
import tarfile
from PIL import Image

import kaolin as kal
from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.rep.QuadMesh import QuadMesh

from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import meshfunc
from kaolin.transforms import voxelfunc
from kaolin.transforms import transforms as tfs
from kaolin import helpers
import kaolin.conversions.meshconversions as mesh_cvt

from .base import KaolinDataset

# Synset to Label mapping (for ycb core classes)
synset_to_label = {
    '019_pitcher_base': '019_pitcher_base',
    '025_mug': '025_mug',
    '035_power_drill': '035_power_drill',
    '048_hammer': '048_hammer',
    '051_large_clamp': '051_large_clamp',
    '022_windex_bottle': '022_windex_bottle',
    '033_spatula': '033_spatula',
    '042_adjustable_wrench': '042_adjustable_wrench',
    '050_medium_clamp': '050_medium_clamp',
    '052_extra_large_clamp': '052_extra_large_clamp'
}

    # '04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
    #                '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
    #                '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
    #                '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
    #                '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
    #                '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
    #                '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
    #                '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
    #                '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    #                '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
    #                '04554684': 'washer', '04090263': 'rifle', '02946921': 'can',
    #                '03797390': 'mug'}

# Label to Synset mapping (for ycb core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}


class print_wrapper(object):
    def __init__(self, text, logger=sys.stdout.write):
        self.text = text
        self.logger = logger

    def __enter__(self):
        self.logger(self.text)

    def __exit__(self, *args):
        self.logger("\t[done]\n")


def _convert_categories(categories):
    assert categories is not None, 'List of categories cannot be empty!'
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            ycbCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets


class ycb_Meshes(data.Dataset):
    r"""ycb Dataset class for meshes.

    Args:
        root (str): Path to the root directory of the ycb dataset.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): If True, return the training set, otherwise the test set
        split (float): fraction of the dataset to be used for training (>=0 and <=1)
        no_progress (bool): if True, disables progress bar
    Returns:
        .. code-block::

           dict: {
               attributes: {name: str, path: str, synset: str, label: str},
               data: {vertices: torch.Tensor, faces: torch.Tensor}
           }
    Example:
        >>> meshes = ycb_Meshes(root='../data/ycb/')
        >>> obj = next(iter(meshes))
        >>> obj['data']['vertices'].shape
        torch.Size([2133, 3])
        >>> obj['data']['faces'].shape
        torch.Size([1910, 3])
    """

    def __init__(self, root: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, no_progress: bool = False):
        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError('Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target)))

            
            # ipdb.set_trace()
            # find all objects in the class
            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            # self.paths += models
            # self.synset_idxs += [i] * len(models)
            
            self.paths += [class_target]
            self.synset_idxs += [i] * len([class_target])

        self.names = [p.name for p in self.paths]

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.paths)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        synset_idx = self.synset_idxs[index]
        # ipdb.set_trace()
        obj_location = self.paths[index] / 'textured.obj'
        mesh = TriangleMesh.from_obj(str(obj_location))

        data['vertices'] = mesh.vertices
        data['faces'] = mesh.faces
        attributes['name'] = self.names[index]
        attributes['path'] = obj_location
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ycb(KaolinDataset):
    r"""ycbV1 Dataset class for meshes.

    Args:
        root (str): path to ycb root directory
        categories (list): List of categories to load from ycb. This list may
                           contain synset ids, class label names (for ycbCore classes),
                           or a combination of both.
        train (bool): If True, return the training set, otherwise the test set
        split (float): fraction of the dataset to be used for training (>=0 and <=1)
    Returns:
        .. code-block::

           dict: {
                attributes: {name: str, path: str, synset: str, label: str},
                data: {vertices: torch.Tensor, faces: torch.Tensor}
           }

    Example:
        >>> meshes = ycb(root='../data/ycb/')
        >>> obj = meshes[0]
        >>> obj['data'].vertices.shape
        torch.Size([2133, 3])
        >>> obj['data'].faces.shape
        torch.Size([1910, 3])
    """

    def initialize(self, root: str, categories: list, train: bool = True, split: float = .7):
        """Initialize the dataset.

        Args:
            root (str): path to ycb root directory
            categories (list): List of categories to load from ycb. This list may
                               contain synset ids, class label names (for ycbCore classes),
                               or a combination of both.
            train (bool): If True, return the training set, otherwise the test set
            split (float): fraction of the dataset to be used for training (>=0 and <=1)"""
        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError('Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target)))

            # find all objects in the class
            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            # self.paths += models
            self.paths += class_target
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.paths)

    def _get_data(self, index):
        synset_idx = self.synset_idxs[index]
        obj_location = self.paths[index] / 'textured.obj'
        mesh = TriangleMesh.from_obj(str(obj_location))
        return mesh

    def _get_attributes(self, index):
        synset_idx = self.synset_idxs[index]
        attributes = {
            'name': self.names[index],
            'path': self.paths[index] / 'textured.obj',
            'synset': self.synsets[synset_idx],
            'label': self.labels[synset_idx]
        }
        return attributes


class ycb_Images(data.Dataset):
    r"""ycb Dataset class for images.

    Arguments:
        root (str): Path to the root directory of the ycb dataset.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): if true use the training set, else use the test set
        split (float): amount of dataset that is training out of
        views (int): number of viewpoints per object to load
        transform (torchvision.transforms) : transformation to apply to images
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, path: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
            params: {
                cam_mat: torch.Tensor,
                cam_pos: torch.Tensor,
                azi: float,
                elevation: float,
                distance: float
            }
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> images = ycb_Images(root='../data/ycbImages')
        >>> train_loader = DataLoader(images, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> image = obj['data']['imgs']
        >>> image.shape
        torch.Size([10, 4, 137, 137])
    """

    def __init__(self, root: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, views: int = 24, transform=None):
        self.root = Path(root)
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.transform = transform
        self.views = views
        self.names = []
        self.synset_idx = []

        # check if images exist
        if not self.root.exists():
            raise ValueError('ycb images were not found at location {0}.'.format(
                str(self.root)))

        # find all needed images
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            assert class_target.exists(), \
                "ycb class, {0}, is not found".format(syn)

            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.names += models

            self.synset_idx += [i] * len(models)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        img_name = self.names[index]
        view_num = random.randrange(0, self.views)
        # load and process image
        img = Image.open(str(img_name / f'rendering/{view_num:02}.png'))
        # apply transformations
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(np.array(img))
            img = img.permute(2, 1, 0)
            img = img / 255.
        # load and process camera parameters
        param_location = img_name / 'rendering/rendering_metadata.txt'
        azimuth, elevation, _, distance, _ = np.loadtxt(param_location)[view_num]
        cam_params = kal.mathutils.geometry.transformations.compute_camera_params(
            azimuth, elevation, distance)

        data['images'] = img
        data['params'] = dict()
        data['params']['cam_mat'] = cam_params[0]
        data['params']['cam_pos'] = cam_params[1]
        data['params']['azi'] = azimuth
        data['params']['elevation'] = elevation
        data['params']['distance'] = distance
        attributes['name'] = img_name
        attributes['synset'] = self.synsets[self.synset_idx[index]]
        attributes['label'] = self.labels[self.synset_idx[index]]
        return {'data': data, 'attributes': attributes}


class ycb_Voxels(data.Dataset):
    r"""ycb Dataset class for voxels.

    Args:
        root (str): Path to the root directory of the ycb dataset.
        cache_dir (str): Path to save cached converted representations.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        resolutions (list): list of resolutions to be returned
        no_progress (bool): if True, disables progress bar
        voxel_range (float): Range of voxelization.

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {[res]: torch.Tensor}
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> voxels = ycb_Voxels(root='../data/ycb/', cache_dir='cache/')
        >>> train_loader = DataLoader(voxels, batch_size=10, shuffle=True, num_workers=8 )
        >>> obj = next(iter(train_loader))
        >>> obj['data']['128'].shape
        torch.Size([10, 128, 128, 128])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, resolutions=[128, 32], no_progress: bool = False,
                 voxel_range: float = 1.0):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'voxels'
        self.cache_transforms = {}
        self.params = {
            'resolutions': resolutions,
        }
        mesh_dataset = ycb_Meshes(root=root,
                                       categories=categories,
                                       train=train,
                                       split=split,
                                       no_progress=no_progress)
        self.names = mesh_dataset.names
        self.synset_idxs = mesh_dataset.synset_idxs
        self.synsets = mesh_dataset.synsets
        self.labels = mesh_dataset.labels

        for res in self.params['resolutions']:
            self.cache_transforms[res] = tfs.CacheCompose([
                tfs.TriangleMeshToVoxelGrid(res,
                                            normalize=True,
                                            vertex_offset=0.5,
                                            voxel_range=voxel_range),
                tfs.FillVoxelGrid(thresh=0.5),
                tfs.ExtractProjectOdmsFromVoxelGrid()
            ], self.cache_dir)

            desc = 'converting to voxels'
            for idx in tqdm(range(len(mesh_dataset)), desc=desc, disable=no_progress):
                name = mesh_dataset.names[idx]
                if name not in self.cache_transforms[res].cached_ids:
                    sample = mesh_dataset[idx]
                    mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                     sample['data']['faces'])
                    self.cache_transforms[res](name, mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        for res in self.params['resolutions']:
            data[str(res)] = self.cache_transforms[res](name)
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ycb_Surface_Meshes(data.Dataset):
    r"""ycb Dataset class for watertight meshes with only the surface preserved.

    Arguments:
        root (str): Path to the root directory of the ycb dataset.
        cache_dir (str): Path to save cached converted representations.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        resolution (int): resolution of voxel object to use when converting
        smoothing_iteration (int): number of applications of laplacian smoothing
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
        }

    Example:
        >>> surface_meshes = ycb_Surface_Meshes(root='../data/ycb', cache_dir='cache/')
        >>> obj = next(iter(surface_meshes))
        >>> obj['data']['vertices'].shape
        torch.Size([11617, 3])
        >>> obj['data']['faces'].shape
        torch.Size([23246, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, resolution: int = 100, smoothing_iterations: int = 3, mode='Tri',
                 no_progress: bool = False):
        assert mode in ['Tri', 'Quad']

        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'surface_meshes'
        dataset_params = {
            'root': root,
            'categories': categories,
            'train': train,
            'split': split,
            'no_progress': no_progress,
        }
        self.params = {
            'resolution': resolution,
            'smoothing_iterations': smoothing_iterations,
            'mode': mode,
        }

        mesh_dataset = ycb_Meshes(**dataset_params)
        voxel_dataset = ycb_Voxels(**dataset_params, cache_dir=cache_dir, resolutions=[resolution])
        combined_dataset = ycb_Combination([mesh_dataset, voxel_dataset])

        self.names = combined_dataset.names
        self.synset_idxs = combined_dataset.synset_idxs
        self.synsets = combined_dataset.synsets
        self.labels = combined_dataset.labels

        if mode == 'Tri':
            mesh_conversion = tfs.VoxelGridToTriangleMesh(threshold=0.5,
                                                          mode='marching_cubes',
                                                          normalize=False)
        else:
            mesh_conversion = tfs.VoxelGridToQuadMesh(threshold=0.5,
                                                      normalize=False)

        def convert(og_mesh, voxel):
            transforms = tfs.Compose([mesh_conversion,
                                      tfs.MeshLaplacianSmoothing(smoothing_iterations)])

            new_mesh = transforms(voxel)
            new_mesh.vertices = pcfunc.realign(new_mesh.vertices, og_mesh.vertices)
            return {'vertices': new_mesh.vertices, 'faces': new_mesh.faces}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to surface meshes'
        for idx in tqdm(range(len(combined_dataset)), desc=desc, disable=no_progress):
            name = combined_dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                sample = combined_dataset[idx]
                voxel = sample['data'][str(resolution)]
                og_mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                    sample['data']['faces'])
                self.cache_convert(name, og_mesh=og_mesh, voxel=voxel)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        data = self.cache_convert(name)
        mesh = TriangleMesh.from_tensors(data['vertices'], data['faces'])
        data['adj'] = mesh.compute_adjacency_matrix_sparse().coalesce()
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ycb_Points(data.Dataset):
    r"""ycb Dataset class for sampled point cloud from each object.

    Args:
        root (str): Path to the root directory of the ycb dataset.
        cache_dir (str): Path to save cached converted representations.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        num_points (int): number of point sampled on mesh
        smoothing_iteration (int): number of application of laplacian smoothing
        surface (bool): if only the surface of the original mesh should be used
        resolution (int): resolution of voxel object to use when converting
        normals (bool): should the normals of the points be saved
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {points: torch.Tensor, normals: torch.Tensor}
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> points = ycb_Points(root='../data/ycb', cache_dir='cache/')
        >>> train_loader = DataLoader(points, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['points'].shape
        torch.Size([10, 5000, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, num_points: int = 5000, smoothing_iterations=3,
                 surface=True, resolution=100, normals=True, no_progress: bool = False):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'points'

        dataset_params = {
            'root': root,
            'categories': categories,
            'train': train,
            'split': split,
            'no_progress': no_progress,
        }
        self.params = {
            'num_points': num_points,
            'smoothing_iterations': smoothing_iterations,
            'surface': surface,
            'resolution': resolution,
            'normals': normals,
        }

        if surface:
            dataset = ycb_Surface_Meshes(**dataset_params,
                                              cache_dir=cache_dir,
                                              resolution=resolution,
                                              smoothing_iterations=smoothing_iterations)
        else:
            dataset = ycb_Meshes(**dataset_params)

        self.names = dataset.names
        self.synset_idxs = dataset.synset_idxs
        self.synsets = dataset.synsets
        self.labels = dataset.labels

        def convert(mesh):
            points, face_choices = mesh_cvt.trianglemesh_to_pointcloud(mesh, num_points)
            face_normals = mesh.compute_face_normals()
            point_normals = face_normals[face_choices]
            return {'points': points, 'normals': point_normals}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to points'
        for idx in tqdm(range(len(dataset)), desc=desc, disable=no_progress):
            name = dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                idx = dataset.names.index(name)
                sample = dataset[idx]
                mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                 sample['data']['faces'])
                self.cache_convert(name, mesh=mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        data = self.cache_convert(name)
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ycb_SDF_Points(data.Dataset):
    r"""ycb Dataset class for signed distance functions.

    Args:
        root (str): Path to the root directory of the ycb dataset.
        cache_dir (str): Path to save cached converted representations.
        categories (str): List of categories to load from ycb. This list may
                contain synset ids, class label names (for ycbCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        resolution (int): resolution of voxel object to use when converting
        num_points (int): number of sdf points sampled on mesh
        occ (bool): should only occupancy values be returned instead of distances
        smoothing_iteration (int): number of application of laplacian smoothing
        sample_box (bool): whether to sample only from within mesh extents
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

            dict: {
                attributes: {name: str, synset: str, label: str},
                data: {
                    Union['occ_values', 'sdf_distances']: torch.Tensor,
                    Union['occ_points, 'sdf_points']: torch.Tensor}
            }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> sdf_points = ycb_SDF_Points(root='../data/ycb', cache_dir='cache/')
        >>> train_loader = DataLoader(sdf_points, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['sdf_points'].shape
        torch.Size([10, 5000, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, resolution: int = 100, num_points: int = 5000, occ: bool = False,
                 smoothing_iterations: int = 3, sample_box=True, no_progress: bool = False):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'sdf_points'

        self.params = {
            'resolution': resolution,
            'num_points': num_points,
            'occ': occ,
            'smoothing_iterations': smoothing_iterations,
            'sample_box': sample_box,
        }

        surface_mesh_dataset = ycb_Surface_Meshes(root=root,
                                                       cache_dir=cache_dir,
                                                       categories=categories,
                                                       train=train,
                                                       split=split,
                                                       resolution=resolution,
                                                       smoothing_iterations=smoothing_iterations,
                                                       no_progress=no_progress)

        self.names = surface_mesh_dataset.names
        self.synset_idxs = surface_mesh_dataset.synset_idxs
        self.synsets = surface_mesh_dataset.synsets
        self.labels = surface_mesh_dataset.labels

        def convert(mesh):
            sdf = mesh_cvt.trianglemesh_to_sdf(mesh, num_points)
            bbox_true = torch.stack((mesh.vertices.min(dim=0)[0],
                                     mesh.vertices.max(dim=0)[0]), dim=1).view(-1)
            points = 1.05 * (torch.rand(self.params['num_points'], 3).to(mesh.vertices.device) - .5)
            distances = sdf(points)
            return {'points': points, 'distances': distances, 'bbox': bbox_true}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to sdf points'
        for idx in tqdm(range(len(surface_mesh_dataset)), desc=desc, disable=no_progress):
            name = surface_mesh_dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                idx = surface_mesh_dataset.names.index(name)
                sample = surface_mesh_dataset[idx]
                mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                 sample['data']['faces'])

                # Use cuda if available to speed up conversion
                if torch.cuda.is_available():
                    mesh.cuda()
                self.cache_convert(name, mesh=mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        cached_data = self.cache_convert(name)
        points = cached_data['points']
        distances = cached_data['distances']

        if self.params['sample_box']:
            bbox_values = kal.rep.bounding_points(points, cached_data['bbox'])
            points = points[bbox_values]
            distances = distances[bbox_values]

        selection = np.random.randint(points.shape[0], size=self.params['num_points'])

        if self.params['occ']:
            data['occ_values'] = distances[selection] <= 0
            data['occ_points'] = points[selection]
        else:
            data['sdf_distances'] = distances[selection]
            data['sdf_points'] = points[selection]

        attributes['name'] = self.names[index]
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}
