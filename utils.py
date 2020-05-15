import zipfile
import time
import json
from PIL import Image
import numpy as np
import glob
import os
import tqdm
import random
from scipy.spatial.transform import Rotation as R
import socket

# def get_cam_int_np_3x3(path, camera_name):
#     d = json.load(open(path, 'r'))[camera_name]
#     return np.array([
#         [d['fx'], 0, d['cx']],
#         [0, d['fy'], d['cy']],
#         [0, 0, 1]
#     ], np.float32)

class nightLocalPointCloud(object):

    def __init__(self, zip_path):
        self.idx = int(os.path.basename(zip_path).replace('.zip', ''))
        self.archive = zipfile.ZipFile(zip_path, 'r')
        self.pc_files, self.mat_files = {}, {}
        for file in self.archive.namelist():
            if file.endswith('.txt'):
                spl = os.path.basename(file).replace('.txt', '').split('_')
                if 'T' in file:
                    self.mat_files[int(spl[2])] = file
                else:
                    self.pc_files[int(spl[-1])] = file
        self.k = sorted(list(self.pc_files.keys()))
        assert(self.k == sorted(list(self.mat_files.keys())))
        self.num = len(self.k)
        self.iter = 0
        self.sample = []
        return

    def get_next_transformed_local_pc(self):
        if self.iter == self.num:
            return None
        else:
            idx = self.k[self.iter]
            pc_file = self.pc_files[idx]
            mat_file = self.mat_files[idx]
            self.iter += 1
            pc_str_lines = [line.strip() for line in self.archive.read(pc_file).decode('utf-8').split('\n') if line]
            mat_str_lines = [line.strip() for line in self.archive.read(mat_file).decode('utf-8').split('\n') if line]
            pc = np.array([[float(item) for item in line.split()[:3]] + [1.0] for line in pc_str_lines])
            mat = np.array([[float(item) for item in line.split()] for line in mat_str_lines[:3]])
        return mat.dot(pc.T).T

def create_autovision_simple_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    _PALETTE = np.array(
        [[90, 120, 150], # Barrier
        [70, 70, 70], # Building
        [0, 0, 142], # Car
        [152, 251, 152], # Terrain
        [0, 60, 100], # Heavy Vehicle
        [119, 11, 32], # Motorcycle
        [128, 64, 128], # Paved Road
        [170, 170, 170], # Pedestrian Area
        [220, 20, 60], # Person
        [250, 170, 30], # Pole Object
        [70, 130, 180], # Sky
        [220, 180, 50], # Unpaved Road
        [107, 142, 35], # Vegetation
        [0, 170, 30], # Water
        [255, 255, 255]], # Ignored Object
    dtype=np.uint8)
    colormap[:15,:] = _PALETTE
    return colormap#.flatten()

def get_cam_int_np_3x3(path, camera_name, s):
    cam_json = json.load(open(path, 'r'))
    mat = np.eye(3)
    for d in cam_json['NCameraSystem']['cameras']:
        prop = d['ptr_wrapper']['data']['Properties']
        model = d['ptr_wrapper']['data']['CameraModel']
        if prop['Name'] == camera_name:
            mat[0, 0] = model['fu']
            mat[1, 1] = model['fv']
            mat[0, 2] = model['cu']
            mat[1, 2] = model['cv']
            img_size = (int(prop['ImageWidth']/s), int(prop['ImageHeight']/s))
            xi = model['xi']
            return mat / s, img_size, xi
    return None, None, None

def get_cam_ext_np_3x4(pose):
    qw, qx, qy, qz, tx, ty, tz = pose
    rot = R.from_quat([qx, qy, qz, qw])
    t = np.array([[tx], [ty], [tz]])
    return np.hstack([rot.as_matrix(), t])

def get_cam_ext_np_4x4(pose):
    mat = get_cam_ext_np_3x4(pose)
    return np.vstack([mat, np.array([0, 0, 0, 1])])

def get_cam_poses_nx7(path):
    camera_poses = [line.strip() for line in open(path).readlines()]
    return np.array([[float(item) for item in line.split()] for line in camera_poses[1:]])[:, 1: 8]

def get_pc_nxstr(path, show_time=True):
    basename = os.path.basename(path).replace('_sample.zip', '.txt').replace('.zip', '.txt')
    tic = time.time()
    archive = zipfile.ZipFile(path, 'r')
    pc_data = archive.read(basename).decode('utf-8')
    toc = time.time()
    if show_time:
        print('Loading point cloud costs %.3lf seconds.' % (toc - tic))
    return [line for line in pc_data.split('\n') if line]

def to_float_li(pc_str_line):
    line = [float(item) for item in pc_str_line.split()]
    return line[:3] + [1] + line[3:]

def pc_str_line2XYZ1RGB(pc_str_line):
    return np.array(to_float_li(pc_str_line))

def pc_str_lines2nxXYZ1_and_RGB(pc_str_lines, show_time=True):
    tic = time.time()
    res = np.array([to_float_li(pc_str_line) for pc_str_line in pc_str_lines])
    toc = time.time()
    if show_time:
        print('Creating point cloud numpy array costs %.3lf seconds.' % (toc - tic))
    return res[:, :4], res[:, 4:].astype(np.uint8)

def pc_str_lines2nxXYZ1(pc_str_lines, show_time=True):
    tic = time.time()
    res = np.array([to_float_li(pc_str_line)[:4] for pc_str_line in pc_str_lines])
    toc = time.time()
    if show_time:
        print('Creating point cloud numpy array costs %.3lf seconds.' % (toc - tic))
    return res[:, :4]

def get_normalized_points(num, dim, abs_axis=None):
    u = np.random.normal(size=(num, dim))
    u_len = np.sqrt(np.sum(u * u, axis=1))
    u = (u.T / u_len).T
    if abs_axis is not None:
        u[:, abs_axis] = np.abs(u[:, abs_axis])
    return u

def verify_depth(img_idx, pc_depth, gt_depth, epsilon=0.01):
    assert(img_idx.shape[0] == pc_depth.shape[0])
    assert(not (pc_depth <= 0).any())
    assert(img_idx.max() < gt_depth.shape[0])
    d = {}
    for i, (pixel, depth) in enumerate(zip(img_idx, pc_depth)):
        if pixel in d:
            d[pixel].append((depth, i))
        else:
            d[pixel] = [(depth, i)]

    res = {}
    for pixel in d:
        min_d, _ = min(d[pixel])
        max_d, _ = max(d[pixel])
        gt = gt_depth[pixel]
        if min_d * (1 - epsilon) < gt and gt < max_d * (1 + epsilon):
            li = [(np.abs(depth - gt), i) for depth, i in d[pixel]]
            res[pixel] = min(li)[1]
        else:
            pass

    li = [res[pixel] for pixel in res]
    res = np.zeros(img_idx.shape, np.bool)
    res[li] = True
    return res


