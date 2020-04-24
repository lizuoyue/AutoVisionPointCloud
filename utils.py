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

def get_normalized_points(num, dim, abs_axis=None):
    u = np.random.normal(size=(num, dim))
    u_len = np.sqrt(np.sum(u * u, axis=1))
    u = (u.T / u_len).T
    if abs_axis is not None:
        u[:, abs_axis] = np.abs(u[:, abs_axis])
    return u

def verify_depth(img_idx, pc_depth, gt_depth=None):
    assert(img_idx.shape[0] == pc_depth.shape[0])
    if gt_depth is not None:
        assert(img_idx.max() < gt_depth.shape[0])
    d = {}
    for i in range(img_idx.shape[0]):
        pixel = img_idx[i]
        if pixel in d:
            old_i = d[pixel]
            if gt_depth is not None:
                depth = gt_depth[pixel]
                if np.abs(pc_depth[i] - depth) < np.abs(pc_depth[old_i] - depth):
                    d[pixel] = i
                else:
                    pass
            else:
                if pc_depth[i] < pc_depth[old_i]:
                    d[pixel] = i
                else:
                    pass
        else:
            d[pixel] = i
    li = [d[key] for key in d]
    res = np.zeros(img_idx.shape, np.bool)
    res[li] = True
    return res



