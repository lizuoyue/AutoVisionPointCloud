import zipfile
import time
import json
from PIL import Image
import numpy as np
import glob
import os
import tqdm
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import socket

def get_cam_int_np_3x3(path, camera_name):
    d = json.load(open(path, 'r'))[camera_name]
    return np.array([
        [d['fx'], 0, d['cx']],
        [0, d['fy'], d['cy']],
        [0, 0, 1]
    ], np.float32)

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
    basename = os.path.basename(path).replace('.zip', '.txt')
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
    return res[:, :4], res[:, 4].astype(np.uint8)



