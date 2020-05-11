import random
import os
import numpy as np
import tqdm, time
from utils import *
from scipy.spatial import cKDTree

def get_next_day_pc(day_pc_path):
    for i in range(9):
        pc_coord = pc_str_lines2nxXYZ1(get_pc_nxstr(day_pc_path % i, show_time=SHOW_TIME), show_time=SHOW_TIME)
        pc_d = np.load('pc_label/pc_label_%d.npz' % i)
        pc_label = pc_d['label']
        pc_color = pc_d['color']
        # print('Day Point Cloud %d' % i)
        yield pc_coord, pc_label, pc_color
    return None

def intersection_area(a, b):
    # xmin xmax ymin ymax
    # returns 0 if rectangles don't intersect
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    if (dx >= 0) and (dy >= 0):
        area = dx * dy
    else:
        area = 0
    return area

if __name__ == '__main__':

    BUFFER = 0.25
    SHOW_TIME = False

    day_pc_path = 'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d_sample_100.zip'
    night_pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/point_cloud_%d_sample_100.txt'
    night_mat_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/icp_T_day_night/point_cloud_%d_T_day_night.txt'

    # day_pc_generator = get_next_day_pc(day_pc_path)
    # for _ in range(9):
    #     pc_coord, _, _ = next(day_pc_generator)
    #     print('Min', pc_coord.min(axis=0))
    #     print('Max', pc_coord.max(axis=0))
    # quit()

    ratio = 0
    for i in range(2, 102):
        night_pc = np.loadtxt(night_pc_path % i)
        night_mat = np.loadtxt(night_mat_path % i)[:3]
        night_pc = night_pc[:,:4]
        night_pc[:,3] = 1
        night_pc = night_mat.dot(night_pc.T).T
        x_min, y_min, _ = night_pc.min(axis=0) - BUFFER
        x_max, y_max, _ = night_pc.max(axis=0) + BUFFER

        # night_area = (x_max - x_min) * (y_max - y_min)
        # ia = intersection_area(day_pc_range, [x_min, x_max, y_min, y_max])
        # ratio = ia / night_area

        day_pc_generator = get_next_day_pc(day_pc_path)
        li = []
        for _ in range(9):
            day_pc_coord, day_pc_label, _ = next(day_pc_generator)
            # day_pc_x_min, day_pc_y_min, _ = day_pc_coord.min(axis=0)
            # day_pc_x_max, day_pc_y_max, _ = day_pc_coord.max(axis=0)
            # day_pc_range = [day_pc_x_min, day_pc_x_max, day_pc_y_min, day_pc_y_max]

            # ia = intersection_area(day_pc_range, [x_min, x_max, y_min, y_max])
            # ratio = ia / night_area
            # assert(ratio >= 0.95)
            idx =        day_pc_coord[:, 0] >= x_min
            idx = idx & (day_pc_coord[:, 0] <= x_max)
            idx = idx & (day_pc_coord[:, 1] >= y_min)
            idx = idx & (day_pc_coord[:, 1] <= y_max)
            li.append(idx.sum())

        print(i, li)
        continue
    quit()


    pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/point_cloud_%d.txt'
    mat_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/icp_T_day_night/point_cloud_%d_T_day_night.txt'
    random.seed(7)
    li = []
    for i in tqdm.tqdm(list(range(2, 102))):
        pc = np.loadtxt(pc_path % i)
        mat = np.loadtxt(mat_path % i)
        num = int(pc.shape[0] / 100)
        np.random.shuffle(pc)
        pc[:,3] = 1
        li.append(mat.dot(pc[:num,:4].T).T)
    pc = np.concatenate(li)
    np.savetxt('merge_night_sample.txt', pc, fmt='%.12f', delimiter=';')


