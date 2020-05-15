import random
import os
import numpy as np
import tqdm, time
from utils import *
from scipy.spatial import cKDTree

def get_day_pc(day_pc_path, day_label_path, show_time=True):
    pc_coord = pc_str_lines2nxXYZ1(get_pc_nxstr(day_pc_path, show_time=show_time), show_time=show_time)
    pc_d = np.load(day_label_path)
    pc_label = pc_d['label']
    # pc_color = pc_d['color']
    return pc_coord, pc_label#, pc_color

def get_label(p, coord, label):
    assert(coord.shape[0] == label.shape[0])
    idx = label < 255
    coord = coord[idx]
    label = label[idx]
    if label.shape[0] == 0:
        return 255
    inv_dist = 1.0 / np.sqrt(np.sum((coord - p) ** 2, axis=-1))
    num = label.max() + 1
    score = np.zeros((num,))
    for i in range(num):
        score[i] = inv_dist[label == i].sum()
    return np.argmax(score)

if __name__ == '__main__':

    BUFFER = 0.1
    SHOW_TIME = True

    colormap = create_autovision_simple_label_colormap()

    day_pc_path = 'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
    day_label_path = '1_day_pc_label/pc_label_%d.npz'
    night_pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/local_point_clouds/%d.zip'

    for i in range(9):
        # day_pc_coord, day_pc_label = get_day_pc(day_pc_path % i, day_label_path % i)
        nightObj = nightLocalPointCloud(night_pc_path)
        for j in tqdm.tqdm(list(range(nightObj.num))):
            night_pc = get_next_transformed_local_pc()

    quit()


    # for i in tqdm.tqdm(list(range(2, 102))):
    for i in range(2, 102):
        night_pc = np.loadtxt(night_pc_path % i)
        night_mat = np.loadtxt(night_mat_path % i)[:3]
        night_pc = night_pc[:,:4]
        night_pc[:,3] = 1
        night_pc = night_mat.dot(night_pc.T).T

        x_min, y_min, _ = night_pc.min(axis=0) - BUFFER
        x_max, y_max, _ = night_pc.max(axis=0) + BUFFER
        idx =        day_pc_coord[:, 0] >= x_min
        idx = idx & (day_pc_coord[:, 0] <= x_max)
        idx = idx & (day_pc_coord[:, 1] >= y_min)
        idx = idx & (day_pc_coord[:, 1] <= y_max)

        local_day_pc_coord = day_pc_coord[idx, :3]
        local_day_pc_label = day_pc_label[idx]

        tree = cKDTree(local_day_pc_coord)

        night_pc_label = []
        for night_p in tqdm.tqdm(list(night_pc)):
            nb = tree.query_ball_point(night_p, BUFFER)
            night_pc_label.append(get_label(night_p, local_day_pc_coord[nb], local_day_pc_label[nb]))
        night_pc_label = np.array(night_pc_label)

        np.savez_compressed(night_pc_path.replace('.txt', '.npz') % i, label=night_pc_label, color=colormap[night_pc_label])
        night_pc_with_color = np.concatenate([night_pc, colormap[night_pc_label].astype(np.float)], axis=1)
        np.random.shuffle(night_pc_with_color)
        np.savetxt(night_pc_path.replace('.txt', '_sample_100_with_color.txt') % i, night_pc_with_color[:int(night_pc_with_color.shape[0]/100)])

    # pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/point_cloud_%d.txt'
    # mat_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/icp_T_day_night/point_cloud_%d_T_day_night.txt'
    # random.seed(7)
    # li = []
    # for i in tqdm.tqdm(list(range(2, 102))):
    #     pc = np.loadtxt(pc_path % i)
    #     mat = np.loadtxt(mat_path % i)
    #     num = int(pc.shape[0] / 100)
    #     np.random.shuffle(pc)
    #     pc[:,3] = 1
    #     li.append(mat.dot(pc[:num,:4].T).T)
    # pc = np.concatenate(li)
    # np.savetxt('merge_night_sample.txt', pc, fmt='%.12f', delimiter=';')

