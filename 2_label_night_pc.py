import random
import os
import numpy as np
import tqdm, time
from utils import *
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

def get_day_pc(day_pc_path, day_label_path, show_time=True):
    pc_coord = pc_str_lines2nxXYZ1(get_pc_nxstr(day_pc_path, show_time=show_time), show_time=show_time)
    pc_d = np.load(day_label_path)
    pc_label = pc_d['label']
    pc_label[pc_label > 200] = 15
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

def get_label_with_dist(dist, label):
    idx = label < 255
    dist = dist[idx]
    label = label[idx]
    if label.shape[0] == 0:
        return 255
    inv_dist = 1.0 / dist
    num = label.max() + 1
    score = np.zeros((num,))
    for i in range(num):
        score[i] = inv_dist[label == i].sum()
    return np.argmax(score)

if __name__ == '__main__':

    BUFFER_NUM = 32
    BUFFER_DIST = 0.1
    SHOW_TIME = True

    colormap = create_autovision_simple_label_colormap()
    one_hot_tabel = np.eye(16, dtype=np.float32)

    day_pc_path = 'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
    day_label_path = '1_day_pc_label/pc_label_%d.npz'
    night_pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/local_point_clouds/%d.zip'
    save_path = '2_night_pc_label_add'

    for i in range(1,9):
        day_pc_coord, day_pc_label = get_day_pc(day_pc_path % i, day_label_path % i, show_time=True)
        day_pc_label = day_pc_label[:day_pc_coord.shape[0]]
        nightObj = nightLocalPointCloud(night_pc_path % (i-1))
        sample = []
        # for j in tqdm.tqdm(list(range(nightObj.num))):
        for j in [-2, -1]:
            n, night_pc = nightObj.get_next_transformed_local_pc(it=j)

            x_min, y_min, _ = night_pc.min(axis=0) - BUFFER_DIST
            x_max, y_max, _ = night_pc.max(axis=0) + BUFFER_DIST
            idx =        day_pc_coord[:, 0] >= x_min
            idx = idx & (day_pc_coord[:, 0] <= x_max)
            idx = idx & (day_pc_coord[:, 1] >= y_min)
            idx = idx & (day_pc_coord[:, 1] <= y_max)

            local_day_pc_coord = day_pc_coord[idx, :3]
            local_day_pc_label = day_pc_label[idx]

            tree = KDTree(local_day_pc_coord, leaf_size=100)

            dist, nb = tree.query(night_pc, k=BUFFER_NUM, return_distance=True, sort_results=False) # N * BUFFER_NUM
            weight = 1.0 / dist
            weight[dist > BUFFER_DIST] = 0
            weight = weight.astype(np.float32)

            score = one_hot_tabel[local_day_pc_label[nb.flatten()]].reshape(nb.shape + (one_hot_tabel.shape[0],)) # N * BUFFER_NUM * CLASS_NUM
            score = (score * weight[..., np.newaxis]).sum(axis=1)
            score_max = score.max(axis=1)
            night_pc_label = score.argmax(axis=1)
            night_pc_label[score_max < 1e-6] = 255
            night_pc_label[night_pc_label == 15] == 255

            np.savez_compressed(f'{save_path}/{i}_{n}.npz', label=night_pc_label, color=colormap[night_pc_label])

            night_pc_with_color = np.concatenate([night_pc, colormap[night_pc_label].astype(np.float)], axis=1)
            np.random.shuffle(night_pc_with_color)
            sample.append(night_pc_with_color[:int(night_pc_with_color.shape[0]/100)])

            if j % 10 == 0:
                np.savetxt(f'{save_path}/{i}_sample_100.txt', np.concatenate(sample))

        np.savetxt(f'{save_path}/{i}_sample_100.txt', np.concatenate(sample))










