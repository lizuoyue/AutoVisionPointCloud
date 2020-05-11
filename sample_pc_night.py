import random
import os
import numpy as np
import tqdm

if __name__ == '__main__':

    pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/point_cloud_%d.txt'
    mat_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/icp_T_day_night/point_cloud_%d_T_day_night.txt'
    random.seed(7)
    # li = []
    # for i in tqdm.tqdm(list(range(2, 102))):
    #     pc = np.loadtxt(pc_path % i)
    #     mat = np.loadtxt(mat_path % i)[:3]
    #     num = int(pc.shape[0] / 100)
    #     np.random.shuffle(pc)
    #     pc[:,3] = 1
    #     li.append(mat.dot(pc[:num,:4].T).T)
    # pc = np.concatenate(li)
    # np.savetxt('merge_night_sample.txt', pc, fmt='%.12f', delimiter=';')
    for i in tqdm.tqdm(list(range(2, 102))):
        pc = np.loadtxt(pc_path % i)
        num = int(pc.shape[0] / 100)
        np.random.shuffle(pc)
        np.savetxt((pc_path % i).replace('.txt', '_sample_100.txt'), pc[:num], fmt='%.12f', delimiter=' ')
