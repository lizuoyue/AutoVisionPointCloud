from utils import get_pc_nxstr
import random
import os
import numpy as np

if __name__ == '__main__':

    path = 'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/'
    random.seed(7)
    for i in range(9):
        pc_str_lines = get_pc_nxstr(path + f'point_cloud_{i}.zip')
        pc_color = np.load('pc_label_%d.npz' % i)['color']

        idx = [i for i in range(len(pc_str_lines))]
        random.shuffle(idx)
        num = int(len(pc_str_lines) / 100)

        with open(path + f'point_cloud_{i}_with_color_sample_100.txt', 'w') as f:
            for i in idx[:num]:
                line = pc_str_lines[i]
                coord = line.split()[:3]
                color = [str(c) for c in list(pc_color[i])]
                f.write(' '.join(coord + color) + '\n')
        break
