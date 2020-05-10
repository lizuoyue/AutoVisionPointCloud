import random
import os
import numpy as np
import tqdm

if __name__ == '__main__':

    pc_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/point_cloud_%d.txt'
    mat_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/point_cloud/icp_T_day_night/point_cloud_%d_T_day_night.txt'
    random.seed(7)
    for i in tqdm.tqdm(list(range(2, 102))):
        pc = np.loadtxt(pc_path % i)
        mat = np.loadtxt(mat_path % i)
        continue
        random.shuffle(pc_str_lines)
        num = int(len(pc_str_lines) / 100)
        with open(path + f'point_cloud_{i}_sample_100.txt', 'w') as f:
            for line in pc_str_lines[:num]:
                f.write(line + '\n')
        # os.system(f'cd {path}; zip point_cloud_{i}_sample_100.zip point_cloud_{i}_sample_100.txt; rm *.txt; cd ../../../;')
        break
