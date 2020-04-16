from utils import get_pc_nxstr
import random
import os

if __name__ == '__main__':

    random.seed(7)
    path="data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud"
    os.system(f'rm {path}*.txt')
    quit()

    for i in range(9):
        os.system(f'zip {path}_sample_{i}.zip {path}_{i}.txt')
    quit()

    for i in range(9):
        print(i)
        path = f'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_{i}.zip'
        pc_str_lines = get_pc_nxstr(path)
        random.shuffle(pc_str_lines)
        num = int(len(pc_str_lines) / 100)
        with open(path.replace('.zip', '.txt'), 'w') as f:
            for line in pc_str_lines[:num]:
                f.write(line + '\n')
