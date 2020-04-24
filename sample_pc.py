from utils import get_pc_nxstr
import random
import os

if __name__ == '__main__':

    path = 'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/'
    random.seed(7)
    for i in range(9):
        print(i)
        pc_str_lines = get_pc_nxstr(path + f'point_cloud_{i}.zip')
        random.shuffle(pc_str_lines)
        num = int(len(pc_str_lines) / 100)
        with open(path.replace('.zip', '_sample_100.txt'), 'w') as f:
            for line in pc_str_lines[:num]:
                f.write(line + '\n')
        os.system(f'cd {path}; zip {path}_{i}_sample_100.zip {path}_{i}_sample_100.txt; rm *.txt; cd ../../../;')
