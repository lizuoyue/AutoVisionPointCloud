import random
import os
from plyfile import PlyData, PlyElement

if __name__ == '__main__':

    path = '../autovision_day_night_data/local_point_cloud/%05d.ply'
    random.seed(7)
    for i in range(400, 408):
        print(i)
        plydata = PlyData.read(path % i)
        pc_cam_coord = [list(item) for item in plydata.elements[0].data]
        # random.shuffle(pc_cam_coord)
        num = int(len(pc_cam_coord))
        with open((path % i).replace('.ply', '.txt'), 'w') as f:
            for line in pc_cam_coord[:num]:
                f.write('%.9lf %.9lf %.9lf\n' % tuple(line))

    # for i in range(9):
    #     os.system(f'zip {path}_{i}_sample.zip {path}_{i}.txt')
    # os.system(f'rm {path}*.txt')
