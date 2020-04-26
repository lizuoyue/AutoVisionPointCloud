import numpy as np
import tqdm

from utils import get_cam_poses_nx7
from utils import get_cam_ext_np_4x4

import matplotlib.pyplot as plt

if __name__ == '__main__':

    cam_day_path   = '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_world_camera.txt'
    cam_night_path = '../autovision_day_night_data/2018-11-01-Lim-Chu-Kang-Run-3-Night/poses_T_world_camera.txt'

    cam_day_poses = get_cam_poses_nx7(cam_day_path)[77: 10064 + 1]
    cam_night_poses = get_cam_poses_nx7(cam_night_path)[2833: 17266 + 1]
    print('Day:      ', len(cam_day_poses))
    print('Night:    ', len(cam_night_poses))
    print('Night/Day:', len(cam_night_poses) / len(cam_day_poses))

    locs = [[], []]
    for i, poses in enumerate([cam_day_poses, cam_night_poses]):
        for pose in poses:
            mat_cam_to_world = get_cam_ext_np_4x4(pose)
            # locs[i].append(mat_cam_to_world[0: 2, 3])
            locs[i].append(mat_cam_to_world[0: 3, 3])
        if True:
            locs[i] = [locs[i][0]] + locs[i]
        locs[i] = np.stack(locs[i])
    day_loc, night_loc = locs

    if False:
        plt.plot(day_loc[:, 0], day_loc[:, 1], 'r')
        plt.plot(night_loc[:, 0], night_loc[:, 1], 'b')
        plt.show()

    if False:
        ref = day_loc[0]
        day_dist = np.sqrt(np.sum((day_loc - ref) ** 2, axis=-1))
        night_dist = np.sqrt(np.sum((night_loc - ref) ** 2, axis=-1))
        plt.plot(day_dist)
        plt.plot(night_dist)
        plt.show()

    if True:
        for loc in [day_loc, night_loc]:
            dist = np.cumsum(np.sqrt(np.sum((loc[1:] - loc[:-1]) ** 2, axis=-1)))
            # plt.plot(dist)
            # plt.show()
            comp = 1000
            for i, d in enumerate(dist):
                if d > comp:
                    # print(i - 1, dist[i - 1])
                    print(i)
                    comp += 1000
            # print(dist[-1])
            print('===')
        quit()

    # Sep after removal:
    day_sep = [0, 1341, 2512, 3482, 4700, 5692, 6641, 7702, 8967, 9988]
    night_sep = [0, 1780, 3404, 4956, 6683, 8424, 10000, 11534, 13122, 14434]

    if True:
        for name, loc, sep in zip(['day', 'night'], [day_loc, night_loc], [day_sep, night_sep]):
            for i, (a, b) in enumerate(zip(sep[:-1], sep[1:])):
                with open(name + '_%d.txt' % i, 'w') as f:
                    for line in loc[a: b]:
                        f.write('%.9lf %.9lf %.9lf\n' % tuple(line))










