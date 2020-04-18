import numpy as np
import tqdm
from utils import get_cam_poses_nx7
from utils import get_cam_ext_np_4x4

if __name__ == '__main__':
    cam_loc_path = '../autovision_day_night_data/T_world_local.txt'
    cam_ext_path = '../autovision_day_night_data/poses_T_local_camera.txt'

    mat_local_to_world = np.loadtxt(cam_loc_path)
    mat_world_to_local = np.linalg.inv(mat_local_to_world)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    with open('../autovision_day_night_data/cam_global_loc.txt', 'w') as f:
        for i, pose in tqdm.tqdm(list(enumerate(cam_poses))):
            mat_cam_to_local = get_cam_ext_np_4x4(pose)
            mat_local_to_cam = np.linalg.inv(mat_cam_to_local)
            mat_world_to_cam = mat_local_to_cam.dot(mat_world_to_local)
            mat_cam_to_world = mat_local_to_world.dot(mat_cam_to_local)
            f.write('%.6f %.6f %.6f\n' % tuple(mat_cam_to_world[:3, -1]))
