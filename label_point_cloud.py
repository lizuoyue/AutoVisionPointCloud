import zipfile
import time
import json
from PIL import Image
import numpy as np
import glob
import os
import tqdm
import socket
from utils import *

if __name__ == '__main__':

    # Day init
    CAM_NAME = 'DEV_000F3102F884'
    CUBE, MAX_Z, PC_NUM_SEP, EPSILON = 100, 50, 9, 0.001
    SEP = [0, 1341, 2512, 3482, 4700, 5692, 6641, 7702, 8967, 9988]
    FRAME_FROM, FRAME_TO = 77, 10065
    WHICH_PC = np.zeros((SEP[-1]), np.int32)
    WHICH_PC[SEP[1:-1]] = 1
    WHICH_PC = np.cumsum(WHICH_PC)
    assert(PC_NUM_SEP == (len(SEP) - 1))
    assert((FRAME_TO - FRAME_FROM) == SEP[-1])
    SHOW_TIME = True

    host_name = socket.gethostname()

    # Server
    if host_name == 'cvg-desktop-17-ubuntu':
        import matplotlib; matplotlib.use('agg')
        import matplotlib.pyplot as plt
        cmap = matplotlib.cm.get_cmap('viridis')
        cam_msk_path  = f'data/2018-10-08-Calibration-Data/mask_{CAM_NAME}.png'
        cam_int_path  =  'data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_world_camera.txt'
        pc_path       =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
        img_path      =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_fisheye/DEV_000F3102F884/%05d.png'
        depth_path    =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_depth/DEV_000F3102F884/%05d.pgm'
        downsampling_scale = 2

    # Local
    if host_name.startswith('lizuoyue') or host_name.startswith('staff-net-vpn-dhcp'):
        import matplotlib
        import matplotlib.pyplot as plt
        cmap = matplotlib.cm.get_cmap('viridis')
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        cam_msk_path  = f'../autovision_day_night_data/2018-10-08-Calibration-Data/mask_{CAM_NAME}.png'
        cam_int_path  =  '../autovision_day_night_data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_world_camera.txt'
        pc_path       =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_cloud/point_cloud_%d_sample_100.zip'
        img_path      =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_fisheye/%05d.png'
        depth_path    =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_depth/%05d.pgm'
        downsampling_scale = 2

    #
    mat_cam_int, img_size, xi = get_cam_int_np_3x3(cam_int_path, CAM_NAME, downsampling_scale)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    cam_mask = np.array(Image.open(cam_msk_path).resize(img_size))[..., 0]

    #
    np.set_printoptions(suppress=True)
    f_log = open('log.out', 'a')
    os.system('mkdir fake_img fake_depth result')

    #
    for i, pose in tqdm.tqdm(list(enumerate(cam_poses[FRAME_FROM: FRAME_TO]))):

        if i in SEP:
            pc_coord = pc_str_lines2nxXYZ1(get_pc_nxstr(pc_path % WHICH_PC[i], show_time=SHOW_TIME), show_time=SHOW_TIME)
            pc_index = np.arange(pc_coord.shape[0])

        if False:
            depth = np.array(Image.open(depth_path % i)) / 32767 * MAX_Z
            depth = depth.reshape((-1))
            depth_valid = (0 < depth) & (depth < MAX_Z)
        else:
            tic = time.time()
            depth = json.load(open(depth_path.replace('.pgm', '.json') % (i + FRAME_FROM)))
            depth = np.array(depth['depth_map']['data'])
            toc = time.time()
            if SHOW_TIME:
                print('Loading depth file costs %.3lf seconds.' % (toc - tic))
            depth_valid = (0 < depth)
        depth_min = depth * (1 - EPSILON)
        depth_max = depth * (1 + EPSILON)

        mat_cam_to_world = get_cam_ext_np_4x4(pose)
        mat_world_to_cam = np.linalg.inv(mat_cam_to_world)
        cam_loc = mat_cam_to_world[:3, 3]

        tic = time.time()

        # Filter 1 - only consider the points near the camera center
        idx =       (pc_coord[:, 0] > cam_loc[0] - CUBE)
        idx = idx & (pc_coord[:, 0] < cam_loc[0] + CUBE)
        idx = idx & (pc_coord[:, 1] > cam_loc[1] - CUBE)
        idx = idx & (pc_coord[:, 1] < cam_loc[1] + CUBE)
        pc_cam_coord = pc_coord[idx]
        pc_cam_index = pc_index[idx]

        # Rotate to the camera coordinate system
        pc_cam_coord = mat_world_to_cam[:3].dot(pc_cam_coord.T)

        # Filter 2 - only consider the points in front of the camera
        idx = pc_cam_coord[-1] > 0
        pc_cam_coord = pc_cam_coord[:, idx]
        pc_cam_index = pc_cam_index[idx]

        # Project to the unit ball and do fisheye distortion
        pc_z = pc_cam_coord[-1].copy()
        pc_cam_coord /= np.sqrt(np.sum(pc_cam_coord ** 2, axis=0))
        pc_cam_coord[-1] += xi
        pc_cam_coord /= pc_cam_coord[-1]
        x, y, _ = mat_cam_int.dot(pc_cam_coord)
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)

        # Filter 3 - only consider visible pixels
        idx = (x >= 0) & (x < img_size[0]) & (y >= 0) & (y < img_size[1])
        x, y = x[idx], y[idx]
        pc_z = pc_z[idx]
        pc_cam_index = pc_cam_index[idx]

        # Get image 1D index
        img_1d_idx = y * img_size[0] + x

        toc = time.time()
        if SHOW_TIME:
            print('Computing part A costs %.3lf seconds.' % (toc - tic))

        # Filter 4 - only choose one point for each pixel
        tic = time.time()
        idx = verify_depth(img_1d_idx, pc_z, depth)
        pc_z = pc_z[idx]
        img_1d_idx = img_1d_idx[idx]
        pc_cam_index = pc_cam_index[idx]
        toc = time.time()
        if SHOW_TIME:
            print('Verifying depth costs %.3lf seconds.' % (toc - tic))

        # Filter 5 - only consider a point which has a valid ground truth depth
        idx = depth_valid[img_1d_idx]
        pc_z = pc_z[idx]
        img_1d_idx = img_1d_idx[idx]
        pc_cam_index = pc_cam_index[idx]

        if True:
            tic = time.time()
            gt_depth = depth
            fake_depth = depth * 0
            fake_depth[img_1d_idx] = pc_z
            to_show = np.vstack([
                np.minimum(  gt_depth, MAX_Z).reshape(img_size[::-1]),
                np.minimum(fake_depth, MAX_Z).reshape(img_size[::-1]),
            ])
            to_show = (cmap((to_show - to_show.min()) / (to_show.max() - to_show.min())) * 255).astype(np.uint8)
            Image.fromarray(to_show).save('fake_depth/%05d.png' % (i + FRAME_FROM))
            toc = time.time()
            if SHOW_TIME:
                print('Creating fake depth costs %.3lf seconds.' % (toc - tic))

        # Filter 6 - only consider a point which has an accurate depth
        idx = (depth_min[img_1d_idx] < pc_z) & (pc_z < depth_max[img_1d_idx])
        pc_z = pc_z[idx]
        img_1d_idx = img_1d_idx[idx]
        pc_cam_index = pc_cam_index[idx]

        # Write log and result
        tic = time.time()
        f_log.write('%d %.6lf %d\n' % (i + FRAME_FROM, idx.mean(), idx.sum()))
        f_log.flush()
        with open('result/%05d.txt' % (i + FRAME_FROM), 'w') as f_res:
            f_res.write(f'{WHICH_PC[i]}\n')
            for a, b in zip(img_1d_idx, pc_cam_index):
                f_res.write(f'{a} {b}\n')
                f_res.flush()
        toc = time.time()
        if SHOW_TIME:
            print('Writing to file costs %.3lf seconds.' % (toc - tic))

        if True:
            tic = time.time()
            fake_img = np.array(Image.open(img_path % (i + FRAME_FROM))).reshape((-1))
            fake_img[img_1d_idx] = 0
            Image.fromarray(fake_img.reshape(img_size[::-1])).save('fake_img/%05d.png' % (i + FRAME_FROM))
            if SHOW_TIME:
                print('Creating fake image costs %.3lf seconds.' % (toc - tic))

    f_log.close()

