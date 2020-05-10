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
    CUBE, MAX_Z, PC_NUM_SEP, EPSILON = 100, 50, 9, 0.01
    SEP = [0, 1780, 3404, 4956, 6683, 8424, 10000, 11534, 13122, 14434]
    FRAME_FROM, FRAME_TO = 2833, 17267
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
        cam_ext_path  =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/poses_T_world_camera.txt'
        pc_path       =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
        img_path      =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_fisheye/DEV_000F3102F884/%05d.png'
        depth_path    =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_depth/DEV_000F3102F884/%05d.pgm'
        downsampling_scale = 2

    # Local
    if host_name.startswith('lizuoyue') or host_name.startswith('staff-net-vpn-dhcp'):
        import matplotlib
        import matplotlib.pyplot as plt
        cmap = matplotlib.cm.get_cmap('viridis')
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        cam_msk_path  = f'../autovision_day_night_data/2018-10-08-Calibration-Data/mask_{CAM_NAME}.png'
        cam_int_path  =  '../autovision_day_night_data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  '../autovision_day_night_data/2018-11-01-Lim-Chu-Kang-Run-3-Night/poses_T_world_camera.txt'
        pc_path       =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_cloud/point_cloud_%d_sample_100.zip'
        img_path      =  '../autovision_day_night_data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_fisheye/%05d.png'
        depth_path    =  '../autovision_day_night_data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_depth/%05d.pgm'
        downsampling_scale = 2

    #
    mat_cam_int, img_size, xi = get_cam_int_np_3x3(cam_int_path, CAM_NAME, downsampling_scale)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    cam_mask = np.array(Image.open(cam_msk_path).resize(img_size))[..., 0]

    #
    np.set_printoptions(suppress=True)
    log = open('log.out', 'w')
    os.system('mkdir night_depth night_semantics')

    #
    for i, pose in tqdm.tqdm(list(enumerate(cam_poses[FRAME_FROM: FRAME_TO]))):

        if i in SEP:
            pc_coord = pc_str_lines2nxXYZ1(get_pc_nxstr(pc_path % WHICH_PC[i], show_time=SHOW_TIME), show_time=SHOW_TIME)
            pc_d = np.load('pc_label_%d.npz' % i)
            pc_label = pc_d['label']
            pc_color = pc_d['color']
            pc_index = np.arange(pc_coord.shape[0])
            assert(pc_label.shape[0] == pc_coord.shape[0])
            assert(pc_color.shape[0] == pc_coord.shape[0])

        if i + FRAME_FROM != 3400:
            continue

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

        pose = np.array([0.465872010462144 -0.517772394781267 0.550068720893017 -0.460759611256837 356270.50130326 158611.501875941 34.2677767156021])

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

        # Sort?
        order = np.argsort(pc_z)[::-1]
        x, y = x[order], y[order]
        pc_z = pc_z[order]
        pc_cam_index = pc_cam_index[order]

        # Get image 1D index
        img_1d_idx = y * img_size[0] + x

        # Filter 4 - only consider a point which has a valid ground truth depth
        if False:
            idx = depth_valid[img_1d_idx]
            pc_z = pc_z[idx]
            img_1d_idx = img_1d_idx[idx]
            pc_cam_index = pc_cam_index[idx]
        num_pixel_has_points = np.unique(img_1d_idx).shape[0]

        # Filter 5 - only consider a point which has an accurate depth
        if False:
            idx = (depth_min[img_1d_idx] < pc_z) & (pc_z < depth_max[img_1d_idx])
            pc_z = pc_z[idx]
            img_1d_idx = img_1d_idx[idx]
            pc_cam_index = pc_cam_index[idx]

        num_pixel_has_acc_points = np.unique(img_1d_idx).shape[0]

        rate_point_acc = num_pixel_has_acc_points / num_pixel_has_points
        rate_pixel_acc = num_pixel_has_acc_points / depth_valid.sum()

        toc = time.time()
        if SHOW_TIME:
            print('Computing costs %.3lf seconds.' % (toc - tic))

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
            Image.fromarray(to_show).save('night_depth/%05d.png' % (i + FRAME_FROM))
            toc = time.time()
            if SHOW_TIME:
                print('Creating fake depth costs %.3lf seconds.' % (toc - tic))

        # Write log and result
        log.write('%d %.6lf %.6lf\n' % (i + FRAME_FROM, rate_point_acc, rate_pixel_acc))
        log.flush()

        if True:
            tic = time.time()
            fake_img = np.array(Image.open(img_path % (i + FRAME_FROM)).convert('RGB')).reshape((-1, 3))
            fake_img[img_1d_idx] = pc_color[pc_cam_index]
            Image.fromarray(fake_img.reshape(img_size[::-1] + (3, ))).save('night_semantics/%05d.png' % (i + FRAME_FROM))
            if SHOW_TIME:
                print('Creating fake image costs %.3lf seconds.' % (toc - tic))

    log.close()

