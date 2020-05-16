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

    # Night init
    CAM_NAME = 'DEV_000F3102F884'
    CUBE, MAX_Z, PC_NUM_SEP, EPSILON = 100, 50, 9, 0.01
    SEP = [0, 1780, 3404, 4956, 6683, 8424, 10000, 11534, 13122, 14434]
    FRAME_FROM, FRAME_TO = 2833, 17267
    WHICH_PC = np.zeros((SEP[-1]), np.int32)
    WHICH_PC[SEP[1:-1]] = 1
    WHICH_PC = np.cumsum(WHICH_PC)
    assert(PC_NUM_SEP == (len(SEP) - 1))
    assert((FRAME_TO - FRAME_FROM) == SEP[-1])
    SHOW_TIME = False

    host_name = socket.gethostname()

    # Server
    if host_name == 'cvg-desktop-17-ubuntu':
        import matplotlib; matplotlib.use('agg')
        import matplotlib.pyplot as plt
        cmap = matplotlib.cm.get_cmap('viridis')
        cam_msk_path  = f'data/2018-10-08-Calibration-Data/mask_{CAM_NAME}.png'
        cam_int_path  =  'data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/poses_T_world_camera.txt'
        # pc_path       =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
        pc_path       =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/local_point_clouds/%d.zip'
        label_path    =  '2_night_pc_label'
        img_path      =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_fisheye/DEV_000F3102F884/%05d.png'
        depth_path    =  'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_depth/DEV_000F3102F884/%05d.pgm'
        downsampling_scale = 2

    #
    mat_cam_int, img_size, xi = get_cam_int_np_3x3(cam_int_path, CAM_NAME, downsampling_scale)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    cam_mask = np.array(Image.open(cam_msk_path).resize(img_size))[..., 0]

    #
    np.set_printoptions(suppress=True)
    log = open('3_night.out', 'w')
    os.system('mkdir 3_night_dep 3_night_sem')

    #
    for i, pose in tqdm.tqdm(list(enumerate(cam_poses[FRAME_FROM: FRAME_TO]))):
    # for i, pose in enumerate(cam_poses[FRAME_FROM: FRAME_TO]):

        if i in SEP:
            nightObj = nightLocalPointCloud(pc_path % WHICH_PC[i])
            pc_label, pc_color = nightObj.get_label_color(label_path)
            pc_coord = nightObj.get_pc()
            pc_index = np.arange(pc_coord.shape[0])
            print(f'Point cloud {WHICH_PC[i]}')
            print(pc_label.shape)
            print(pc_coord.shape)
            assert(pc_label.shape[0] == pc_coord.shape[0])

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

        # Sort?
        order = np.argsort(pc_z)[::-1]
        x, y = x[order], y[order]
        pc_z = pc_z[order]
        pc_cam_index = pc_cam_index[order]

        # Get image 1D index
        img_1d_idx = y * img_size[0] + x

        # Filter 4 - only consider a point which has a valid ground truth depth
        if True:
            idx = depth_valid[img_1d_idx]
            pc_z = pc_z[idx]
            img_1d_idx = img_1d_idx[idx]
            pc_cam_index = pc_cam_index[idx]
        num_pixel_has_points = np.unique(img_1d_idx).shape[0]

        # Filter 5 - only consider a point which has an accurate depth
        if True:
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
            Image.fromarray(to_show).save('3_night_dep/%05d.png' % (i + FRAME_FROM))
            toc = time.time()
            if SHOW_TIME:
                print('Creating fake depth costs %.3lf seconds.' % (toc - tic))

        # Write log and result
        log.write('%d %.6lf %.6lf\n' % (i + FRAME_FROM, rate_point_acc, rate_pixel_acc))
        log.flush()

        if True:
            tic = time.time()
            fake_img = np.array(Image.open(img_path % (i + FRAME_FROM)).convert('RGB')).reshape((-1, 3))
            fake_img[img_1d_idx] = (fake_img[img_1d_idx] * 0.5 + pc_color[pc_cam_index] * 0.5).astype(np.uint8)
            Image.fromarray(fake_img.reshape(img_size[::-1] + (3, ))).save('3_night_sem/%05d_vis.png' % (i + FRAME_FROM))

            fake_sem = np.ones(img_size[::-1], dtype=np.uint8).reshape((-1)) * 255
            fake_sem[img_1d_idx] = pc_label[pc_cam_index].astype(np.uint8)
            Image.fromarray(fake_sem.reshape(img_size[::-1])).save('3_night_sem/%05d.png' % (i + FRAME_FROM))
            toc = time.time()
            if SHOW_TIME:
                print('Creating fake image costs %.3lf seconds.' % (toc - tic))

    log.close()

