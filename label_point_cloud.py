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
    #
    CUBE, MAX_Z, EPSILON = 128, 50, 0.001
    host_name = socket.gethostname()
    cam_name = 'DEV_000F3102F884'

    # Server
    if host_name == 'cvg-desktop-17-ubuntu':
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        cam_msk_path  = f'data/2018-10-08-Calibration-Data/mask_{cam_name}.png'
        cam_int_path  =  'data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_world_camera.txt'
        pc_path       =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_0.zip'
        img_path      =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_fisheye/DEV_000F3102F884/%05d.png'
        depth_path    =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_depth/DEV_000F3102F884/%05d.pgm'
        downsampling_scale = 2

    # Local
    if host_name.startswith('lizuoyue') or host_name.startswith('staff-net-vpn-dhcp'):
        import matplotlib
        import matplotlib.pyplot as plt
        from plyfile import PlyData, PlyElement
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        cam_msk_path  = f'../autovision_day_night_data/2018-10-08-Calibration-Data/mask_{cam_name}.png'
        cam_int_path  =  '../autovision_day_night_data/2018-10-08-Calibration-Data/camera_system_cal.json'
        cam_ext_path  =  '../autovision_day_night_data/poses_T_world_camera.txt'
        pc_path       =  '../autovision_day_night_data/point_cloud/point_cloud_0_sample_100.zip'
        img_path      =  '../autovision_day_night_data/img_fisheye/%05d.png'
        depth_path    =  '../autovision_day_night_data/img_depth/%05d.pgm'
        downsampling_scale = 2

    #
    mat_cam_int, img_size, xi = get_cam_int_np_3x3(cam_int_path, cam_name, downsampling_scale)
    # mat_local_to_world = get_cam_ext_np_4x4(np.loadtxt(cam_loc_path))
    # mat_world_to_local = np.linalg.inv(mat_local_to_world)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    cam_mask = np.array(Image.open(cam_msk_path).resize(img_size))[..., 0]

    #
    pc_coord, pc_color = pc_str_lines2nxXYZ1_and_RGB(get_pc_nxstr(pc_path))

    #
    np.set_printoptions(suppress=True)
    cmap = matplotlib.cm.get_cmap('bwr')

    #
    # for i, pose in tqdm.tqdm(list(enumerate(cam_poses[:1500]))):
    for i, pose in list(enumerate(cam_poses[:1500])):

        if i < 400 or i > 400:
            continue

        if False:
            depth = np.array(Image.open(depth_path % i)) / 32767 * MAX_Z
            depth = depth.reshape((-1))
            depth_valid = (0.1 < depth) & (depth < MAX_Z)
        else:
            depth = json.load(open(depth_path.replace('.pgm', '.json') % i))
            depth = np.array(depth['depth_map']['data'])
            depth_valid = (0 < depth)

        depth_min = depth * (1 - EPSILON)
        depth_max = depth * (1 + EPSILON)

        # mat_cam_to_local = get_cam_ext_np_4x4(pose)
        # mat_local_to_cam = np.linalg.inv(mat_cam_to_local)
        # mat_world_to_cam = mat_local_to_cam.dot(mat_world_to_local)
        # mat_cam_to_world = mat_local_to_world.dot(mat_cam_to_local)

        mat_cam_to_world = get_cam_ext_np_4x4(pose)
        mat_world_to_cam = np.linalg.inv(mat_cam_to_world)
        cam_coord = mat_cam_to_world[:3, 3]

        # Filter 1 - only consider points near camera
        idx =       (pc_coord[:, 0] > cam_coord[0] - CUBE)
        idx = idx & (pc_coord[:, 0] < cam_coord[0] + CUBE)
        idx = idx & (pc_coord[:, 1] > cam_coord[1] - CUBE)
        idx = idx & (pc_coord[:, 1] < cam_coord[1] + CUBE)
        pc_near_cam_coord = pc_coord[idx]
        pc_near_cam_color = pc_color[idx]

        # Filter 2 - only consider points in front of camera
        pc_cam_coord = mat_world_to_cam[:3].dot(pc_near_cam_coord.T)

        if False: # world
            pc_cam_coord = np.loadtxt(pc_world_path % i)
            pc_cam_coord = mat_world_to_cam[:3, :3].dot(pc_cam_coord.T).T + mat_world_to_cam[:3, 3]
            # pc_cam_coord = mat_world_to_local[:3, :3].dot(pc_cam_coord.T).T + mat_world_to_local[:3, 3]
            pc_cam_coord = pc_cam_coord.T
            print(pc_cam_coord[:,:5])

        if False: # local
            pc_cam_coord = np.loadtxt(pc_local_path % i)
            pc_cam_coord = mat_local_to_cam[:3, :3].dot(pc_cam_coord.T).T + mat_local_to_cam[:3, 3]
            # pc_cam_coord = mat_local_to_world[:3, :3].dot(pc_cam_coord.T).T + mat_local_to_world[:3, 3]
            pc_cam_coord = pc_cam_coord.T
            print(pc_cam_coord[:,:5])

        # continue

        idx = pc_cam_coord[-1] > 0
        pc_cam_coord = pc_cam_coord[:, idx]
        pc_cam_color = pc_near_cam_color[idx]
        pc_z = pc_cam_coord[-1].copy()
        pc_dist = np.sqrt(np.sum(pc_cam_coord * pc_cam_coord, axis=0))
        pc_cam_coord /= pc_dist

        if False:
            pc_cam_coord = get_normalized_points(100000, 3, abs_axis=-1).T

        pc_cam_coord[-1] += xi
        pc_cam_coord /= pc_cam_coord[-1]

        # Filter 3 - only consider points visible in the image
        x, y, _ = mat_cam_int.dot(pc_cam_coord)
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)
        idx = ((x >= 0) & (x < img_size[0]) & (y >= 0) & (y < img_size[1])).nonzero()[0]
        x, y, pc_z, pc_dist, pc_cam_color = x[idx], y[idx], pc_z[idx], pc_dist[idx], pc_cam_color[idx]
        img_1d_idx = y * img_size[0] + x

        # Filter 4 - only consider the closest point for each pixel
        valid = verify_distance(img_1d_idx, pc_dist)
        img_1d_idx = img_1d_idx[valid]
        pc_z = pc_z[valid]
        pc_cam_color = pc_cam_color[valid]

        # Filter 5 - only consider point has a valid ground truth depth
        if_has_gt_depth = depth_valid[img_1d_idx]
        img_1d_idx_has_gt_dep = img_1d_idx[if_has_gt_depth]
        pc_z_has_gt_dep = pc_z[if_has_gt_depth]

        if True:
            fake_depth = depth * 0
            fake_depth[img_1d_idx] = pc_z

            fake_depth_has_gt_dep = depth * 0
            fake_depth_has_gt_dep[img_1d_idx_has_gt_dep] = pc_z_has_gt_dep

            gt_depth = depth

            # log_depth = np.log(pc_z / depth[img_1d_idx])
            # show_log_depth = depth * 0
            # show_log_depth[img_1d_idx] = log_depth
            # log_depth = ((np.clip(log_depth, -2, 3) + 2) / 5 * 255).astype(np.uint8)
            # log_depth = (cmap(log_depth) * 255).astype(np.uint8)
            # show_log_depth = np.zeros((depth.shape[0], 3), np.uint8)
            # show_log_depth[img_1d_idx] = log_depth[...,:3]

            to_show = np.vstack([
                np.hstack([np.minimum(  gt_depth, MAX_Z).reshape(img_size[::-1])] * 2),
                np.hstack([np.minimum(fake_depth, MAX_Z).reshape(img_size[::-1]), np.minimum(fake_depth_has_gt_dep, MAX_Z).reshape(img_size[::-1])]),
            ])
            plt.figure(figsize=(20.48, 10.88))
            plt.imshow(to_show)
            plt.savefig('res.pdf')

        # Filter 6 - only consider point has an accurate depth
        valid = (depth_min[img_1d_idx_has_gt_dep] < pc_z_has_gt_dep) & (pc_z_has_gt_dep < depth_max[img_1d_idx_has_gt_dep])
        for aaa, bbb in zip(depth[img_1d_idx_has_gt_dep], pc_z_has_gt_dep):
            print(aaa, bbb)
        print(valid.mean(), valid.sum())
        
        if False:
            fake_img = np.ones(img_size[::-1], np.uint8) * 255
            fake_img[cam_mask < 10] = 127
        else:
            fake_img = np.array(Image.open(img_path % i).convert('RGB'))

        fake_imgs = [fake_img.reshape((-1, 3)).copy() for _ in range(4)]

        fake_imgs[1][img_1d_idx] = pc_cam_color
        fake_imgs[2][img_1d_idx] = pc_cam_color

        fake_imgs = [item.reshape(img_size[::-1]+(3,)) for item in fake_imgs]

        # gt_img = np.array(Image.open(img_files[i])).reshape(-1)
        # print((gt_img[y * 1024 + x] == pc_near_cam_color[idx]).mean())
        to_save = np.vstack([np.hstack([fake_imgs[0], fake_imgs[1]]), np.hstack([fake_imgs[2], fake_imgs[3]])])
        Image.fromarray(to_save).save('fake_img_%05d_%02d.png' % (i, CUBE))
        # plt.imshow(fake_img)
        # plt.show()

