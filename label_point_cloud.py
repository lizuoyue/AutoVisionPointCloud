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
    CUBE, EPSILON = 128, 0.2
    host_name = socket.gethostname()
    cam_name = 'DEV_000F3102F884'

    # Server
    if host_name == 'cvg-desktop-17-ubuntu':
        cam_int_path =  'data/2018-08-10-Calibration-Data/camera_system_cal.json'
        cam_loc_path =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/T_world_local.txt'
        cam_ext_path =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_local_camera.txt'
        cam_msk_path = f'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/2018-08-10-Calibration-Data/mask_{cam_name}_undist.png'
        pc_path      =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_0.zip'
        img_path     =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_undistorted/DEV_000F3102F884/%05d.png'
        depth_path   =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_depth/DEV_000F3102F884/%05d.pgm'
        downsampling_scale = 2

    # Local
    if host_name == 'lizuoyue.local' or host_name.startswith('staff-net-vpn-dhcp'):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        cam_int_path =  '../autovision_day_night_data/2018-08-10-Calibration-Data/camera_system_cal.json'
        cam_loc_path =  '../autovision_day_night_data/T_world_local.txt'
        cam_ext_path =  '../autovision_day_night_data/poses_T_local_camera.txt'
        cam_msk_path = f'../autovision_day_night_data/2018-08-10-Calibration-Data/mask_{cam_name}_undist.png'
        pc_path      =  '../autovision_day_night_data/point_cloud/point_cloud_0_sample.zip'
        img_path     =  '../autovision_day_night_data/img_undistorted/%05d.png'
        depth_path   =  '../autovision_day_night_data/img_depth/%05d.pgm'
        downsampling_scale = 2

    #
    mat_cam_int, img_size, xi = get_cam_int_np_3x3(cam_int_path, cam_name, downsampling_scale)
    mat_local_to_world = np.loadtxt(cam_loc_path)
    mat_world_to_local = np.linalg.inv(mat_local_to_world)
    cam_poses = get_cam_poses_nx7(cam_ext_path)
    cam_mask = np.array(Image.open(cam_msk_path).resize(img_size))

    #
    pc_coord, pc_color = pc_str_lines2nxXYZ1_and_RGB(get_pc_nxstr(pc_path))

    #
    np.set_printoptions(suppress=True)

    #
    for i, pose in tqdm.tqdm(list(enumerate(cam_poses[:1500]))):
    # for i, pose in list(enumerate(cam_poses[:1500])):

        # if i < 402 or i > 402:
        #     continue

        # a = np.array(Image.open(depth_path % i))
        # b = json.load(open(depth_path.replace('.pgm', '.json') % i))
        # plt.imshow(np.array(b['depth_map']['data']).reshape((544, 1024)))
        # plt.show()
        # # a[cam_mask[..., 0] < 10] = 16000
        # a[cam_mask[..., 0] > 10] = 0
        # plt.imshow(a)
        # plt.show()
        # continue

        # if i < 396 or i > 407:
        #     continue

        # depth = np.array(Image.open(depth_path % i)).reshape((-1))
        depth = json.load(open(depth_path.replace('.pgm', '.json') % i))
        depth = np.array(depth['depth_map']['data'])
        depth_min = depth * (1 - EPSILON)
        depth_max = depth * (1 + EPSILON)

        mat_cam_to_local = get_cam_ext_np_4x4(pose)
        mat_local_to_cam = np.linalg.inv(mat_cam_to_local)
        mat_world_to_cam = mat_local_to_cam.dot(mat_world_to_local)
        mat_cam_to_world = mat_local_to_world.dot(mat_cam_to_local)

        cam_coord = mat_cam_to_world[:3, 3]

        idx =       (pc_coord[:, 0] > cam_coord[0] - CUBE)
        idx = idx & (pc_coord[:, 0] < cam_coord[0] + CUBE)
        idx = idx & (pc_coord[:, 1] > cam_coord[1] - CUBE)
        idx = idx & (pc_coord[:, 1] < cam_coord[1] + CUBE)

        pc_near_cam_coord = pc_coord[idx]
        pc_near_cam_color = pc_color[idx]

        pc_cam_coord = mat_world_to_cam[:3].dot(pc_near_cam_coord.T)
        idx = pc_cam_coord[-1] > 0
        pc_cam_coord = pc_cam_coord[:, idx]
        pc_cam_color = pc_near_cam_color[idx]

        dist = np.sqrt(np.sum(pc_cam_coord * pc_cam_coord, axis=0))
        pc_cam_coord /= dist
        dist = dist.reshape((-1))

        # pc_cam_coord = get_normalized_points(100000, 3, abs_axis=-1).T

        # continue
        pc_cam_coord[-1] += xi
        pc_cam_coord /= pc_cam_coord[-1]

        x, y, z = mat_cam_int.dot(pc_cam_coord)
        x, y = np.round(x).astype(int), np.round(y).astype(int)
        idx = ((x >= 0) & (x < img_size[0]) & (y >= 0) & (y < img_size[1])).nonzero()[0]
        one_dim_idx = y[idx] * img_size[0] + x[idx]

        valid = ((depth_min[one_dim_idx] < dist[idx]) & (dist[idx] < depth_max[one_dim_idx]))
        idx = idx[valid]
        one_dim_idx = one_dim_idx[valid]

        if False:
            fake_img = np.ones(img_size[::-1], np.uint8) * 255
            fake_img[cam_mask < 10] = 127
        else:
            fake_img = np.array(Image.open(img_path % i))

        fake_img = fake_img.reshape((-1))
        fake_img[one_dim_idx] = pc_cam_color[idx]
        fake_img = fake_img.reshape(img_size[::-1])

        # print(dist[idx].shape, dist[idx].min(), dist[idx].max())
        # print(depth[one_dim_idx].shape, depth[one_dim_idx].min(), depth[one_dim_idx].max())
        # continue

        # hm = np.zeros((200, 100))
        # for iii, jjj in zip(dist[idx], depth[one_dim_idx].astype(float)):
        #     hm[199-int(iii), int(jjj)] += 1
        # plt.imshow(hm, vmin=0, vmax=20)
        # plt.show()

        # gt_img = np.array(Image.open(img_files[i])).reshape(-1)
        # print((gt_img[y * 1024 + x] == pc_near_cam_color[idx]).mean())

        Image.fromarray(fake_img).save('fake_img/%05d_%02d.png' % (i, CUBE))
        # plt.imshow(fake_img)
        # plt.show()






