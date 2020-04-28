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
        cam_ext_path  =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/poses_T_world_camera.txt'
        pc_path       =  'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/point_clouds_length_1000m_overlap_100m/point_cloud_%d.zip'
        img_path      = f'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_fisheye/{CAM_NAME}/%05d.png'
        depth_path    = f'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_depth/{CAM_NAME}/%05d.pgm'
        sem_path      = f'data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_semantics/{CAM_NAME}/%05d.png'
        mapping_path  =  'result/%05d.npz'
        downsampling_scale = 2
        pc_size = [59050601, 59677974, 51778606, 63369535, 53160322, 49842875, 56162369, 63787850, 77742044]

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
        sem_path      =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/img_semantics/%05d.png'
        mapping_path  =  '../autovision_day_night_data/2018-10-18-Lim-Chu-Kang-Run-1-Day/mapping/%05d.npz'
        downsampling_scale = 2
        pc_size = [590506]

    #
    np.set_printoptions(suppress=True)

    #
    for which_pc, (a, b) in enumerate(zip(SEP[:-1], SEP[1:])):
        pc_num = pc_size[which_pc]
        pc_label = [[] for _ in range(pc_num)]
        for i in tqdm.tqdm(list(range(a, b))):
            fid = i + FRAME_FROM
            sem = np.array(Image.open(sem_path % fid)).reshape((-1))
            d = np.load(mapping_path % fid)
            assert(d['which_pc'] == which_pc)
            pc_idx = d['pc_cam_index']
            pc_sem = sem[d['img_1d_idx']]
            for pci, pcs in zip(pc_idx, pc_sem):
                if pcs < 255:
                    pc_label[pci].append(pcs)
        for li in pc_label:
            if len(li) > 1:
                print(li)
        break



