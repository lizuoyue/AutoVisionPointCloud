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

def create_autovision_simple_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    _PALETTE = np.array(
        [[90, 120, 150], # Barrier
        [70, 70, 70], # Building
        [0, 0, 142], # Car
        [152, 251, 152], # Terrain
        [0, 60, 100], # Heavy Vehicle
        [119, 11, 32], # Motorcycle
        [128, 64, 128], # Paved Road
        [170, 170, 170], # Pedestrian Area
        [220, 20, 60], # Person
        [250, 170, 30], # Pole Object
        [70, 130, 180], # Sky
        [220, 180, 50], # Unpaved Road
        [107, 142, 35], # Vegetation
        [0, 170, 30], # Water
        [255, 255, 255]], # Ignored Object
    dtype=np.uint8)
    colormap[:15,:] = _PALETTE
    return colormap#.flatten()

if __name__ == '__main__':

    # Day init
    CAM_NAME = 'DEV_000F3102F884'
    SEP = [0, 1341, 2512, 3482, 4700, 5692, 6641, 7702, 8967, 9988]
    FRAME_FROM, FRAME_TO = 77, 10065
    assert((FRAME_TO - FRAME_FROM) == SEP[-1])
    NUM_CLASSES = 15

    host_name = socket.gethostname()
    colormap = create_autovision_simple_label_colormap()

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
        pc_label_count = np.zeros((pc_num, NUM_CLASSES), np.int32)
        for i in tqdm.tqdm(list(range(a, b))):
            fid = i + FRAME_FROM
            sem = np.array(Image.open(sem_path % fid)).reshape((-1))
            d = np.load(mapping_path % fid)
            assert(d['which_pc'] == which_pc)
            pc_idx = d['pc_cam_index']
            pc_sem = sem[d['img_1d_idx']]
            sel = pc_sem < 255
            pc_idx = pc_idx[sel]
            pc_sem = pc_sem[sel]
            for pci, pcs in zip(pc_idx, pc_sem):
                pc_label_count[pci, pcs] += 1

        pc_label = np.argmax(pc_label_count, axis=-1)
        pc_count = np.sum(pc_label_count, axis=-1)
        pc_label[pc_count == 0] = 255

        np.savez_compressed('pc_label_%d.npz' % which_pc, label=pc_label, color=colormap[pc_label])
        break



