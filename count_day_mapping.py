import numpy as np
import glob
import json

for file in sorted(glob.glob('0_day_mapping/*.npz')):
	n0 = np.load(file)['img_1d_idx'].shape[0]
	n1 = np.unique(np.load(file)['img_1d_idx']).shape[0]
	depth = json.load(open('data/2018-10-18-Lim-Chu-Kang-Run-1-Day/gt_depth/%s.json' % file[-9: -4]))
	depth = np.array(depth['depth_map']['data'])
	n2 = (depth > 0).sum()
	print(int(file[-9: -4]), n1/n0, n1/n2)
