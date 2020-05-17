import numpy as np
import glob
import json

for file in sorted(glob.glob('0_day_mapping/*.npz')):
	n1 = np.load(file)['img_1d_idx'].shape[0]
	depth = json.load(open('data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_depth/DEV_000F3102F884/%s.json' % file[-9: -4]))
	depth = np.array(depth['depth_map']['data'])
	n2 = (depth > 0).sum()
	print(int(file[-9: -4]), n1/n2)
