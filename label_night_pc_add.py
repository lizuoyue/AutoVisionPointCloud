import numpy as np
import glob
import os

files = sorted(glob.glob('2_night_pc_label_add/*.npz'))
for file in files:
	a, b = os.path.basename(file).split('_')
	a = int(a) - 1
	label = np.load(f'2_night_pc_label/{a}_{b}')['label']
	new_label = np.load(file)['label']
	print(file, (new_label[label == 255] < 255).sum())
