import numpy as np
import glob
import os

files = sorted(glob.glob('2_night_pc_label_add/*.npz'))
for file in files[0:1]:
	a, b = os.path.basename(file).split('_')
	label = np.load(f'2_night_pc_label/{int(a)-1}_{b}')['label']
	new_label = np.load(file)['label']
	label[label == 15] = 255
	new_label[new_label == 15] = 255
	assert(label.shape == new_label.shape)
	print(file, (new_label[label == 255] < 255).sum())
