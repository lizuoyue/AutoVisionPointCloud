import numpy as np
import glob, tqdm, os
from PIL import Image
from utils import *

if __name__ == '__main__':

	colormap = create_autovision_simple_label_colormap()

	img_path = 'data/2018-11-01-Lim-Chu-Kang-Run-3-Night/img_fisheye/DEV_000F3102F884'
	sem_path = '3_night_sem'
	sem_path_add = '3_night_sem_add'

	save_path = 'vis'
	save_alpha_path = 'vis_alpha'

	os.system(f'mkdir {save_path} {save_alpha_path}')

	for i in tqdm.tqdm(list(range(20000))):

		sem_add = f'{sem_path_add}/%05d.png' % i
		sem = f'{sem_path}/%05d.png' % i
		img = f'{img_path}/%05d.png' % i

		if os.path.isfile(sem_add):
			sem = np.array(Image.open(sem_add).convert('P'))
		elif os.path.isfile(sem):
			sem = np.array(Image.open(sem).convert('P'))
		else:
			continue
		img = np.array(Image.open(img).convert('RGB'))
		valid = (sem < 15)
		sem = colormap[sem.flatten()].reshape(img.shape[:2] + (3,))
		Image.fromarray(np.vstack([img, sem])).save(f'{save_path}/%05d.png' % i)
		img[valid] = (img[valid] * 0.7 + sem[valid] * 0.3).astype(np.uint8)
		Image.fromarray(img).save(f'{save_alpha_path}/%05d.png' % i)
		
		break
