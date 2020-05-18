import numpy as np
import matplotlib.pyplot as plt

raw = np.loadtxt('../autovision_day_night_data/2018-11-01-Lim-Chu-Kang-Run-3-Night/poses_T_world_camera.txt', skiprows=1)
loc = raw[:, -3:]
loc_p = loc.copy()
loc_p[1:] = loc[:-1]
dist = np.sqrt((loc - loc_p) ** 2).sum(axis=-1)
dist_cum = np.cumsum(dist)
# plt.plot(np.arange(2833, 17267), dist_cum[2833: 17267])
# plt.show()
# dist_cum -= dist_cum[2833]
# dist_cum = dist_cum[2833: 17267]
print(dist_cum[[ 2833,  4613,  6237,  7789,  9516, 11257, 12833, 14367, 15955, 17267]])