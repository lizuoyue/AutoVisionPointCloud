import numpy as np
import torch

def get_mat(theta, t1, t2, t3):
	mat = torch.eye(4)


if __name__ == '__main__':

	torch.set_default_dtype(torch.float64)

	t = '_sample_20'
	world = torch.from_numpy(np.loadtxt(f'../autovision_day_night_data/world_point_cloud/00400{t}.txt'))
	local = torch.from_numpy(np.loadtxt(f'../autovision_day_night_data/local_point_cloud/00400{t}.txt'))

	local = torch.cat([local, torch.ones(local.shape[0], 1)], dim=1)

	zero  = torch.autograd.Variable(torch.Tensor([0]), requires_grad=False)
	one   = torch.autograd.Variable(torch.Tensor([1]), requires_grad=False)
	theta = torch.autograd.Variable(torch.Tensor([0.0745693268]), requires_grad=True)
	trans = torch.autograd.Variable(torch.Tensor([[356117.02294527151], [158608.59657918208], [33.445163726799997]]), requires_grad=True)

	optimizer = torch.optim.Adam([theta, trans], lr=0.001)

	li = []
	for i in range(10000):

		optimizer.zero_grad()

		mat = torch.stack([
			torch.cat([theta.cos(), -theta.sin(), zero, trans[0]]),
			torch.cat([theta.sin(), theta.cos(), zero, trans[1]]),
			torch.cat([zero, zero, one, trans[2]]),
		])
		pred_world = torch.t(torch.matmul(mat, torch.t(local)))
		loss = torch.mean(torch.sum((pred_world - world) ** 2, dim=1))

		loss.backward()
		optimizer.step()

		li.append((loss.detach().numpy() * 10000, theta.detach().numpy()[0]) + tuple(trans.detach().numpy().flatten()))

		if i % 100 == 0:
			print(i, min(li))




