# gradient descent python using torch

from matplotlib import pyplot as plt
import torch


n = 100
x = torch.ones(n,2)
x[:, 0].uniform_(-1, 1)
a = torch.tensor([ 2., 3])
y = x@a


plt.scatter(x[:, 0], y)
plt.show()