import torch

a = torch.zeros([8,8])
print(a)
a[2:3,2:3]=1
print(a)