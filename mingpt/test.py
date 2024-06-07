import torch

a = torch.arange(5)
b = torch.arange(5)

# print(torch.(a, (5, 1)) < torch.reshape(b, (1, 5)))
print (a.view(1, 5) < b.view(5, 1))
# print (a.view(5, 1) <= b.view(1, 5))
