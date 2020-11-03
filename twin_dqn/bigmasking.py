import torch

# a = torch.arange(1, 10)
# a = a.reshape(a.shape[0], 1)
# b = torch.tensor([0, 100, 0, 0, 100, 0, 0, 100, 0])
# b = b.reshape(b.shape[0], 1)

# mask = b > a
# a = b * mask + a * ~mask
# print(a)


# # print(a)
# # print(a.shape)

a = torch.tensor([0])
b = torch.tensor([1])
print(a > b)
print(b > a)

if a > b:
    print("breh")
if b > a:
    print("breh2")