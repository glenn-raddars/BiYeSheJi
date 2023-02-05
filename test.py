import torch
import torch.nn as nn

# x = torch.FloatTensor([1, 2, 3, 4, 5, 6]).view(1, -1, 1, 1)
# print(x)
# conv = nn.Conv2d(in_channels=6, out_channels=18, kernel_size=1, stride=1, padding=0, groups=2, bias=False)
# print(conv.weight.data.size())
# conv.weight.data = torch.arange(1, 55).view(conv.weight.data.size()).float()
# print(conv.weight.data)

# output = conv(x)
# print(output, output.size())

# --------------------------------------------------

# x = torch.tensor([[1, 2],
#                   [3, 4]], dtype=float)
# print("___________________________________")
# print(x.mean(1).mean(0))

#------------------------------------------------------

x = torch.ones([1, 3, 2, 2])
x = torch.reshape(x, [-1, 2])
print(x)