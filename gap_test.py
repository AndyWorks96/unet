import torch
import torch.nn as nn

# # target output size of 10x7
# m = nn.AdaptiveMaxPool2d((None, 7))
# # input = torch.randn(1, 64, 10, 9)
# input = torch.randn(14,3)
# output = m(input)
# print(output.shape)


a = torch.rand([4,3,4,4])

b = torch.nn.functional.adaptive_avg_pool2d(a, (1,1))  # 自适应池化，指定池化输出尺寸为 1 * 1


print(b.size())
