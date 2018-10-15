from __future__ import print_function
import torch as t
t.__version__

# 指定tensor的形状
a = t.Tensor(2, 3)
a

# 用list的数据创建tensor
b = t.Tensor([[1,2,3],[4,5,6]])
b

b.tolist() # 把tensor转为list

b_size = b.size()
b_size