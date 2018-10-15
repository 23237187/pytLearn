from __future__ import print_function
import torch as t
t.__version__

# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)

x = t.Tensor([[1, 2], [3, 4]])
x

# 使用[0,1]均匀分布随机初始化二维数组
x = t.rand(5, 3)
x

print(x.size())
x.size()[1], x.size(1)

y = t.rand(5, 3)
# 加法的第一种写法
x + y
# 加法的第二种写法
t.add(x, y)
# 加法的第三种写法：指定加法结果的输出目标为result
result = t.Tensor(5, 3)
t.add(x, y, out=result)
result

print('最初y')
print(y)

print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)

print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)

# Tensor的选取操作与Numpy类似
x[:, 1]

a = t.ones(5) # 新建一个全1的Tensor
a

b = a.numpy() # Tensor -> Numpy
b

import numpy as np
a = np.ones(5)
b = t.from_numpy(a) # Numpy->Tensor
print(a)
print(b)

scalar = b[0]
scalar

scalar.size()  #0-dim

scalar.item() # 使用scalar.item()能从中取出python对象的数值

tensor = t.tensor([2]) # 注意和scalar的区别
tensor, scalar

tensor.size(), scalar.size()

# 只有一个元素的tensor也可以调用`tensor.item()`
tensor.item(), scalar.item()

tensor = t.tensor([3, 4]) # 新建一个包含 3，4 两个元素的tensor
scalar = t.tensor(3)
scalar

old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
old_tensor, new_tensor

new_tensor = old_tensor.detach()
new_tensor[0] = 1111
old_tensor, new_tensor

# 在不支持CUDA的机器下，下一步还是在CPU上运行
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
z = x + y

# 为tensor设置 requires_grad 标识，代表着需要求导数
# pytorch 会自动调用autograd 记录操作
x = t.ones(2, 2, requires_grad=True)
# 上一步等价于
# x = t.ones(2,2)
# x.requires_grad = True
x

y = x.sum()
y

y.grad_fn

y.backward()  # 反向传播,计算梯度

# y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# 每个值的梯度都为1
x.grad

y.backward()
x.grad

# 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
x.grad.data.zero_()

y.backward()
x.grad