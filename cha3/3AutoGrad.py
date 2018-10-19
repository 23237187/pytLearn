from __future__ import print_function
import torch as t

#在创建tensor的时候指定requires_grad
a = t.randn(3, 4, requires_grad=True)
# 或者
a = t.randn(3, 4).requires_grad_()
# 或者
a = t.randn(3, 4)
a.requires_grad=True
a

b = t.zeros(3, 4).requires_grad_()
b

# 也可写成c = a + b
c = a.add(b)
c

d = c.sum()
d.backward()

d # d还是一个requires_grad=True的tensor,对它的操作需要慎重
d.requires_grad

a.grad

# 此处虽然没有指定c需要求导，但c依赖于a，而a需要求导，
# 因此c的requires_grad属性会自动设为True
a.requires_grad, b.requires_grad, c.requires_grad

# 由用户创建的variable属于叶子节点，对应的grad_fn是None
a.is_leaf, b.is_leaf, c.is_leaf

# c.grad是None, 因c不是叶子节点，它的梯度是用来计算a的梯度
# 所以虽然c.requires_grad = True,但其梯度计算完之后即被释放
c.grad is None

def f(x):
    '''计算y'''
    y = x ** 2 * t.exp(x)
    return y

def gradf(x):
    '''手动求导函数'''
    dx = 2*x*t.exp(x) + x**2*t.exp(x)
    return dx

x = t.randn(3, 4, requires_grad=True)
y = f(x)
y

y.backward(t.ones(y.size())) # gradient形状与y一致
x.grad

# autograd的计算结果与利用公式手动计算的结果一致
gradf(x)

x = t.ones(1)
b = t.randn(1, requires_grad+True)
w = t.randn(1, requires_grad+True)
y = w * x
z = y + b

x.requires_grad, b.requires_grad, w.requires_grad

# 虽然未指定y.requires_grad为True，但由于y依赖于需要求导的w
# 故而y.requires_grad为True
y.requires_grad

x.is_leaf, w.is_leaf, b.is_leaf

y.is_leaf, z.is_leaf

# grad_fn可以查看这个variable的反向传播函数，
# z是add函数的输出，所以它的反向传播函数是AddBackward
z.grad_fn

# next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
# 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
z.grad_fn.next_functions 

# variable的grad_fn对应着和图中的function相对应
z.grad_fn.next_functions[0][0] == y.grad_fn

# 第一个是w，叶子节点，需要求导，梯度是累加的
# 第二个是x，叶子节点，不需要求导，所以为None
y.grad_fn.next_functions

# 叶子节点的grad_fn是None
w.grad_fn,x.grad_fn