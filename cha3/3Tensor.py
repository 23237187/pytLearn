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

b.numel() # b中元素总个数，2*3，等价于b.nelement()

# 创建一个和b形状一样的tensor
c = t.Tensor(b_size)
# 创建一个元素为2和3的tensor
d = t.Tensor((2,3))
c, d

c.shape

t.ones(2, 3)
t.zeros(2, 3)
t.arange(1, 6, 2)
t.linspace(1, 10, 3)
t.randn(2, 3, device=t.device('cpu'))
t.randperm(5)
t.eye(2, 3, dtype=t.int)

scalar = t.tensor(3.14159)
print('scalar: %s, shape of sclar: %s' %(scalar, scalar.shape))

vector = t.tensor([1, 2])
print('vector: %s, shape of vector: %s' %(vector, vector.shape))

tensor = t.Tensor(1, 2)
tensor.shape

matrix = t.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
matrix, matrix.shape

t.tensor([[0.11111, 0.22222, 0.33333]],
        dtype=t.float64,
        device=t.device("cpu"))

empty_tensor = t.tensor([])
empty_tensor.shape

a = t.arange(0, 6)
a.view(2, 3)

b = a.view(-1, 3) # 当某一维为-1的时候，会自动计算它的大小
b.shape

b.unsqueeze(1) # 注意形状，在第1维（下标从0开始）上增加“１” 
b[:, None].shape
b

b.unsqueeze(-2) # -2表示倒数第二个维度

c = b.view(1, 1, 1, 2, 3)
c.squeeze(0) # 压缩第0维的“１”

a[1] = 100
b # a修改，b作为view之后的，也会跟着修改

b.resize_(1, 3)
b

b.resize_(3,3)
b

a = t.randn(3, 4)
a

a[0] # 第0行(下标从0开始)

a[:, 0] # 第0列

a[0][2] # 第0行第2个元素，等价于a[0, 2]

a[0][-1] # 第0行最后一个元素

a[:2] # 前两行

a[:2, 0:2] # 前两行，第0,1列

print(a[0:1, :2]) # 第0行，前两列 
print(a[0, :2]) # 注意两者的区别：形状不同

# None类似于np.newaxis, 为a新增了一个轴
# 等价于a.view(1, a.shape[0], a.shape[1])
a[None].shape
a[:, None, :].shape
a[:, None,:, None, None].shape

a > 1 # 返回一个ByteTensor

a[a>1] # 等价于a.masked_select(a>1)
# 选择结果与原tensor不共享内存空间

a[t.LongTensor([0, 1])] # 第0行和第1行

a = t.arange(0, 16).view(4, 4)
a

# 选取对角线的元素
index = t.LongTensor([[0, 1, 2, 3]])
a.gather(0, index)

# 选取反对角线上的元素
index = t.LongTensor([[3,2,1,0]]).t()
a.gather(1, index)

# 选取反对角线上的元素，注意与上面的不同
index = t.LongTensor([[3,2,1,0]])
a.gather(0, index)

# 选取两个对角线上的元素
index = t.LongTensor([[0,1,2,3],[3,2,1,0]]).t()
b = a.gather(1, index)
b

# 把两个对角线元素放回去到指定位置
c = t.zeros(4,4)
c.scatter_(1, index, b)

a[0, 0] #依旧是tensor

a[0,0].item # python float

d = a[0:1, 0:1, None]
print(d.shape)
d.item() # 只包含一个元素的tensor即可调用tensor.item,与形状无关

# a[0].item()  ->
# raise ValueError: only one element tensors can be converted to Python scalars

x = t.arange(0, 27).view(3,3,3)
x

x[[1,2], [1,2], [2, 0]]

x[[2, 1, 0], [0], [1]]

x[[0, 2], ...]

# 设置默认tensor，注意参数是字符串
t.set_default_dtype('torch.DoubleTensor')

a = t.Tensor(2, 3)
a.dtype # 现在a是DoubleTensor,dtype是float64

# 恢复之前的默认设置
t.set_default_dtype('torch.FloatTensor')

c = a.type_as(b)
c

a.new(2, 3) # 等价于torch.DoubleTensor(2,3)，建议使用a.new_tensor

t.zeros_like(a, dtype=t.int) #可以修改某些属性

t.rand_like(a)

a.new_ones(4, 5, dtype=t.int)

a.new_tensor([3, 4])

a = t.arange(0, 6).view(2, 3)
t.cos(a)

a % 3  # 等价于t.fmod(a, 3)

a ** 2  # 等价于t.pow(a, 2)

# 取a中的每一个元素与3相比较大的一个 (小于3的截断成3)
print(a)
t.clamp(a, min=3)

b = a.sin_() # 效果同 a = a.sin();b=a ,但是更高效节省显存
a

b = t.ones(2, 3)
b.sum(dim=0, keepdim=True)

# keepdim=False，不保留维度"1"，注意形状
b.sum(dim=0, keepdim=False)

b.sum(dim=1)

a = t.arange(0, 6).view(2, 3)
print(a)
a.cumsum(dim=1)

a = t.linspace(0, 15, 6).view(2, 3)
a

b = t.linspace(15, 0, 6).view(2, 3)
b

a[a>b]  # a中大于b的元素

t.max(a)

t.max(a, dim=1)
# 第一个返回值的15和6分别表示第0行和第1行最大的元素
# 第二个返回值的0和0表示上述最大的数是该行第0个元素

t.max(a, b)

t.clamp(a, min=10)

b = a.t()
b.is_contiguous()

b.contiguous()

import numpy as np
a = np.ones([2, 3], dtype=np.float32)
a

b = t.from_numpy(a)
b

b = t.Tensor(a) # 也可以直接将numpy对象传入Tensor
b

a[0,1]=100
b

c = b.numpy()  # a, b, c三个对象共享内存
c

a = np.ones([2, 3])
# 注意和上面的a的区别（dtype不是float32）
a.dtype

b = t.Tensor(a) # 此处进行拷贝，不共享内存
b.dtype

c = t.from_numpy(a) # 注意c的类型（DoubleTensor）
c

a[0, 1] = 100
b # b与a不共享内存，所以即使a改变了，b也不变

c # c与a共享内存

tensor = t.tensor(a)
tensor[0,0]=0
a

a = t.ones(3, 2)
b = t.zeros(2, 3, 1)

# 自动广播法则
# 第一步：a是2维,b是3维，所以先在较小的a前面补1 ，
#               即：a.unsqueeze(0)，a的形状变成（1，3，2），b的形状是（2，3，1）,
# 第二步:   a和b在第一维和第三维形状不一样，其中一个为1 ，
#               可以利用广播法则扩展，两个形状都变成了（2，3，2）
a+b

# 手动广播法则
# 或者 a.view(1,3,2).expand(2,3,2)+b.expand(2,3,2)
a[None].expand(2, 3, 2) + b.expand(2, 3, 2)

# expand不会占用额外空间，只会在需要的时候才扩充，可极大节省内存
e = a.unsqueeze(0).expand(10000000000000, 3,2)

a = t.arange(0, 6)
a.storage()

b = a.view(2, 3)
b.storage()

# 一个对象的id值可以看作它在内存中的地址
# storage的内存地址一样，即是同一个storage
id(b.storage()) == id(a.stroage())

# a改变，b也随之改变，因为他们共享storage
a[1] = 100
b

c = a[2:]
c.storage()

c.data_ptr(), a.data_ptr() # data_ptr返回tensor首元素的内存地址
# 可以看出相差8，这是因为2*4=8--相差两个元素，每个元素占4个字节(float)

d = t.Tensor(c.storage())
d[0] = 6666
b

# 下面４个tensor共享storage
id(a.storage()) == id(b.storage()) == id(c.storage()) == id(d.storage())

a.storage_offset(), c.storage_offset(), d.storage_offset()

e = b[::2,::2]# 隔2行/列取一个元素
id(e.storage()) == id(a.storage())

b.stride(), e.stride()
e.is_contiguous()

a = t.randn(3, 4)
a.device

if t.cuda.is_available():
    a = t.randn(3, 4, device=t.device("cuda:1"))
     # 等价于
    # a.t.randn(3,4).cuda(1)
    # 但是前者更快
    a.device

device = t.device('cpu')
a.to(device)

if t.cuda.is_available():
    a = a.cuda(1) # 把a转为GPU1上的tensor,
    t.save(a, 'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')
    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})


def for_loop_add(x, y):
    result = []
    for i, j in zip(x, y):
        result.append(i + j)
    return t.Tensor(result)

x = t.zeros(100)
y = t.ones(100)
%timeit -n  10  for_loop_add(x,y)
%timeit -n  10  x + y

a = t.arange(0, 20000000)
print(a[-1], a[-2]) # 32bit的IntTensor精度有限导致溢出
b = t.LongTenor()
t.arangr(0, 20000000, out=b) # 64bit的LongTensor不会溢出
b[-1],b[-2]

a = t.randn(2, 3)
a

t.set_printoptions(precision=10)
a

import torch as t
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display

device = t.device('cpu')

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000)

def get_fake_data(batch_size=8):
     ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
     x = t.rand(batch_size, device=device) * 5
     y = x * 2 + 3 + t.randn(batch_size, 1, device=device)
     return x, y

# 来看看产生的x-y分布
x, y = get_fake_data(batch_size=16)
plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())

# 随机初始化参数
w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr = 0.02  # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=4)

    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2 # 均方误差
    loss = loss.mean()

    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)
    
    if ii%50 ==0:
       
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy()) # predicted
        
        x2, y2 = get_fake_data(batch_size=32) 
        plt.scatter(x2.numpy(), y2.numpy()) # true data
        
        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)
        
print('w: ', w.item(), 'b: ', b.item())