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

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)

params = list(net.parameters())
print(len(params))

for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

input = t.randn(1, 1, 32, 32)
out = net(input)
out.size()

net.zero_grad()  # 所有参数的梯度清零
out.backward(t.ones(1, 10)) # 反向传播

output = net(input)
target = t.arange(0, 10, out=t.FloatTensor()).view(1, 10)
criterion = nn.MSELoss()
loss = criterion(output, target)
loss

# 运行.backward，观察调用之前和调用之后的grad
net.zero_grad()
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad)

import torch.optim as optim
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练过程中
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()

# 计算损失
output = net(input)
loss = criterion(output, target)

#反向传播
loss.backward()

#更新参数
optimizer.step()

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(), # 转为Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 归一化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='/home/yang/tmp/CIFAR10',
    train=True,
    download=True,
    transform=transform
)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# 测试集
testset = tv.datasets.CIFAR10(
    root='/home/yang/tmp/CIFAR10',
    train=False,
    download=True,
    transform=transform
)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label])

show((data + 1) / 2).resize((100, 100))

dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
print(''.join('%11s'% classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400, 100))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)


from torch import optim
criterion = nn.CrossEntropyLoss()
# criterion.cuda()
# net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9).cuda()

import time

t.set_num_threads(8)
start_time = time.time()
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 输入数据
        inputs, labels = data

        # inputs = inputs.cuda()
        # labels = labels.cuda()

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数 
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print("--- %s seconds ---" % (time.time() - start_time))
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
print('实际的label: ', ' '.join(\
            '%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100))

# 计算图片在每个类别上的分数
outputs = net(images)
# 得分最高的那个类
_, predicted = t.max(outputs.data, 1)
print('预测结果: ', ' '.join('%5s'\
            % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0

# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

net.to(device)
images = images.to(device)
labels = labels.to(device)
outputs = net(images)
loss = criterion(outputs, labels)

loss
