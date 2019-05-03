import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #full connect layer,convert input scalar feature
        #16 * 5 * 5 input, 120 output scalar features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)
    
    def num_flat_features(self):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#it's convinent to get parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn()
out = net(input)
print(out)

#if you need accumalate grad in a mini-batch, you needn't grad_zero
net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
# loss func criterion
criterion = nn.MSELoss()

#compare output and target
loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

#BackProp
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

#Do the backward
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#optimize mannually
learning_rate = 0.01
for f in net.parameters():
    #notice how to get grads for paramaters: paramters->grad->data
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim
#using buidin optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
#Does the update
optimizer.step()

    
