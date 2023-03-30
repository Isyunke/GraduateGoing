import torch

#张量：不初始化（都是0）
print("构造张量：不初始化")
x = torch.empty((5,3))
print(x)

#随机初始化
print("构造张量：随机初始化")
x = torch.rand((5,3))
print(x)

#全0矩阵
print("构造张量：全0")
x = torch.zeros((5,3), dtype=torch.long)
print(x)

#从列表构造张量
print("构造张量：从列表构造张量")
x = torch.tensor([1, 2, 3])
print(x)

#从已有张量构造张量：y会继承x的属性
print("构造张量：从已有张量构造张量")
y = x.new_ones((5, 3), dtype=torch.long)
print(y)
y = torch.rand_like(x, dtype=torch.float)
print(y)

#获取维度信息
print("获取维度信息")
x = torch.empty((5,3))
print(x.shape)

#基本运算
print("基本运算")
x = torch.rand((5,3))
y = torch.rand_like(x)
print(x+y)
print(x.add(y))

#带有_的函数会改变调用此函数的对象
x.add_(y)
print(x)

#改变形状
print("改变形状")
x = torch.rand(4,4)
y = x.view(-1, 8)
print(x.size(), y.size())

#如果只有一个元素，获取元素
print("获取元素")
x = torch.rand((1,1))
print(x)
print(x.item())

#%%
#自动微分
import torch
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
#out是标量，因此反向传播无需指定参数
out.backward()
#打印梯度
print("打印梯度")
print(x.grad)

#雅克比向量积
print("雅克比向量积")
x = torch.rand(3, requires_grad=True)
y = x * 2
print(y)
#反向传播的参数是权重
#I = w1*y1 + w2*y2 + w3 * y3
#这里是I对x的各个分量求导
w = torch.tensor([1,2,3], dtype=torch.float)
y.backward(w)
print(x.grad)

#%%
#神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

#神经网络需要定义两个函数
#分别是构造函数，前向传播
#自定义的神经网络需要继承nn.Module
class Net(nn.Module):

    #构造函数
    def __init__(self):
        super(Net, self).__init__()
        #卷积层三个参数：in_channel, out_channels, 5*5 kernal
        self.con1 = nn.Conv2d(3, 116, 5)
        self.con2 = nn.Conv2d(116, 100, 5)
        #全连接层两个参数：in_channels, out_channels
        self.fc1 = nn.Linear(100 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)

    #前向传播
    def forward(self, input):
        #卷积 --> 激活函数（Relu) --> 池化
        x = self.con1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #重复上述过程
        x = self.con2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #展平
        x = x.view(-1, self.num_flat_features(x))

        #全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)

        return x


    #展平
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features = num_features * i
        return num_features

#测试神经网络的输入与输出
# net = Net()
# input = torch.randn((1, 3, 32, 32))
# out = net(input)
# print(out.size())

#%%
#训练一个神经网络的步骤
#1. 定义函数集：确定神经网络的结构
#2. 定义损失函数
#3. 寻找最优函数
#3.1 设置优化器种类（随机梯度下降，Adam等）
#3.2 前向传播
#3.3 计算梯度：反向传播
#3.4 更新参数
#3.5 梯度清零

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)

#开始训练
net = Net().to(device)
#损失函数
criterion = nn.CrossEntropyLoss()
#优化器
optimizer = optim.Adam(net.parameters(), lr = 0.001)
#定义变换
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

#训练集和测试集
transet = torchvision.datasets.CIFAR10(root = './data', train = True, download=True, transform = transform)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform = transform)

trainLoader = Data.DataLoader(dataset = transet, batch_size = 400, shuffle = True)
testLoader = Data.DataLoader(dataset = testset, batch_size = 400, shuffle = False)

num_epoches = 15


#开始训练
for epoch in range(num_epoches):
    correct = 0
    total = 0
    run_loss = 0.0
    for i, data in enumerate(trainLoader):
        input, label = data
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(input)
        lossValue = criterion(outputs, label)
        lossValue.backward()
        optimizer.step()

        run_loss += lossValue.item()

        num = 20

        if i % num == num - 1:
            print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, run_loss / num))
            run_loss = 0

        #训练集准确率
        _, pred = outputs.max(1)
        correct += (pred == label).sum().item()
        total += label.size()[0]

    print("训练集准确率：", correct / total)

print("finished training!")

#预测
correct = 0
total = 0
#预测的时候，无需梯度更新
with torch.no_grad():
    for data in testLoader:
        input, label = data
        input, label = input.to(device), label.to(device)
        outputs = net(input)
        _, pre = torch.max(outputs, 1)
        total += label.size(0)
        correct += (pre == label).sum().item()

print("准确率：", correct / total)
