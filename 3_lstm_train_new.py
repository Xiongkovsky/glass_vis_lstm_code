# _*_coding:utf-8 _*_
# @Time    : 2021/8/19 18:25
# @Author  : XCH
# @FileName: 3_bilstm_wzt1bd.py.py
# @Software: PyCharm
#%%导入原始数据

## import data,moudle
import torch
import os
import numpy as np
from torch import nn
from torch import nn
from torch.autograd import Variable
torch.set_default_tensor_type(torch.DoubleTensor)

from torch import nn
from torch.autograd import Variable
torch.set_default_tensor_type(torch.DoubleTensor)

hs = 6
nl = 3

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True) # rnn
        self.reg = nn.Linear(hidden_size * 2, output_size) # 回归
        self.gelu = nn.GELU()

    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = self.gelu(x)
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

def zerolize_ts(ts):

    ts0 = []
    for item in ts:
        if item < -1:
            ts0.append(-1)
        elif item > 1:
            ts0.append(1)
        else:
            ts0.append(item)
    return ts0

print('\n')

npy_data = np.load('sample_v11_shuffled.npy', allow_pickle=True)  # 读取训练数据
print('shape:')
print(npy_data.shape)
print('type:')
print(type(npy_data))


#数据预处理，划分测试集训练集
print('\n')
########################################################################################################################
data_X = []
data_Y = []

for item in npy_data:

    b1 = np.transpose(np.array(item[6]))[0]
    b2 = np.transpose(np.array(item[6]))[1]
    b3 = np.transpose(np.array(item[5]))[2]


    sza = np.transpose(np.array(item[5]))[7]

    b1 = np.where(sza < 8500, b1, -2)
    b2 = np.where(sza < 8500, b2, -2)
    b3 = np.where(sza < 8500, b3, -2)


    input = [list(b1), list(b2), list(b3),]

    # input = list(np.transpose(np.array(item[6]))[:2])
    # input.append(list(np.transpose(np.array(item[5]))[2]))

    data_X.append(input)
    data_Y.append(list(item[10]))
########################################################################################################################
data_X = np.array(data_X).astype(float)
data_Y = np.array(data_Y).astype(float)
# data_Y = np.reshape(data_Y,(-1,1,92))

nd, nb, nt = np.shape(data_X)
ond, onb, ont = np.shape(data_Y)

print(np.shape(data_X))
print(np.shape(data_Y))


print('\n')
print('X nan?')
print(np.isnan(data_X).any())

print('Y nan?')
print(np.isnan(data_Y).any())
print('\n')

num_train = int(len(data_X) * 0.8)
num_test = int(len(data_X) * 0.1)
num_val = int(len(data_X) - num_train - num_test)

train_X = data_X[:num_train]
train_Y = data_Y[:num_train]
test_X = data_X[num_train: num_train + num_test]
test_Y = data_Y[num_train: num_train + num_test]
val_X = data_X[num_train + num_test:]
val_Y = data_Y[num_train + num_test:]

train_X = np.transpose(train_X, (2,0,1))
test_X = np.transpose(test_X, (2,0,1))
train_Y = np.transpose(train_Y, (2,0,1))
test_Y = np.transpose(test_Y, (2,0,1))

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

train_x = train_x.cuda()
train_y = train_y.cuda()
test_x = test_x.cuda()
########################################################################################################################

#%%定义网络

net = lstm_reg(nb, hs, onb, nl)
net = net.cuda()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0.00001, last_epoch=-1)
#%%迭代训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(1, e + 1, loss.data * 10000)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0.00001, last_epoch=-1)
# %%迭代训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(1, e + 1, loss.data * 10000)


min_loss = loss.data * 10000
min_loss_name = (loss.data * 10000).cpu().numpy()

for iband in range(100):

    if iband > 0:

        print('times ' + str(iband + 1))
        net_name = 'bilstm_times_' + str(min_loss_name) + '.pkl'
        net = torch.load(net_name)
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0.00000000001, last_epoch=-1)
    # %%迭代训练
    for e in range(2500):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(2, e + 1, loss.data * 10000)

        if loss.data * 10000 < min_loss:

            last_path = 'bilstm_times_' + str(min_loss_name) + '.pkl'
            if os.path.exists(last_path):
                os.remove(last_path)

            min_loss = loss.data * 10000
            min_loss_name = (loss.data * 10000).cpu().numpy()

            net_name = 'bilstm_times_' + str(min_loss_name) + '.pkl'
            torch.save(net, net_name)

    net_name = 'bilstm_times_' + str(iband) + '_' + str(min_loss_name) + '.pkl'
    torch.save(net, net_name)
