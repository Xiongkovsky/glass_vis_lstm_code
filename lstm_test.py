# _*_coding:utf-8 _*_
# @Time    : 2021/8/26 10:43
# @Author  : XCH
# @FileName: 3_lstm_test.py
# @Software: PyCharm

import os
import torch
import glob2
import random

from torch import nn
from math import sqrt
from tqdm import tqdm
from torch.autograd import Variable
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error, r2_score
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import numpy as np
import cartopy.crs as ccrs
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

torch.set_default_tensor_type(torch.DoubleTensor)


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, num_layers=2):
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


def density_scatter(x, y, ax = None, sort = True, bins = 20, **kwargs )   :

    if ax is None :
        fig, ax = plt.subplots()

    min(x) - ((min(x) + max(x))/2 - min(x))

    data, x_e, y_e = np.histogram2d( x, y, bins=bins, range=[[1.5*min(x)-0.5*max(x), 1.5*max(x)-0.5*min(x)], [1.5*min(y)-0.5*max(y),  1.5*max(y)-0.5*min(y)]])
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T,
                 method="splinef2d", bounds_error=False )

    # Sort the points by density, so that the densest points are plotted last

    # ax.scatter(x, y, color='blue', s=1)

    if sort:

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, **kwargs, s = 1)

    # plt.plot([min(x), max(x)], [min(y), max(y)], color='black')
    xl = np.linspace(min(x), max(x), 100)
    #生成每个x对应的baiy
    k, b = np.polyfit(x, y, 1)
    yl = k*xl+b
    #画直线du
    plt.plot(xl, yl, c='red')
    plt.plot(xl, xl, c='black')
    k=('%.2f' % k)
    b=('%.2f' % b)
    linetxt = 'y=' + str(k) + '*x+'  + str(b)
    plt.text(min(x), max(y), linetxt)
    rmse = sqrt(mean_squared_error(x, y))
    r2 = r2_score(x, y)
    bias = x.mean() - y.mean()
    title = 'rmse=' + str(round(rmse, 4)) + ' r2=' + str(round(r2, 4))

    plt.title(title)
    return ax


def zerolize_ts(ts):

    ts0 = []
    for item in ts:
        if item < 0:
            ts0.append(0)
        else:
            ts0.append(item)
    return ts0


netname = 'bilstm_times_6.466398204884499.pkl'
shuffled_npy_name = 'sample_v11_shuffled.npy'

# rootpath = 'E:\\PROG\\FAPAR_LSTM_MHLAI'
# os.chdir(rootpath)
npy_data = np.load(shuffled_npy_name, allow_pickle=True)  # 读取训练数据
print(npy_data.shape)
print(type(npy_data))

#数据预处理，划分测试集训练集
########################################################################################################################
data_X = []
data_Y = []

for item in npy_data:

    # input = list(np.transpose(np.array(item[6]))[:2])
    # input.append(list(np.transpose(np.array(item[5]))[2]))

    b1 = np.transpose(np.array(item[6]))[0]
    b2 = np.transpose(np.array(item[6]))[1]
    b3 = np.transpose(np.array(item[5]))[2]

    sza = np.transpose(np.array(item[5]))[7]

    b1 = np.where(sza < 8500, b1, -2)
    b2 = np.where(sza < 8500, b2, -2)
    b3 = np.where(sza < 8500, b3, -2)


    input = [list(b1), list(b2), list(b3),]

    data_X.append(input)
    data_Y.append(list(item[10]))
########################################################################################################################

data_X = np.array(data_X).astype(float)
data_Y = np.array(data_Y).astype(float)

print(np.shape(data_X))
print(np.shape(data_Y))

print('X nan?')
print(np.isnan(data_X).any())
print('Y nan?')
print(np.isnan(data_Y).any())

num_all = len(data_X)
num_train = int(len(data_X) * 0.8)
num_test = int(len(data_X) * 0.1)
num_val = int(len(data_X) - num_train - num_test)

train_X = data_X[:num_train]
train_Y = data_Y[:num_train]
test_X = data_X[num_train: num_train + num_test]
test_Y = data_Y[num_train: num_train + num_test]
val_X = data_X[num_train + num_test:]
val_Y = data_Y[num_train + num_test:]

data_X = np.transpose(data_X, (2,0,1))
train_X = np.transpose(train_X, (2,0,1))
test_X = np.transpose(test_X, (2,0,1))
val_X = np.transpose(val_X, (2,0,1))

data_Y = np.transpose(data_Y, (2,0,1))
train_Y = np.transpose(train_Y, (2,0,1))
test_Y = np.transpose(test_Y, (2,0,1))
val_Y = np.transpose(val_Y, (2,0,1))

data_x = torch.from_numpy(data_X)
train_x = torch.from_numpy(train_X)
test_x = torch.from_numpy(test_X)
val_x = torch.from_numpy(val_X)

# netnamepath = 'pkl\\*_' + str(iband) + '_*.pkl'
# netnamelist  =glob2.glob(netnamepath)

net1 = torch.load(netname, map_location=torch.device('cpu'))
net1 = net1.eval()

pred_data = net1(data_x)
pred_data = pred_data.detach().numpy()

pred_train = net1(train_x)
pred_train = pred_train.detach().numpy()

pred_test = net1(test_x)
pred_test = pred_test.detach().numpy()

pred_val = net1(val_x)
pred_val = pred_val.detach().numpy()

rootpath = os.getcwd()

for ivi, vi_name in enumerate(['ndvi', 'evi']):

    flodpath = rootpath + '/' + netname + '_lstm_test_pic/' + vi_name
    # 判断文件夹是否存在，不存在则创建文件夹
    if not os.path.exists(flodpath):
        os.makedirs(flodpath)
    os.chdir(flodpath)
    # ##################################################################################
    plt.rcParams['figure.figsize'] = (8, 8)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    x = pred_train[:, :, ivi].flatten()
    y = train_Y[:, :, ivi].flatten()

    density_scatter(y, x, bins=[30, 30])
    plt.savefig('1_bilstm_train.jpg')
    plt.close('all')

    x = pred_test[:, :, ivi].flatten()
    y = test_Y[:, :, ivi].flatten()

    density_scatter(y, x, bins=[30, 30])
    plt.savefig('2_bilstm_test.jpg')
    plt.close('all')

    x = pred_val[:, :, ivi].flatten()
    y = val_Y[:, :, ivi].flatten()

    density_scatter(y, x, bins=[30, 30])
    plt.savefig('3_bilstm_val.jpg')
    plt.close('all')
    ########################################################################################################################
    #随机展示100个预测与输入序列
    x = pred_data[:, :, ivi]
    y = data_Y[:, :, ivi]

    resultList=random.sample(range(0, num_all), 100)
    plt.rcParams['figure.figsize'] = (20.0, 30.0)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    for i, p in enumerate(resultList):

        ax1 = plt.subplot(20, 5, i + 1)
        ax2 = ax1.twinx()
        # ax2.ylim(0, 1)
        ax2.set_ylim(-0.5, 1)

        rmse = mean_squared_error(x[:, p], y[:, p])
        r2 = r2_score(x[:, p], y[:, p])
        bias = x[:, p].mean() - y[:, p].mean()

        type = npy_data[p][1]
        hvrc = npy_data[p][2]
        lat = npy_data[p][4][0]
        lon = npy_data[p][4][1]

        title = 'id:' + str(npy_data[p][0]) + ' tp:' + str(int(type)) + ' ll:' + str(
            ('%.1f' % lat)) + ' ' + str(('%.1f' % lon) + ' hvrc:' + str(hvrc))
        plt.title(title, fontsize=8)

        mh_lai_ts = np.array(npy_data[p][7])/10
        ndvi_ts = npy_data[p][9][ivi]
        mod_ndvi_ts = np.transpose(npy_data[p][8])[ivi]/10000
        mod_qa_ts = np.transpose(npy_data[p][8])[3]

        ax1.set_ylim(0, 10)
        ax1.plot(range(1, 730, 8), mh_lai_ts, color='green', linestyle='-')
        ax2.plot(range(1, 730, 8), x[:, p], 'r', label='prediction',  linestyle='-')
        ax2.plot(range(1, 730, 8), y[:, p], 'black', label='real',  linestyle='--')
        ax2.plot(range(1, 730, 8), ndvi_ts, color='gray', linestyle='--')
        # ax2.plot(range(1, 730, 16), mod_ndvi_ts, color='blue', linestyle='--')

        ax3 = ax2.twinx()
        ax3.set_ylim(-0.5, 1)
        ax2.set_yticks([])
        scat_sym_list = ['*', 'o', '+', 'x']
        for itim in range(1, 730, 16):
            point = ax3.scatter(itim, mod_ndvi_ts[int((itim - 1) / 16)], c='black',
                                marker=scat_sym_list[mod_qa_ts[int((itim - 1) / 16)]], s=8)

    plt.tight_layout()
    plt.savefig('4_bilstm_ts.jpg')
    plt.show()
    plt.close('all')
    ########################################################################################################################
    # 展示预测与输入差别最大的100个序列
    x = pred_data[:, :, ivi]
    y = data_Y[:, :, ivi]


    plt.rcParams['figure.figsize'] = (20.0, 30.0)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    rmse_list = []

    for i in range(num_all):

        rmse = mean_squared_error(x[:, i], y[:, i])
        rmse_list.append([i, rmse])

    rmse_list = np.array(rmse_list)
    rmse_list = rmse_list[np.argsort(rmse_list[:,1])]

    for i, id_rmse in enumerate(rmse_list[-100:]):

        p = int(id_rmse[0])

        ax1 = plt.subplot(20, 5, i + 1)
        ax2 = ax1.twinx()
        ax2.set_ylim(-0.5, 1)

        # ax2.set_ylim([-1, 1])
        # ax2.ylim(0, 1)

        rmse = mean_squared_error(x[:, p], y[:, p])
        r2 = r2_score(x[:, p], y[:, p])
        bias = x[:, p].mean() - y[:, p].mean()

        type = npy_data[p][1]

        lat = npy_data[p][4][0]
        lon = npy_data[p][4][1]

        hvrc = npy_data[p][2]


        mh_lai_ts = np.array(npy_data[p][7])/10
        ndvi_ts = npy_data[p][9][ivi]
        mod_ndvi_ts = np.transpose(npy_data[p][8])[ivi]/10000
        mod_qa_ts = np.transpose(npy_data[p][8])[3]

        title = 'id:' + str(npy_data[p][0]) + ' tp:' + str(int(type)) + ' ll:' + str(
            ('%.1f' % lat)) + ' ' + str(('%.1f' % lon) + ' hvrc:' + str(hvrc))
        plt.title(title, fontsize=8)
        ax1.set_ylim(0, 10)
        ax1.plot(range(1, 730, 8), mh_lai_ts, color='green', linestyle='-')
        ax2.plot(range(1, 730, 8), x[:, p], 'r', label='prediction', linestyle='-')
        ax2.plot(range(1, 730, 8), y[:, p], 'black', label='real', linestyle='--')
        ax2.plot(range(1, 730, 8), ndvi_ts, color='gray', linestyle='--')
        # ax2.plot(range(1, 730, 16), mod_ndvi_ts, color='blue', linestyle='--')

        ax3 = ax2.twinx()
        ax3.set_ylim(-0.5, 1)
        ax2.set_yticks([])
        scat_sym_list = ['*', 'o', '+', 'x']
        for itim in range(1, 730, 16):
            point = ax3.scatter(itim, mod_ndvi_ts[int((itim - 1) / 16)], c='black',
                                marker=scat_sym_list[mod_qa_ts[int((itim - 1) / 16)]], s=8)

    plt.tight_layout()
    plt.savefig('5_bilstm_wrost100_ts.jpg')
    plt.show()
    plt.close('all')

    ########################################################################################################################
    #绘制最差100的位置

    values = range(16)
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    # 设置plt参数，绘制图像
    map_proj = ccrs.PlateCarree()
    data_proj = ccrs.PlateCarree()
    plt.rcParams['figure.figsize'] = (20.0, 10.0)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    '''ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))'''
    ax.stock_img()
    ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=ccrs.PlateCarree())  # 设置大刻度和小刻度
    ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=ccrs.PlateCarree())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(15))
    ax.xaxis.set_major_formatter(LongitudeFormatter())  # 利用Formatter格式化刻度标签
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    for id_rmse in rmse_list[-100:]:
        p = int(id_rmse[0])
        lat = npy_data[p][4][0]
        lon = npy_data[p][4][1]

        ax.plot(lon, lat, marker='.', color='indigo', transform=data_proj, ms=10)

    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0

    plt.savefig('6_wrost100_pp.jpg')
    plt.close('all')

    ########################################################################################################################
    #绘制直方图
    plt.figure(figsize=(20, 8), dpi=100)
    ndvi_data = pred_data.flatten()
    bad_num = 0
    for it in range(len(ndvi_data)):
        if ndvi_data[it] < -1 :
            ndvi_data[it] = -1
            bad_num = bad_num + 1
        elif ndvi_data[it] > 1:
            ndvi_data[it] = 1
            bad_num = bad_num + 1

    print(bad_num)

    plt.figure(figsize=(20, 8), dpi=100)

    plt.hist(ndvi_data, 200)
    plt.xlim([-1, 1])
    plt.ylim([0, 100000])

    plt.savefig('7_pred_hist.jpg')
    plt.close('all')

    ########################################################################################################################
    # %%
    # 随机展示100个预测与输入序列 hv
    x = pred_data[:, :, ivi]
    y = data_Y[:, :, ivi]

    # hv_list = [[17, 8], [19, 8], [13, 4], [28, 11], [8, 6], [26, 6]]
    hv_list = [[19, 8], [19, 9], [17, 8], [13, 4], [28, 11], [8, 6], [26, 6], [12, 8], [14, 9]]

    for h_v in hv_list:


        h = h_v[0]
        v = h_v[1]
        hv_id_list = []

        for i in range(len(npy_data)):
            if  npy_data[i][2][0] == h and npy_data[i][2][1] == v:

            # if npy_data[i][2][0] == h:
            #     if npy_data[i][2][3] < 2400 and (
            #             (npy_data[i][2][1] == 8 and npy_data[i][2][2] > 2400) or (npy_data[i][2][1] == 9 and npy_data[i][2][2] < 2400)):
            # if npy_data[i][2][1] == v:
                    hv_id_list.append(i)

        # final_id_list = [item for item in npy_data if item[0] in hv_id_list]

        if len(hv_id_list) > 100:
            resultList = random.sample(range(0, len(hv_id_list)), 100)
        else:
            resultList = range(len(hv_id_list))

        plt.rcParams['figure.figsize'] = (20.0, 30.0)  # 设置figure_size尺寸
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 100  # 分辨率

        for i, pr in enumerate(resultList):

            p = hv_id_list[pr]
            ax1 = plt.subplot(20, 5, i + 1)
            ax2 = ax1.twinx()
            # ax2.ylim(0, 1)
            ax2.set_ylim(-0.5, 1)

            rmse = mean_squared_error(x[:, p], y[:, p])
            r2 = r2_score(x[:, p], y[:, p])
            bias = x[:, p].mean() - y[:, p].mean()

            type = npy_data[p][1]

            lat = npy_data[p][4][0]
            lon = npy_data[p][4][1]
            title = 'num=' + str(npy_data[p][0]) + ' type:' + str(int(type)) + ' lat:' + str(
                ('%.2f' % lat)) + ' lon:' + str(('%.2f' % lon))
            plt.title(title)

            mh_lai_ts = np.array(npy_data[p][7]) / 10
            ndvi_ts = npy_data[p][9][ivi]
            mod_ndvi_ts = np.transpose(npy_data[p][8])[ivi] / 10000
            mod_qa_ts = np.transpose(npy_data[p][8])[3]

            ax1.set_ylim(0, 10)
            ax1.plot(range(1, 730, 8), mh_lai_ts, color='green', linestyle='-')
            ax2.plot(range(1, 730, 8), x[:, p], 'r', label='prediction', linestyle='-')
            ax2.plot(range(1, 730, 8), y[:, p], 'black', label='real', linestyle='--')
            ax2.plot(range(1, 730, 8), ndvi_ts, color='gray', linestyle='--')
            # ax2.plot(range(1, 730, 16), mod_ndvi_ts, color='blue', linestyle='--')

            ax3 = ax2.twinx()
            ax3.set_ylim(-0.5, 1)
            ax2.set_yticks([])
            scat_sym_list = ['*', 'o', '+', 'x']
            for itim in range(1, 730, 16):
                point = ax3.scatter(itim, mod_ndvi_ts[int((itim - 1) / 16)], c='black',
                                    marker=scat_sym_list[mod_qa_ts[int((itim - 1) / 16)]], s=8)


        plt.tight_layout()
        plt.savefig('8_bilstm_ts_h' + str(h) + 'v' + str(v) + '.jpg')
        plt.show()
        plt.close('all')


    ####################################################################################################################

    # %%
    # 随机展示100个预测与输入序列 hv
    x = pred_data[:, :, ivi]
    y = data_Y[:, :, ivi]

    # hv_list = [[17, 8], [19, 8], [13, 4], [28, 11], [8, 6], [26, 6]]


    for i in range(len(npy_data)):
        if npy_data[i][2][1] < 3:
            # if npy_data[i][2][0] == h:
            #     if npy_data[i][2][3] < 2400 and (
            #             (npy_data[i][2][1] == 8 and npy_data[i][2][2] > 2400) or (npy_data[i][2][1] == 9 and npy_data[i][2][2] < 2400)):
            # if npy_data[i][2][1] == v:
            hv_id_list.append(i)

    # final_id_list = [item for item in npy_data if item[0] in hv_id_list]

    if len(hv_id_list) > 100:
        resultList = random.sample(range(0, len(hv_id_list)), 100)
    else:
        resultList = range(len(hv_id_list))

    plt.rcParams['figure.figsize'] = (20.0, 30.0)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    for i, pr in enumerate(resultList):

        p = hv_id_list[pr]
        ax1 = plt.subplot(20, 5, i + 1)
        ax2 = ax1.twinx()
        # ax2.ylim(0, 1)
        ax2.set_ylim(-0.5, 1)

        rmse = mean_squared_error(x[:, p], y[:, p])
        r2 = r2_score(x[:, p], y[:, p])
        bias = x[:, p].mean() - y[:, p].mean()

        type = npy_data[p][1]

        lat = npy_data[p][4][0]
        lon = npy_data[p][4][1]
        title = 'num=' + str(npy_data[p][0]) + ' type:' + str(int(type)) + ' lat:' + str(
            ('%.2f' % lat)) + ' lon:' + str(('%.2f' % lon))
        plt.title(title)

        mh_lai_ts = np.array(npy_data[p][7]) / 10
        ndvi_ts = npy_data[p][9][ivi]
        mod_ndvi_ts = np.transpose(npy_data[p][8])[ivi] / 10000
        mod_qa_ts = np.transpose(npy_data[p][8])[3]

        ax1.set_ylim(0, 10)
        ax1.plot(range(1, 730, 8), mh_lai_ts, color='green', linestyle='-')
        ax2.plot(range(1, 730, 8), x[:, p], 'r', label='prediction', linestyle='-')
        ax2.plot(range(1, 730, 8), y[:, p], 'black', label='real', linestyle='--')
        ax2.plot(range(1, 730, 8), ndvi_ts, color='gray', linestyle='--')
        # ax2.plot(range(1, 730, 16), mod_ndvi_ts, color='blue', linestyle='--')

        ax3 = ax2.twinx()
        ax3.set_ylim(-0.5, 1)
        ax2.set_yticks([])
        scat_sym_list = ['*', 'o', '+', 'x']
        for itim in range(1, 730, 16):
            point = ax3.scatter(itim, mod_ndvi_ts[int((itim - 1) / 16)], c='black',
                                marker=scat_sym_list[mod_qa_ts[int((itim - 1) / 16)]], s=8)

    plt.tight_layout()
    plt.savefig('9_bilstm_ts_h012.jpg')
    plt.show()
    plt.close('all')
    ####################################################################################################################

    # %%
    # 随机展示100个预测与输入序列 hv
    x = pred_data[:, :, ivi]
    y = data_Y[:, :, ivi]

    # hv_list = [[17, 8], [19, 8], [13, 4], [28, 11], [8, 6], [26, 6]]
    id_list = 17459


    for i in range(len(npy_data)):
        if npy_data[i][0] == 17459:
            # if npy_data[i][2][0] == h:
            #     if npy_data[i][2][3] < 2400 and (
            #             (npy_data[i][2][1] == 8 and npy_data[i][2][2] > 2400) or (npy_data[i][2][1] == 9 and npy_data[i][2][2] < 2400)):
            # if npy_data[i][2][1] == v:
            hv_id_list.append(i)

    # final_id_list = [item for item in npy_data if item[0] in hv_id_list]

    if len(hv_id_list) > 100:
        resultList = random.sample(range(0, len(hv_id_list)), 100)
    else:
        resultList = range(len(hv_id_list))

    plt.rcParams['figure.figsize'] = (20.0, 30.0)  # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 100  # 分辨率

    for i, pr in enumerate(resultList):

        p = hv_id_list[pr]
        ax1 = plt.subplot(20, 5, i + 1)
        ax2 = ax1.twinx()
        # ax2.ylim(0, 1)
        ax2.set_ylim(-0.5, 1)

        rmse = mean_squared_error(x[:, p], y[:, p])
        r2 = r2_score(x[:, p], y[:, p])
        bias = x[:, p].mean() - y[:, p].mean()

        type = npy_data[p][1]

        lat = npy_data[p][4][0]
        lon = npy_data[p][4][1]
        title = 'num=' + str(npy_data[p][0]) + ' type:' + str(int(type)) + ' lat:' + str(
            ('%.2f' % lat)) + ' lon:' + str(('%.2f' % lon))
        plt.title(title)

        mh_lai_ts = np.array(npy_data[p][7]) / 10
        ndvi_ts = npy_data[p][9][ivi]
        mod_ndvi_ts = np.transpose(npy_data[p][8])[ivi] / 10000
        mod_qa_ts = np.transpose(npy_data[p][8])[3]

        ax1.set_ylim(0, 10)
        ax1.plot(range(1, 730, 8), mh_lai_ts, color='green', linestyle='-')
        ax2.plot(range(1, 730, 8), x[:, p], 'r', label='prediction', linestyle='-')
        ax2.plot(range(1, 730, 8), y[:, p], 'black', label='real', linestyle='--')
        ax2.plot(range(1, 730, 8), ndvi_ts, color='gray', linestyle='--')
        # ax2.plot(range(1, 730, 16), mod_ndvi_ts, color='blue', linestyle='--')

        ax3 = ax2.twinx()
        ax3.set_ylim(-0.5, 1)
        ax2.set_yticks([])
        scat_sym_list = ['*', 'o', '+', 'x']
        for itim in range(1, 730, 16):
            point = ax3.scatter(itim, mod_ndvi_ts[int((itim - 1) / 16)], c='black',
                                marker=scat_sym_list[mod_qa_ts[int((itim - 1) / 16)]], s=8)

    plt.tight_layout()
    plt.savefig('10_bilstm_ts_id' + str(id_list) + '.jpg')
    plt.show()
    plt.close('all')



