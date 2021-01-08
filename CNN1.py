import torch
import numpy as np
import torch.utils.data as Data
from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import argparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--IN_FEATURE", type=str, help="the shape of the INfeature Default = 32*64", default=4 * 32 * 64)
parser.add_argument("--OUT_FEATURE", type=str, help="the shape of the outfeature Default = 32*64", default=32 * 64)
parser.add_argument("--BATCH_SIZE", type=str, help="BATCH_SIZE Default = 32", default=32)
parser.add_argument("--EPOCH", type=str, help="EPOCH Default = 150", default=1)
parser.add_argument("--feature1", type=str, help="Location of features 1", default='train_test/layer1.npz')
parser.add_argument("--feature2", type=str, help="Location of features 2", default='train_test/layer2.npz')
parser.add_argument("--feature3", type=str, help="Location of features 3", default='train_test/rainfall.npz')
parser.add_argument("--feature4", type=str, help="Location of features 4", default='train_test/soilType.npz')
parser.add_argument("--feature5", type=str, help="Location of features 5", default='train_test/temperature.npz')
args = parser.parse_args()

IN_FEATURE = args.IN_FEATURE
OUT_FEATURE = args.OUT_FEATURE
BATCH_SIZE = args.BATCH_SIZE
EPOCH = args.EPOCH
feature1 = args.feature1
feature2 = args.feature2
feature3 = args.feature3
feature4 = args.feature4
feature5 = args.feature5


# IN_FEATURE = 4 * 32 * 64
# OUT_FEATURE = 32 * 64
# BATCH_SIZE = 32
# EPOCH = 2


def transformData(data):
    scalar = MinMaxScaler()
    re = data.reshape(data.shape[0], -1)
    re = scalar.fit_transform(re)
    re = re.reshape(data.shape)
    return re, scalar


def transformTest(data, scalar):
    re = data.reshape(data.shape[0], -1)
    re = scalar.transform(re)
    re = re.reshape(data.shape)
    return re


class DataProcessing:
    def __init__(self):
        self.temperature_scalar = None
        self.layer1_scalar = None
        self.layer2_scalar = None
        self.rainfall_scalar = None
        self.soilType_scalar = None

    def getdataloader(self, file='train_test', mode=True):
        """
        :param file:源头文件
        :param mode: 是否训练
        :return:
        """
        layer1_data = np.load(feature1)['train_label' if mode else 'test_label']
        layer2_data = np.load(feature2)['train_data' if mode else 'test_data']
        rainfall_data = np.load(feature3)['train_data' if mode else 'test_data']
        soilType_data = np.load(feature4)['train_data' if mode else 'test_data']
        temperature_data = np.load(feature5)['train_data' if mode else 'test_data']

        ############################
        if mode:
            temperature_data, self.temperature_scalar = transformData(temperature_data)
            layer1_data, self.layer1_scalar = transformData(layer1_data)
            layer2_data, self.layer2_scalar = transformData(layer2_data)
            rainfall_data, self.rainfall_scalar = transformData(rainfall_data)
            soilType_data, self.soilType_scalar = transformData(soilType_data)

        else:
            temperature_data = transformTest(temperature_data, self.temperature_scalar)
            layer1_data = transformTest(layer1_data, self.layer1_scalar)
            layer2_data = transformTest(layer2_data, self.layer2_scalar)
            rainfall_data = transformTest(rainfall_data, self.rainfall_scalar)
            soilType_data = transformTest(soilType_data, self.soilType_scalar)

        ###########################
        label = layer1_data
        print(label.shape)
        data = np.concatenate((layer2_data, rainfall_data, soilType_data, temperature_data), axis=2)
        print("dataset data shape", data.shape, " label shape", label.shape)
        data = torch.tensor(data.reshape(data.shape[0], 16, 32, 64), dtype=torch.float32).cuda()
        label = torch.tensor(label.reshape(-1, OUT_FEATURE), dtype=torch.float32).cuda()
        print(data.shape)
        dataset = TensorDataset(data, label)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            num_workers=0
        )
        return dataloader


class myCNN(torch.nn.Module):
    def __init__(self, input_channel, out_channel=64, kernel_size=3, stride=2):
        """
        :param size:一日数据
        :param day: n日
        :param hidden_dim:隐藏层神经元
        :param layer_dim: 隐藏层个数
        :param output_dim: 输出
        """
        super(myCNN, self).__init__()
        self.cnn = torch.nn.Conv2d(in_channels=input_channel, kernel_size=3, stride=3, out_channels=out_channel)
        self.dense = torch.nn.Linear(in_features=64 * 10 * 21, out_features=OUT_FEATURE)

    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x


from torchnet.meter import AverageValueMeter
from torch.optim import lr_scheduler

if __name__ == '__main__':
    lr = 0.0005
    # total_epoch = 30
    sumWriter = SummaryWriter('cnn_log')
    # 数据迭代器
    dataprocessing = DataProcessing()
    train_loader = dataprocessing.getdataloader()
    net = myCNN(input_channel=16)
    net.cuda()
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("trainable parameters ", trainable_num)
    x = torch.rand(size=[17, 16, 32, 64]).cuda()
    sumWriter.add_graph(model=net, input_to_model=x)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [i for i in range(0,500,150)][1:], gamma=0.05)
    loss_func = torch.nn.MSELoss()
    global_step = 1
    loss_metrics = AverageValueMeter()

    for epoch in range(EPOCH):
        epoch_loss = 0
        for step, (x, y) in tqdm(enumerate(train_loader)):
            output = net(x)
            train_loss = loss_func(output, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            global_step = global_step + 1
            epoch_loss += train_loss.item()
            loss_metrics.add(train_loss.item())
        print("[epcho {}]:loss {}".format(epoch, loss_metrics.value()[0]))
        loss_metrics.reset()
        scheduler.step()
    test_loader = dataprocessing.getdataloader(mode=False)
    test_loss = 0
    global_step = 0
    loss_metrics.reset()
    for step, (x, y) in tqdm(enumerate(test_loader)):
        print(epoch, " global step ", global_step)
        output = net(x)
        train_loss = loss_func(output, y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        test_loss += train_loss.item()
        global_step += 1
        loss_metrics.add(train_loss.item())
    print("MSE:", loss_metrics.value()[0])
    rmse = pow(loss_metrics.value()[0], 0.5)
    print('rmse:', rmse)

    for (x, y) in test_loader:
        break

    predict = net(x)
    print("predict_shape", predict.shape)
    predict = predict.detach().cpu().numpy()
    y = y.cpu().numpy()
    y = y[0]
    predict = predict[0]
    r2 = r2_score(y, predict)
    print('R-squared', r2)
    label_scalar = dataprocessing.layer1_scalar
    predict = label_scalar.inverse_transform(predict.reshape(1, -1))
    predict = predict.reshape(32, 64)
    y = y.reshape(32, 64)
    print("predict shape2", predict.shape)
    predict[predict < 0.0001] = 0
    # map = Basemap(lon_0=-180)
    # a = np.linspace(-360, map.urcrnrx, predict.shape[1])
    # b = np.linspace(-90, map.urcrnry, predict.shape[0])
    # xx, yy = np.meshgrid(a, b)
    # map.contourf(xx, yy, predict)
    # map.drawcoastlines(linewidth=0.5)
    # plt.colorbar(orientation='horizontal', cax=plt.axes([0.1, 0.06, 0.8, 0.1]))
    # plt.show()
    # torch.save(net, "mymodel_lstm.pkl")
