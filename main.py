import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--IN_FEATURE",type=str,help="the shape of the INfeature Default = 4*32*64*4,(4for 4features and 4 days)",default=4*32*64*4)
parser.add_argument("--OUT_FEATURE",type=str,help="the shape of the outfeature Default = 32*64",default=32*64)
parser.add_argument("--BATCH_SIZE",type=str,help="BATCH_SIZE Default = 20",default=20)
parser.add_argument("--EPOCH",type=str,help="EPOCH Default = 200",default=150)
parser.add_argument("--feature1",type=str,help="Location of features 1",default='train_test/layer1.npz')
parser.add_argument("--feature2",type=str,help="Location of features 2",default='train_test/layer2.npz')
parser.add_argument("--feature3",type=str,help="Location of features 3",default='train_test/rainfall.npz')
parser.add_argument("--feature4",type=str,help="Location of features 4",default='train_test/soilType.npz')
parser.add_argument("--feature5",type=str,help="Location of features 5",default='train_test/temperature.npz')
args = parser.parse_args()
IN_FEATURE=args.IN_FEATURE
OUT_FEATURE=args.OUT_FEATURE
BATCH_SIZE=args.BATCH_SIZE
EPOCH=args.EPOCH
feature1=args.feature1
feature2=args.feature2
feature3=args.feature3
feature4=args.feature4
feature5=args.feature5

# IN_FEATURE = 4 * 32 * 64 * 4
# OUT_FEATURE = 32 * 64
# BATCH_SIZE = 20
# EPOCH = 2
TEMP_max=350
TEMP_min=200

# design model using class
def getData(path='data.npy'):
    data = np.load(path)
    print(data.shape)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(in_features=IN_FEATURE, out_features=OUT_FEATURE)

    def forward(self, x):
        x = x.view(-1, IN_FEATURE)
        y_pred = self.linear(x)
        return y_pred


def getdataloader(file='train_test', mode=True):
    """
    :param file:源头文件
    :param mode: 是否训练
    :return:
    """
    layer1_data = np.load(feature1)['train_data' if mode else 'test_data']
    layer1_data = np.nan_to_num(layer1_data)
    layer2_data = np.load(feature2)['train_label' if mode else 'test_label']
    layer2_data = np.nan_to_num(layer2_data)
    rainfall_data = np.load(feature3)['train_data' if mode else 'test_data']
    rainfall_data = np.nan_to_num(rainfall_data)
    soilType_data = np.load(feature4)['train_data' if mode else 'test_data']
    soilType_data = np.nan_to_num(soilType_data)
    temperature_data = np.load(feature5)['train_data' if mode else 'test_data']
    temperature_data = (temperature_data-TEMP_min)/(TEMP_max-TEMP_min)
    label = layer2_data
    data = np.concatenate((layer1_data, rainfall_data, soilType_data,temperature_data),axis=2)  # layer2_data,rainfall_data,soilType_data,temperature_data
    print("dataset data shape", data.shape, " label shape", label.shape)

    data = torch.tensor(data.reshape(-1, IN_FEATURE)).cuda()
    label = torch.tensor(label.reshape(-1, OUT_FEATURE)).cuda()
    dataset = TensorDataset(data, label)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    return dataloader
from torchnet.meter import AverageValueMeter

if __name__ == '__main__':
    dataloader = getdataloader(mode=True)
    model = LinearModel()
    model.cuda()
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters ",trainable_num)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作
    summary = SummaryWriter('log')
    x = torch.rand(size=(17,IN_FEATURE)).cuda()
    summary.add_graph(model,x)
    loss_meter = AverageValueMeter()
    # training cycle forward, backward, update
    for i in range(EPOCH):
        myloss = 0
        iter=0
        for epoch, (x_data, y_data) in enumerate(dataloader):
            y_pred = model(x_data)  # forward:predict
            loss = criterion(y_pred, y_data)  # forward: loss
            myloss += loss.item()
            loss_meter.add(loss.item())
            # print(myloss)
            optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
            loss.backward()  # backward: autograd
            optimizer.step()  # update 参数，即更新w和b的值
            iter = epoch
        print("[epcho {}] loss {}:".format(i, loss_meter.value()[0]))
        summary.add_scalar("train_loss",myloss/iter,global_step=i)
        loss_meter.reset()




    model.eval()
    testLoader = getdataloader(mode=False)
    myloss = 0
    loss_meter.reset()
    for epoch, (test_x, test_y) in enumerate(testLoader):
        pred = model(test_x)
        loss = criterion(pred, test_y)
        loss_meter.add(loss.item())
        myloss += loss.item()
    print('mse:',loss_meter.value()[0])
    rmse=pow(loss_meter.value()[0],0.5)
    print('rmse:',rmse)


    testdata=[]
    testlabel=[]
    for (data,label) in testLoader:
        testdata.extend(data)
        testlabel.extend(label)
    testdata=np.array(testdata)
    testlabel=np.array(testlabel)
    #print(testdata.shape)
    #print(testlabel.shape)
    temp=testdata[0]
    #print(temp.shape)
    y_pred=model(temp)
    #print(y_pred.shape)
    y_true=testlabel[0]
    #print(y_true.shape)
    y_true=y_true.cpu().numpy()
    y_true=y_true.reshape(-1,1)
    y_pred=y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1, 1)
    # mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(y_true, y_pred)
    print('R-squared',r2)

    y_pred=y_pred.reshape(32,64)
    y_pred[y_pred<0.0001]=0


    map = Basemap(lon_0 = -180)
    x = np.linspace(-360, map.urcrnrx, y_pred.shape[1])
    y = np.linspace(-90, map.urcrnry, y_pred.shape[0])
    xx, yy = np.meshgrid(x, y)
    map.contourf(xx, yy, y_pred)
    map.drawcoastlines(linewidth=0.5)
    plt.colorbar(orientation='horizontal',cax=plt.axes([0.1, 0.06, 0.8, 0.1]))
    plt.show()
    torch.save(model, "mymodel_linear.pkl")
