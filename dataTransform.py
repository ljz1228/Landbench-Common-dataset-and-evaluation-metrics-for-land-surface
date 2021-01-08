"""
将所有数据从原始文件中读取，降清晰度，并存入单一npy文件
"""

from scipy import interpolate
import numpy as np
import netCDF4 as nc
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--PATH", type=str,help="The intout directory default = F:\data",default='F:\data')
parser.add_argument("-d", "--DES", type=str,help="The outout directory default = npydata",default='npydata')
parser.add_argument("-f", "--First_year", type=str,help="The start of the year Default = 1981",default='1981')
parser.add_argument("--e",type=str,help="File ending. Default = .npy",default='.npy')
parser.add_argument("--X_direction",type=int,help="X direction Default = 64",default=64)
parser.add_argument("--Y_direction",type=int,help="Y direction Default = 32",default=32)

args = parser.parse_args()
P=args.PATH
D=args.DES
F=args.First_year
file_ending=args.e
R1=args.X_direction
R2=args.Y_direction
RX=360/R1
RY=180/R2

#print(RX,RY)

#PATH = 'F:\data'  # 所有数据位置
#DES = 'npydata'
#First_year = '1981'
KEY_dic = {
    'layer1': 'swvl1',
    'layer2': 'swvl2',
    'rainfall': 'tp',
    'soilType': 'slt',
    'temperature': 'stl1'
}


def transform_data(src, key='swvl1'):
    """
    数据处理
    :param src: 源地址 .nc
    :param des: 目的地址 目录
    :return:
    """
    old_NC = nc.Dataset(src)
    # 获取维度的值，一般有时间、等级、经纬度
    time = old_NC.variables['time'][:].data
    latitude = old_NC.variables['latitude'][:].data
    longitude = old_NC.variables['longitude'][:].data
    swvl1 = old_NC.variables[key][:].data

    bands = times = time.size  # 时间数
    cols = longitude.size  # 列数
    rows = latitude.size  # 行数

    data = np.zeros((bands, rows, cols), dtype=np.float32)  # 存放数据的数组
    for i in range(bands):
        data[i] = data[i] = swvl1[i, :, :]

    datax = np.arange(0, 360, 0.703125)  # 原图的x方向尺寸，从0~360，间隔0.70125
    datay = np.arange(90.317025, -90, -0.703125)  # 原图的y方向尺寸，从90~-90，间隔-0.70125
    dataxx, datayy = np.meshgrid(datax, datay)  # 以datax/datay建立格网

    ############## 4间隔的插值
    dataxnew005 = np.arange(0, 360, RX)  # 插值后的x方向尺寸，从0~360，间隔4
    dataynew005 = np.arange(90, -90, -RY)  # 插值后的y方向尺寸，从-90~90，间隔4
    rows005 = len(dataynew005)  # y方向个数，即行数
    cols005 = len(dataxnew005)  # x方向个数，即列数
    datanew005 = np.zeros((bands, rows005, cols005), dtype=np.float32)  # 建立新data存放插值后的数据

    swvl1_new = []
    for i in range(bands):
        # 0.05插值
        f1 = interpolate.interp2d(datax, datay, data[i], kind='linear')
        datanew005[i] = f1(dataxnew005, dataynew005)
        swvl1_new.append(datanew005[i])
    # # 给变量填充数据
    # new_NC.variables['swvl1'][:] = swvl1_new
    #
    # # 关闭文件
    # new_NC.close()
    return np.array(swvl1_new)


# def transformFiles(rawDir='raw_data/1981', desDir='data/'):
#     """对一个文件夹的数据进行批量处理"""
#     filelist = os.listdir(rawDir)
#     for file in filelist:
#         try:
#             transform_data(os.path.join(rawDir, file), os.path.join(desDir, file))
#             print(file, "is success")
#         except Exception:
#             print(file, " is failed!")


def getDataFromNC(src, key='swvl1', shape=(1, 32, 64)):
    """
    :param src: 目标nc
    :param key: 键值
    :param shape: 形状，进行判断
    :return:
    """
    dataset = nc.Dataset(src)
    data = dataset.variables[key]
    data = np.array(data)
    assert data.shape == shape
    return data


def cmp(string):
    """
    根据文件名，对年份、月份、日期排序
    :param string:
    :return:
    """
    elem = string.split('_')

    return int(elem[-4]), int(elem[-3]), int(elem[-2])


def compileToNpy(src, key):
    """
    压缩为一个npy
    :param src: 源数据 dir
    :param des: 目的数据 .npy
    :return:
    """
    result = []

    filelist = os.listdir(src)
    filelist.sort(key=cmp)
    for file in filelist:
        try:
            tmp = transform_data(src=os.path.join(src, file), key=key)
            result.append(tmp)
            print(file, " success")
        except Exception:
            print(file, " failed")

    result = np.array(result)
    print("data shape ", result.shape)

    return result


if __name__ == '__main__':
    for item in os.listdir(P):

        itemPath = os.path.join(P, item)
        year_path_list = os.listdir(itemPath)
        for year in year_path_list:
            tmp = compileToNpy(src=os.path.join(itemPath, year), key=KEY_dic[item])
            if year == F:
                data = tmp
            else:
                data = np.concatenate((data, tmp), axis=0)
        np.save(os.path.join(D, str(item) + file_ending), data)
