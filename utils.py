import numpy as np
import torch
import os
DELTA_DATE = 13148  # 0-13148=1981-1-1~2016-12-31


def get_data_and_lable(file='total_data.npy', begin=0, end=DELTA_DATE):
    rawdata = np.load(file)
    if end is None:
        end = rawdata.shape[0]
    # print("data shape load from ", file, " shape ", rawdata.shape)
    data = []
    label = []
    for i in range(begin, end - 6):
        tmp_data = rawdata[i:i + 4]
        tmp_label = rawdata[i + 6]
        data.append(tmp_data)
        label.append(tmp_label)
    data = np.array(data)
    label = np.array(label)
    # print("output ", data.shape, label.shape)
    return data, label


def split_train_test(src,des='train_test.npz'):
    train_data, train_label = get_data_and_lable(src)
    test_data, test_label = get_data_and_lable(file=src,begin=DELTA_DATE, end=None)
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    np.savez(des, train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label)


if __name__ == '__main__':

    npydir ='Npydata'
    des='train_test'
    npyfiles = os.listdir(npydir)
    for file in npyfiles:
        name = file.split('.')[0]
        print(file)
        split_train_test(src=os.path.join(npydir,file),des=os.path.join(des,str(name)+".npz"))
