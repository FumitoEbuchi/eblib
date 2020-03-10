import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

#PATH関数
import os
import sys
from pathlib import Path
Path().resolve()
def parentpath(path='.', f=0):
    return Path(path).resolve().parents[f]
#print(parentpath(Path().resolve(), 0))
lib_path = str(parentpath(Path().resolve(), 0))
sys.path.append(lib_path)


import matplotlib.pyplot as plt
from eblib.img2vec import CNNFeature
def TestCNNFeature():
    train = MNIST('./', train=True, download=True)
    test = MNIST('./', train=False, download=True)

    train_X, train_Y = train.data.numpy(), train.targets.numpy()
    test_X, test_Y = test.data.numpy(), test.targets.numpy()
    
    train_X, test_X = np.stack([train_X]*3, axis=1), np.stack([test_X]*3, axis=1)
    #この時点で[データ数, チャンネル数, 縦, 横]
    #CNNFeature()は[データ数, 縦, 横, チャンネル数]を入力とする
    train_X, test_X = train_X.transpose(0,2,3,1), test_X.transpose(0,2,3,1)

    fe = CNNFeature()
    train_X, test_X = fe.transform(train_X), fe.transform(test_X)
    print(train_X.shape)
    

if __name__=='__main__':
    TestCNNFeature()
