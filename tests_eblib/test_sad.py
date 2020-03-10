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

#異常データを選択する関数
def get_anomalous(classnumber, train_X, train_Y, class_list=[0,1,2,3,4]):
    np.random.seed(0)
    #正常クラスが異常クラス候補にあれば除外
    if classnumber in class_list:
        class_list.remove(classnumber)
    #class_listの各クラスからはじめの1点のみ取得
    for idx, i in enumerate(class_list):
        tmp_X = train_X[train_Y==i]
        all_idx = np.arange(tmp_X.shape[0])
        np.random.shuffle(all_idx)
        tmp_X = tmp_X[all_idx]
        if idx==0:
            ano_X = tmp_X[0].reshape(1,-1)
        else:
            ano_X = np.r_[ano_X, tmp_X[0].reshape(1,-1)]
    return ano_X


from eblib.sad import ISM
def TestISM():
    train = MNIST('./', train=True, download=True)
    test = MNIST('./', train=False, download=True)

    train_X, train_Y = train.data.numpy(), train.targets.numpy()
    test_X, test_Y = test.data.numpy(), test.targets.numpy()

    train_X, train_Y = train_X.reshape(train_X.shape[0], -1), train_Y.reshape(-1)
    test_X, test_Y = test_X.reshape(test_X.shape[0], -1), test_Y.reshape(-1)

    for class_idx in range(10):
            print(f'Class {class_idx}')
            #クラスclass_idxのデータを選択
            cls_idx = np.where(train_Y==class_idx)
            cls_X = train_X[cls_idx]
            pr = MinMaxScaler()
            pr.fit(cls_X)
            train_X, test_X = pr.transform(train_X), pr.transform(test_X)
            #正規化したデータを再度抽出
            cls_X = train_X[cls_idx]
            ano_X = get_anomalous(class_idx, train_X, train_Y, class_list=[0,1,2,3,4])
            #学習データの評価ラベル生成
            bi_y = np.ones(len(train_Y))
            bi_y[cls_idx] = 0
            #テストデータの評価ラベル生成
            test_bi_y = np.ones(len(test_Y))
            test_bi_y[test_Y==class_idx] = 0

            #部分空間法の学習
            ism = ISM(eta=0.99, C=0.05)
            ism.fit(cls_X, ano_X)
            #評価値の算出
            train_D = ism.decision_function(train_X)
            test_D = ism.decision_function(test_X)

            #AUCの表示
            print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
            print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

            del ism

from eblib.sad import SSAD
def TestSSAD():
    train = MNIST('./', train=True, download=True)
    test = MNIST('./', train=False, download=True)

    train_X, train_Y = train.data.numpy(), train.targets.numpy()
    test_X, test_Y = test.data.numpy(), test.targets.numpy()

    train_X, train_Y = train_X.reshape(train_X.shape[0], -1), train_Y.reshape(-1)
    test_X, test_Y = test_X.reshape(test_X.shape[0], -1), test_Y.reshape(-1)

    for class_idx in range(10):
            print(f'Class {class_idx}')
            #クラスclass_idxのデータを選択
            cls_idx = np.where(train_Y==class_idx)
            cls_X = train_X[cls_idx]
            pr = MinMaxScaler()
            pr.fit(cls_X)
            train_X, test_X = pr.transform(train_X), pr.transform(test_X)
            #正規化したデータを再度抽出
            cls_X = train_X[cls_idx]
            ano_X = get_anomalous(class_idx, train_X, train_Y, class_list=[0,1,2,3,4])
            #学習データの評価ラベル生成
            bi_y = np.ones(len(train_Y))
            bi_y[cls_idx] = 0
            #テストデータの評価ラベル生成
            test_bi_y = np.ones(len(test_Y))
            test_bi_y[test_Y==class_idx] = 0

            #部分空間法の学習
            ssad = SSAD(kernel='rbf', d = 3, gamma=0.05, kappa = 1.0, Cp = 1.0, Cu = 1.0, Cn =1.0)
            ssad.fit(cls_X, ano_X)
            #評価値の算出
            train_D = ssad.decision_function(train_X)
            test_D = ssad.decision_function(test_X)

            #AUCの表示
            print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
            print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

            del ssad

if __name__=='__main__':
    #TestISM()
    #TestSSAD()
