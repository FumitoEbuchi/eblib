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

from eblib.uad import SM
def TestSM():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #部分空間法の学習
        sm = SM(eta=0.99)
        sm.fit(cls_X)
        #評価値の算出
        train_D = sm.decision_function(train_X)
        test_D = sm.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

        del sm

from eblib.uad import KSM
def TestKSM():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #部分空間法の学習
        ksm = KSM(kernel = 'rbf', d = 3, gamma=0.05, eta = 0.95)
        ksm.fit(cls_X)
        #評価値の算出
        train_D = ksm.decision_function(train_X)
        test_D = ksm.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')
        
        del ksm

from eblib.uad import DAE
def TestDAE():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #オートエンコーダの学習
        ae = DAE(max_epoch=100, batch_size=128, lr=1e-3, weight_decay=0, dims=[784, 128, 128, 10], seed=0)
        ae.fit(cls_X)
        #評価値の算出
        train_D = ae.decision_function(train_X)
        test_D = ae.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

        del ae

from eblib.uad import CSM
def TestCSM():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #部分空間法の学習
        csm = CSM(n_components = 200, max_iter=200)
        csm.fit(cls_X)
        #評価値の算出
        train_D = csm.decision_function(train_X)
        test_D = csm.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

        del csm

from eblib.uad import AnoGAN
def TestAnoGAN():
    train = MNIST('./', train=True, download=True)
    test = MNIST('./', train=False, download=True)

    train_X, train_Y = train.data.numpy(), train.targets.numpy()
    test_X, test_Y = test.data.numpy(), test.targets.numpy()

    train_X, train_Y = np.stack([train_X]*3, axis=1).transpose(0, 2, 3, 1), train_Y.reshape(-1)
    test_X, test_Y = np.stack([test_X]*3, axis=1).transpose(0, 2, 3, 1), test_Y.reshape(-1)

    for class_idx in range(10):
        print(f'Class {class_idx}')
        #クラスclass_idxのデータを選択
        cls_idx = np.where(train_Y==class_idx)
        cls_X = train_X[cls_idx]
        
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #DCGANの学習
        anogan = AnoGAN(max_epoch=10, lr=0.0002, batch_size=128, img_size=64, nc=3, ndf=64, ngf=64, nz=100)
        anogan.fit(cls_X)
        anogan.check_img(name=f'AnoGAN_class{class_idx}')
        #評価値の算出
        train_D = anogan.decision_function(train_X[:100])
        test_D = anogan.decision_function(test_X[:100])

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y[:100], train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y[:100], test_D)}')

        del anogan

from eblib.uad import SVDD
def TestSVDD():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #超球学習
        svdd = SVDD(kernel='rbf', gamma=0.05, C=10)
        svdd.fit(cls_X)

        #評価値の算出
        train_D = svdd.decision_function(train_X)
        test_D = svdd.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')
        
        del svdd

from eblib.uad import OCSVM
def TestOCSVM():
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
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #超球学習
        ocsvm = OCSVM(kernel='rbf', gamma=0.05, nu=0.01)
        ocsvm.fit(cls_X)

        #評価値の算出
        train_D = ocsvm.decision_function(train_X)
        test_D = ocsvm.decision_function(test_X)

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')
        
        del ocsvm


from eblib.uad import VAE
def TestVAE():
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
        pr = MinMaxScaler(feature_range=(-1,1))
        pr.fit(cls_X)
        train_X, test_X = pr.transform(train_X), pr.transform(test_X)
        #正規化したデータを再度抽出
        cls_X = train_X[cls_idx]
        #学習データの評価ラベル生成
        bi_y = np.ones(len(train_Y))
        bi_y[cls_idx] = 0
        #テストデータの評価ラベル生成
        test_bi_y = np.ones(len(test_Y))
        test_bi_y[test_Y==class_idx] = 0

        #オートエンコーダの学習
        vae = VAE(max_epoch=100, batch_size=128, lr=1e-3, weight_decay=0, dims=[784, 128, 128, 10], seed=0)
        vae.fit(cls_X)
        #評価値の算出
        train_D = vae.decision_function(train_X)
        print(np.where(np.isnan(train_D)))
        test_D = vae.decision_function(test_X)
        print(np.where(np.isnan(test_D)))

        #AUCの表示
        print(f'Train data AUC : {roc_auc_score(bi_y, train_D)}')
        print(f'Test data AUC : {roc_auc_score(test_bi_y, test_D)}')

        del vae
def main():
    #TestSM()
    #TestKSM()
    #TestDAE()
    #TestCSM()
    #TestAnoGAN()
    #TestSVDD()
    #TestOCSVM()
    TestVAE()


if __name__=='__main__':
    main()
