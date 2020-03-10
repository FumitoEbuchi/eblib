##############################################
############時系列データの特徴抽出############
##############################################

import numpy as np
import sys
from progressbar import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import librosa

class MyIterator(object):
    def __init__(self, data):
        self._data = data
        self._i = 0

    def __iter__(self):
        return self
    def __next__(self):
        if self._i==self._data.shape[0]:
            raise StopIteration()
        value = self._data[self._i]
        self._i += 1
        return value

class FFT(object):
    def __init__(self):
        pass

    def transform(self, X, t=None):#Xはデータ数×データ点数(2のn乗 : 2**n)
        N = X.shape[1]
        dataset = MyIterator(X)
        for idx, inputs in enumerate(dataset):
            F = np.fft.fft(inputs)
            F_abs = np.abs(F)[:int(N/2)]
            F_abs_amp = F_abs/N*2
            F_abs_amp[0] = F_abs_amp[0]
            F_abs_amp = F_abs_amp.reshape(1,-1)
            if idx==0:
                transformed_X = F_abs_amp
            else:
                transformed_X = np.r_[transformed_X, F_abs_amp]
        if t is None:
            return transformed_X
        else:
            dt = np.fabs(t[1]-t[0])
            freq = np.linspace(0, 1.0/dt, N)
            return transformed_X, freq[:int(N/2)]


class BPF(object):#バンドパスフィルタ
    def __init__(self, fL = None, fH = None):
        self.fL = fL
        self.fH = fH

    def transform(self, X, t):
        N = X.shape[1]
        #周波数の計算
        dt = np.fabs(t[1]-t[0])
        freq = np.linspace(0, 1.0/dt, N)
        if self.fL is None:
            self.fL = 0
        if self.fH is None:
            self.fH = freq[int(N/2)]
        
        dataset = MyIterator(X)
        for idx, inputs in enumerate(dataset):
            F = np.fft.fft(inputs)
            F_abs = np.abs(F)[:int(N/2)]
            F_abs_amp = F_abs/N*2
            F_abs_amp[0] = F_abs_amp[0]
            F_abs_amp = F_abs_amp.reshape(1,-1)

            F2 = np.copy(F)
            F2[freq<self.fL] = 0
            F2[freq>self.fH] = 0

            F2_ifft = np.fft.ifft(F2)
            F2_ifft_real = F2_ifft.real*2
            F2_ifft_real = F2_ifft_real.reshape(1, -1)

            if idx==0:
                transformed_X = F2_ifft_real
            else:
                transformed_X = np.r_[transformed_X, F2_ifft_real]
        return transformed_X

from scipy.fftpack.realtransforms import dct
class MFCC(object):
    #librosaを使えば, mfccs = librosa.feature.mfcc(x, sr=sr)で各フレームごとのMFCCが得られる
    def __init__(self, numChannels=20, cdim = 13):
        self.numChannels = numChannels
        self.cdim = cdim

    def transform(self, X, t):
        self.N = X.shape[1]
        #周波数の計算
        dt = np.fabs(t[1]-t[0])
        freq = np.linspace(0, 1.0/dt, self.N)
        self.fs = freq[-1]

        #窓関数の定義
        hamming = np.hamming(self.N)

        filterbank, fcenters = self._melFilterBank()
        """
        #生成されたフィルタバンクの確認
        df = self.fs / self.N
        plt.figure()
        for c in range(self.numChannels):
            plt.plot(np.arange(0, self.N / 2) * df, filterbank[c])
        plt.title('Mel filter bank')
        plt.xlabel('Frequency[Hz]')
        plt.savefig(f'./tmp.png')
        plt.close()
        """
                
        dataset = MyIterator(X)
        for idx, inputs in enumerate(dataset):
            inputs*=hamming
            F = self._fft(inputs)
            #振幅スペクトルにメルフィルタバンクを適用
            mspec = filterbank@F
            #離散コサイン変換
            ceps = dct(mspec, type=2, norm="ortho", axis=-1)
            mfcc = ceps[:self.cdim].reshape(1,-1)

            if idx == 0:
                transformed_X = mfcc
            else:
                transformed_X = np.r_[transformed_X, mfcc]
        return transformed_X


    def _hz2mel(self, f):
        ###Hzをmelに変換###
        return 2595 * np.log(f / 700.0 + 1.0)

    def _mel2hz(self, m):
        ###melをHzに変換###
        return 700 * (np.exp(m / 2595) - 1.0)

    def _melFilterBank(self):
        #URL : https://qiita.com/tmtakashi_dist/items/eecb705ea48260db0b62#
        ###メルフィルタバンクを作成###
        # ナイキスト周波数（Hz）
        fmax = self.fs / 2
        # ナイキスト周波数（mel）
        melmax = self._hz2mel(fmax)
        # 周波数インデックスの最大数
        nmax = int(self.N / 2)
        # 周波数解像度（周波数インデックス1あたりのHz幅）
        df = self.fs / self.N
        # メル尺度における各フィルタの中心周波数を求める
        dmel = melmax / (self.numChannels + 1)
        melcenters = np.arange(1, self.numChannels + 1) * dmel
        # 各フィルタの中心周波数をHzに変換
        fcenters = self._mel2hz(melcenters)
        # 各フィルタの中心周波数を周波数インデックスに変換
        indexcenter = np.round(fcenters / df)
        # 各フィルタの開始位置のインデックス
        indexstart = np.hstack(([0], indexcenter[0:self.numChannels - 1]))
        # 各フィルタの終了位置のインデックス
        indexstop = np.hstack((indexcenter[1:self.numChannels], [nmax]))
        filterbank = np.zeros((self.numChannels, nmax))
        for c in range(0, self.numChannels):
            # 三角フィルタの左の直線の傾きから点を求める
            increment= 1.0 / (indexcenter[c] - indexstart[c])
            for i in range(int(indexstart[c]), int(indexcenter[c])):
                filterbank[c, i] = (i - indexstart[c]) * increment
            # 三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
            for i in range(int(indexcenter[c]), int(indexstop[c])):
                filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
        return filterbank, fcenters

    def _fft(self, x):
        F = np.fft.fft(x)
        F_abs = np.abs(F)[:int(self.N/2)]
        F_abs_amp = F_abs/self.N*2
        F_abs_amp[0] = F_abs_amp[0]
        F_abs_amp = F_abs_amp
        return F_abs_amp.reshape(-1)

from scipy import signal
class STFT(object):
    def __init__(self, fs = 22050, window = 'hann', nperseg = 256, noverlap=None):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
    
    def transform(self, X):
        dataset = MyIterator(X)
        for idx, inputs in enumerate(dataset):
            f, t, Zxx = signal.stft(inputs, fs = self.fs, window = self.window, nperseg = self.nperseg, noverlap=self.noverlap)
            Zxx = Zxx.reshape(1, Zxx.shape[0], Zxx.shape[1])
            if idx==0:
                transformed_X = Zxx
            else:
                transformed_X = np.r_[transformed_X, Zxx]
        return transformed_X, f, t

import os
import time
from PIL import Image
from progressbar import progressbar

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import torchvision.utils as vutils
class Sig2CNN(object): #ImageNet学習済みモデルを利用した特徴抽出
    def __init__(self, seed=-1):
        self.path = f'./sig2vec/Sig2CNN'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        if(seed>=0):
            np.random.seed(seed)
            torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device("cpu")

        resnet50 = torchvision.models.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-1]
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)

        self.data_transforms = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    def transform(self, t, X):
        #STFTの実行
        stft = STFT(fs = 1/(t[1]-t[0]), window = 'hann', nperseg = 256, noverlap=None)
        transformed_X, f, t = stft.transform(X)
        dataset = MyIterator(transformed_X)
        for idx, inputs in progressbar(enumerate(dataset)):
            plt.figure()
            plt.pcolormesh(t,f, np.abs(inputs), vmin=0, vmax=X.max())
            plt.tick_params(labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False)
            plt.tick_params(bottom=False,
            left=False,
            right=False,
            top=False)
            plt.savefig(f'{self.path}/tmp.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()
            time.sleep(0.1)

            img = Image.open(f'{self.path}/tmp.jpg')
            img = self.data_transforms(img)
            with torch.no_grad():
                img = Variable(img.to(self.device)).float()
                outputs = self.model(img.unsqueeze(0))
                outputs = outputs.cpu().data.numpy().reshape(1, -1)
            if idx==0:
                transformed_X = outputs
            else:
                transformed_X = np.r_[transformed_X, outputs]
        return transformed_X

from sklearn import metrics
class Takens(object):#ターケンス埋め込み, 時系列データ→アトラクター
    tau_max = 100
    def __init__(self, tau = 1, k = 3):
        self.tau = int(tau)
        self.k = int(k)
    
    def reconstruct(self, data):
        #if self.tau is None:
        #   self.tau = self.__search_tau(data)
        x = np.zeros((len(data[:-self.k+1]), self.k))
        x[:,0] = data[:-self.k+1]
        for i in range(1, self.k):
            x[:,i] = np.roll(data, -i*self.tau)[:-self.k+1]
        return x
    
    def __search_tau(self, data):#tauの自動最適化 (上では未使用)
        # Create a discrete signal from the continunous dynamics
        hist, bin_edges = np.histogram(data, bins=200, density=True)
        bin_indices = np.digitize(data, bin_edges)
        data_discrete = data[bin_indices]

        # find usable time delay via mutual information
        before     = 1
        nmi        = []
        res        = None
        for tau in range(1, self.tau_max):
            unlagged = data_discrete[:-tau]
            lagged = np.roll(data_discrete, -tau)[:-tau]
            nmi.append(metrics.normalized_mutual_info_score(unlagged, lagged))

            if res is None and len(nmi) > 1 and nmi[-2] < nmi[-1]:
                res = tau - 1
        if res is None:
            res = 50
        return res, nmi

import nolds
class Lyapunov1d(object):#1次元最大リアプノフ指数
    """
    参照
    https://nolds.readthedocs.io/en/latest/nolds.html#lyapunov-exponent-eckmann-et-al
    nolds.lyap_r(data, emb_dim=10,
                lag=None, min_tsep=None,
                tau=1, min_neighbors=20,
                trajectory_len=20, fit=u'RANSAC',
                debug_plot=False, debug_data=False,
                plot_file=None, fit_offset=0)
    
    emb_dim : 埋め込み次元
    lag : 埋め込むときのタイムラグ
    min_tsep : 2つの「隣接」間の最小の時間分離
    tau : 時系列データ間のステップサイズ
    fit : 最小二乗多項式フィッティング→poly, 外れ値にロバスト→RANSAC
    """
    def __init__(self, fs, num_point = None):
        self.num_point = num_point
        self.dt = 1/fs
        self.eps = 1e-10

    def transform(self, X):
        if self.num_point is None:
            self.num_point = X.shape[1]
        
        dataset = MyIterator(X)
        for idx, inputs in enumerate(dataset):
            val = np.array([np.log(np.fabs((self.eps+inputs[i+1]-inputs[i])/self.dt)) for i in range(len(inputs)-1)])
            #max_lyapunov= nolds.lyap_r(inputs.tolist(), emb_dim=2, fit='poly', debug_plot=True, debug_data=True, plot_file=f'./tmp.png')
            max_lyapunov= nolds.lyap_r(inputs.tolist(), emb_dim=2, fit='poly', min_tsep=int(len(inputs)//4))
            if idx==0:
                transformed_X = max_lyapunov
            else:
                transformed_X = np.r_[transformed_X, max_lyapunov]
        return transformed_X

