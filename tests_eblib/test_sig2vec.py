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
from eblib.sig2vec import FFT
def TestFFT():
    t = np.linspace(0,1,2**12).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t),
            np.sin(2*np.pi*100*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    t = t.reshape(-1)

    fft = FFT()
    transformed_X, freq = fft.transform(X, t=t)

    path = f'./sig2vec/TestFFT'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(X.shape[0]):
        plt.figure()
        plt.plot(t, X[i])
        plt.xlabel(f'time[sec]')
        plt.ylabel(f'signal')
        plt.savefig(f'{path}/sig{i}.png')
        plt.close()

        plt.figure()
        plt.plot(freq, transformed_X[i])
        plt.xlabel(f'freqency[Hz]')
        plt.ylabel(f'amplitude')
        plt.savefig(f'{path}/freq{i}.png')
        plt.close()

from eblib.sig2vec import BPF
def TestBPF():
    N = 2**12
    t = np.linspace(0,1,N).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t),
            np.sin(2*np.pi*100*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    X_noised=X+np.random.randn(N)*0.3
    t = t.reshape(-1)

    bpf = BPF(fL = 30, fH = 120)
    transformed_X = bpf.transform(X_noised, t)

    path = f'./sig2vec/TestBPS'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(X.shape[0]):
        plt.figure()
        plt.plot(t[:100], X[i,:100],'blue', label='original')
        plt.plot(t[:100], X_noised[i,:100],'green', label='noised signal')
        plt.plot(t[:100], transformed_X[i,:100], 'red', label='bpf')
        plt.xlabel(f'time[sec]')
        plt.ylabel(f'signal')
        plt.legend()
        plt.savefig(f'{path}/sig{i}.png')
        plt.close()


from eblib.sig2vec import MFCC
def TestMFCC():
    N = 2**12
    t = np.linspace(0,1,N).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t),
            np.sin(2*np.pi*100*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    X_noised=X+np.random.randn(N)*0.3
    t = t.reshape(-1)

    mfcc = MFCC(numChannels=20)
    transformed_X = mfcc.transform(X_noised, t)
    print(transformed_X[0])

from eblib.sig2vec import STFT
def TestSTFT():
    N = 2**12
    t = np.linspace(0,1,N).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t),
            np.sin(2*np.pi*100*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    X_noised=X+np.random.randn(N)*0.3
    t = t.reshape(-1)

    stft = STFT(fs = 1/(t[1]-t[0]), window = 'hann', nperseg = 256, noverlap=None)
    print(X_noised.shape)
    transformed_X, _, _ = stft.transform(X_noised)
    print(transformed_X.shape)

from eblib.sig2vec import Sig2CNN
def TestSig2CNN():
    N = 2**12
    t = np.linspace(0,1,N).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t),
            np.sin(2*np.pi*100*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    X_noised=X+np.random.randn(N)*0.3
    t = t.reshape(-1)

    sig2cnn = Sig2CNN()
    transformed_X = sig2cnn.transform(t, X_noised)
    print(transformed_X.shape)

from eblib.sig2vec import Lyapunov1d
def TestLyapunov1d():
    N = 2**12
    t = np.linspace(0,1,N).reshape(1,-1)
    X = np.r_[np.sin(2*np.pi*50*t)*np.exp(t),
            np.sin(2*np.pi*100*t)*np.exp(-50*t),
            np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)]
    X_noised=X+np.random.randn(N)*0.3
    t = t.reshape(-1)
    """
    plt.figure()
    plt.plot(X[0])
    plt.savefig(f'./tmp_sig.png')
    plt.close()
    """
    lyapunov1d = Lyapunov1d(fs=1/(t[1]-t[0]))
    transformed_X = lyapunov1d.transform(X)
    print(transformed_X)

def main():
    #TestFFT()
    #TestBPF()
    #TestMFCC()
    #TestSTFT()
    #TestSig2CNN()
    TestLyapunov1d()

if __name__=='__main__':
    main()

