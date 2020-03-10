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

from eblib.load_data import TestPiece
def TestTestPiece():
    x,y,d = TestPiece(material='all', domain_label = True, normalize=True, num_point=2**12).load()
    print(f'xsize = {x.shape}, x[5] = {x[5]}')
    print(f'ysize = {y.shape}, y[5] = {y[5]}')
    print(f'dsize = {d.shape}, d[5] = {d[5]}')

    path = f'./load_data/TestTestPiece'
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.figure()
    plt.plot(x[5],'blue')
    plt.savefig(f'{path}/TestTestPiece.jpg')
    plt.close()

from eblib.load_data import RealBridge
def TestRealBridge():
    mx, gx, y, d = RealBridge(place=-1,domain_label = True, normalize=True, num_point=2**12).load()
    print(f'mxsize = {mx.shape}, mx[5] = {mx[5]}')
    print(f'gxsize = {gx.shape}, gx[5] = {gx[5]}')
    print(f'ysize = {y.shape}, y[5] = {y[5]}')
    print(f'dsize = {d.shape}, d[5] = {d[5]}')

    path = f'./load_data/TestRealBridge'
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.figure()
    plt.plot(mx[5],'blue')
    plt.savefig(f'{path}/TestRealBridge_mic.jpg')
    plt.close()

    plt.figure()
    plt.plot(gx[5],'blue')
    plt.savefig(f'{path}/TestRealBridge_gmm.jpg')
    plt.close()


if __name__=='__main__':
    TestTestPiece()
    TestRealBridge()