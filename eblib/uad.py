##############################################
##############教師なし異常検知器##############
##############################################

import numpy as np
import sys

class SM(object):
    def __init__(self, eta = 0.99, r = None):
        self.eta = eta
        self.r = r
    
    def fit(self, train_X):
        #共分散行列の作成
        A = train_X.T@train_X/train_X.shape[0]
        #固有値問題
        e_val, e_vec = np.linalg.eigh(A)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        #0以上の固有値選択
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        #部分空間の次元数の決定
        if self.r is None:
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1, len(e_val)+1)])
            self.r = int(np.min(np.where(sum_value>=self.eta)[0]))+1
        if self.r>=train_X.shape[1]:
            self.r = train_X.shape[1]-1
        #基底ベクトルの選択
        self.coef_, self.components_ = e_val[:self.r], e_vec.T[:self.r].T
    
    def decision_function(self, test_X, score_mode = 'recon'):
        if score_mode == 'recon':
            return self._decision_recon(test_X)
        elif score_mode=='sin':
            return self._decision_sin(test_X)
        else:
            print(f'Argment error... -> reconstruction error')
            return self._decision_recon(test_X)

    def _decision_recon(self, test_X):
        #print(f'Reconstruction error...')
        #再構成誤差
        I = np.eye(test_X.shape[1])
        return np.linalg.norm(test_X@(I-self.components_@self.components_.T).T, axis=1)

    def _decision_sin(self, test_X):
        #print(f'Sine value error...')
        #sinθ
        recon_error = self._decision_recon(test_X)
        norm_X = np.linalg.norm(test_X, axis=1).reshape(-1)
        recon_error[norm_X>0] /= norm_X[norm_X>0]
        return recon_error
            
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
kernel_list = ['linear', 'poly', 'rbf']
class KSM(object):
    def __init__(self, kernel = 'rbf', d = 3, gamma=3.0, eta = 0.99, r = None):
        if not(kernel in kernel_list):
            print(f'kernel error...')
            sys.exit()
        self.kernel = kernel
        self.d = d
        self.gamma = gamma
        self.eta = eta
        self.r = r

    def _kernel_matrix(self, X, Y=None):
        if self.kernel=='linear':
            return linear_kernel(X,Y)
        elif self.kernel=='poly':
            return polynomial_kernel(X, Y, degree=self.d, gamma=1, coef0=1)
        else:
            return rbf_kernel(X, Y, gamma=self.gamma)
    
    def fit(self, train_X):
        self.train_X = train_X
        #カーネル行列の生成
        K = self._kernel_matrix(train_X)
        #固有値問題
        e_val, e_vec = np.linalg.eigh(K/train_X.shape[0])
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        #0以上の固有値選択
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        #部分空間の次元数の決定
        if self.r is None:
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1, len(e_val)+1)])
            self.r = int(np.min(np.where(sum_value>=self.eta)[0]))+1
        if self.r>=train_X.shape[1]:
            self.r = train_X.shape[1]-1
        #基底ベクトルの選択
        self.coef_, self.components_ = e_val[:self.r], e_vec.T[:self.r].T
    
    def decision_function(self, test_X, score_mode = 'recon'):
        if score_mode == 'recon':
            return self._decision_recon(test_X)
        elif score_mode=='sin':
            return self._decision_sin(test_X)
        else:
            print(f'Argment error... -> reconstruction error')
            return self._decision_recon(test_X)

    def _decision_recon(self, test_X):
        ee_K = self._kernel_matrix(test_X)
        ee_K = np.diag(ee_K)
        et_K = self._kernel_matrix(test_X, self.train_X)
        tmp = et_K@self.components_/np.sqrt(self.coef_*self.train_X.shape[0]).reshape(1,-1)
        return ee_K.reshape(-1)-(np.linalg.norm(tmp, axis=1)**2).reshape(-1)

    def _decision_sin(self, test_X):
        ee_K = self._kernel_matrix(test_X)
        ee_K = np.diag(ee_K).reshape(-1)
        recon_error = self._decision_recon(test_X)
        return recon_error/ee_K


import torch
import torch.nn as nn
import torch.nn.functional as F
class DaeModel(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.depth = len(layers)
        self.units = layers
        
        #Encode layers
        for i in range(self.depth-1):
            layer=nn.Linear(self.units[i], self.units[i+1])
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            setattr(self, f'encoder{i+1}', layer)
        
        #Decode layers
        for i in range(self.depth-1):
            layer=nn.Linear(self.units[self.depth-i-1], self.units[self.depth-i-2])
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            setattr(self, f'decoder{i+1}', layer)

    def forward(self,x):
        x=F.relu(self.encode(x))
        x=self.decode(x)
        return x
    
    def encode(self, x=None, layer=None):
        if layer is None:
            mark_pos=self.depth-1
        else:
            mark_pos=layer
        
        encode_layers=[f'encoder{i+1}' for i in range(mark_pos)]

        #flatten inputs to vector
        x=x.view(-1, self.num_flat_feature(x))
        for idx, layer in enumerate(encode_layers[:-1]):
            x=getattr(self, layer)(x)
            x=F.relu(x)
        x=getattr(self, encode_layers[-1])(x)
        return x

    def decode(self, x=None, layer=None):
        if layer is None:
            mark_pos=self.depth-1
        else:
            mark_pos=layer

        decode_layers=[f'decoder{i+1}' for i in range(mark_pos)]
        for idx, layer in enumerate(decode_layers[:-1]):
            x=getattr(self, layer)(x)
            x=F.relu(x)
        x=getattr(self, decode_layers[-1])(x)
        return x

    def num_flat_feature(self, x):
        size=x.size()[1:]
        dim=1
        for s in size:
            dim *= s
        return dim

import torch.utils.data
from torch.autograd import Variable
class DAE(object):
    def __init__(self, max_epoch=20, lr=1e-5, batch_size=128, weight_decay=0, dims=[784, 100, 10], seed=-1):
        if(seed>=0):
            np.random.seed(seed)
            torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device("cpu")

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model = DaeModel(*dims).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, train_X):
        self.loss_list = []
        for epoch in range(self.max_epoch):
        #for epoch in range(self.max_epoch):
            self.loss_list.append(self._epoch_procedure(train_X))

    def fit_evaluate(self, train_X, test_X):
        self.train_loss_list = []
        self.test_loss_list = []
        for epoch in range(self.max_epoch):
            self.train_loss_list.append(self._epoch_procedure(train_X))
            self.test_loss_list.append(self._cal_loss(test_X))
    
    def _epoch_procedure(self, X):
        loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=self.batch_size, shuffle=True)

        running_loss = 0.0
        for idx, (inputs) in enumerate(loader):
            inputs = Variable(inputs.to(self.device)).float()

            output = self.model(inputs)
            loss = self.criterion(output, inputs)
            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return running_loss/len(loader)

    def reconstruction(self, X):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs = Variable(inputs.to(self.device)).float()
                output = self.model(inputs)
                output = output.cpu().data.numpy()
                if batch_idx==0:
                    output_X = output.reshape(1,-1)
                else:
                    output_X = np.r_[output_X, output.reshape(1,-1)]
        return output_X

    def encoder(self, X):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs = Variable(inputs.to(self.device)).float()
                output = self.model.encode(inputs)
                output = output.cpu().data.numpy()
                if batch_idx==0:
                    output_X = output.reshape(1,-1)
                else:
                    output_X = np.r_[output_X, output.reshape(1,-1)]
        return output_X

    def _cal_loss(self, X):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=1, shuffle=False)
        output_loss = []
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs = Variable(inputs.to(self.device)).float()
                output = self.model(inputs)
                loss = self.criterion(output, inputs)
                output_loss.append(loss.item())
        return np.array(output_loss)

    def decision_function(self, X):
        recon_X = self.reconstruction(X)
        S = np.linalg.norm(X-recon_X, axis=1)**2
        return S

from sklearn.decomposition import NMF
class CSM(object):
    def __init__(self, n_components = None, max_iter=200):
        self.n_components = n_components
        self.max_iter = max_iter
    
    def fit(self, train_X):
        self.nmf = NMF(n_components=self.n_components, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=self.max_iter, random_state=0, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False)
        self.nmf.fit(train_X)

    def decision_function(self, test_X, score_mode = 'recon'):
        if score_mode == 'recon':
            return self._decision_recon(test_X)
        elif score_mode=='sin':
            return self._decision_sin(test_X)
        else:
            print(f'Argment error... -> reconstruction error')
            return self._decision_recon(test_X)

    def _decision_recon(self, test_X):
        test_W = self.nmf.transform(test_X)
        return np.linalg.norm(test_X-test_W@self.nmf.components_, axis=1)

    def _decision_sin(self, test_X):
        recon_error = self._decision_recon(test_X)
        return recon_error/np.linalg.norm(test_X, axis=1).reshape(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGANDiscriminator, self).__init__()

        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
             # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer5 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feature = out.view(out.size(0), -1)
        out = self.layer5(out)
        return out, feature

class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
class MyDatasets(torch.utils.data.Dataset):
    def __init__(self, train_X, transform = None):
        self.transform = transform
        self.data = train_X
        self.data_num = train_X.shape[0]
    def __len__(self):
        return self.data_num
    def __getitem__(self, idx):
        out_data = self.data[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data


class AnoGAN(object):
    def __init__(self, max_epoch=5, lr=0.0002, batch_size=128, img_size=64, nc=3, ndf=64, ngf=64, nz=100):
        np.random.seed(0)
        torch.manual_seed(0)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device("cpu")

        self.max_epoch = max_epoch
        self.batch_size = batch_size

        self.netD = DCGANDiscriminator(nc=nc, ndf=ndf).to(self.device)
        self.netG = DCGANGenerator(nz=nz, ngf=ngf, nc=nc).to(self.device)

        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.real_label = 1
        self.fake_label = 0
        self.nz = nz
        self.fixed_nose = torch.randn(64, nz, 1, 1, device = self.device)
    
    def fit(self, train_X):
        dataset = MyDatasets(train_X, self.data_transforms) #train_X ＝＞ num_data×height×width×channel
        self.loss_G, self.loss_D = [], []
        for epoch in range(self.max_epoch):
            lossG, lossD = self._epoch_procedure(epoch, dataset)
            self.loss_G.append(lossG)
            self.loss_D.append(lossD)

    def _epoch_procedure(self, epoch, dataset):
        loader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size, shuffle=True)
        running_lossD = 0.0
        running_lossG = 0.0
        for batch_idx, inputs in enumerate(loader):
            inputs = Variable(inputs.to(self.device)).float()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            self.netD.zero_grad()
            self.optimizerD.zero_grad()
            #real image
            b_size = inputs.size(0)
            label = torch.full((b_size, ), self.real_label, device = self.device)
            output, _ = self.netD(inputs)
            output = output.view(-1)
            errD_real = self.criterion(output, label)         
            errD_real.backward(retain_graph=True)

            #fake image
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            fake_img = self.netG(noise)
            label.fill_(self.fake_label)
            output, _ = self.netD(fake_img)
            output = output.view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward(retain_graph=True)
            errD = errD_real+errD_fake
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            self.optimizerG.zero_grad()
            label.fill_(self.real_label)
            output, _ = self.netD(fake_img)
            output = output.view(-1)
            errG = self.criterion(output, label)
            errG.backward(retain_graph=True)
            self.optimizerG.step()

            ###########################
            #表示######################
            ###########################
            print(f'[{epoch+1}/{self.max_epoch}][{batch_idx+1}/{len(loader)}] lossD : {errD.item()}, lossG : {errG.item()}')
            running_lossD+=errD.item()
            running_lossG+=errG.item()
        return running_lossG, running_lossD
            

    def check_img(self,name='tmp'):
        with torch.no_grad():
            fake_img = self.netG(self.fixed_nose).detach().cpu()
            image = vutils.make_grid(fake_img, padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(image,(1,2,0)))
        plt.savefig(f'./{name}.png')
        plt.close()


    def Anomaly_score(self, x, G_z, Lvalue=0.1):
        _, x_feature = self.netD(x)
        _, G_z_feature = self.netD(G_z)

        residual_loss = torch.sum(torch.abs(x-G_z))
        discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature))

        total_loss = (1-Lvalue)*residual_loss+Lvalue*discrimination_loss
        return total_loss
    

    def decision_function(self, test_X, max_iter=5000, lr_score=1e-4):
        test_dataset = MyDatasets(test_X, self.data_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.netG.eval()
        score_list = []
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(self.device).float()
            z = torch.randn(1, self.nz, 1, 1, device = self.device)
            z_optimizer = torch.optim.Adam([z], lr=lr_score)

            for i in range(max_iter):
                with torch.no_grad():
                    gen_x = self.netG(z)
                loss = self.Anomaly_score(inputs, gen_x, Lvalue=0.01)
                z_optimizer.zero_grad()
                loss.backward()
                z_optimizer.step()
            with torch.no_grad():
                gen_x = self.netG(z)
                score = self.Anomaly_score(inputs, gen_x, Lvalue=0.01)
            score_list.append(score.item())
            print(f'Test[{batch_idx+1}/{len(test_loader)}] : {score_list[-1]}')
        return np.array(score_list)

import cvxopt
from cvxopt import matrix
class SVDD(object):
    def __init__(self, kernel='linear', d=2, gamma=3.0, C = 10):
        if not(kernel in kernel_list):
            print(f'kernel error...')
            sys.exit()
        self.kernel = kernel
        self.d = d
        self.gamma = gamma
        self.C = C

        cvxopt.solvers.options['abstol']=1e-15
        cvxopt.solvers.options['reltol']=1e-15
        cvxopt.solvers.options['feastol']=1e-15
        cvxopt.solvers.options['maxiters']=500

    def _kernel_matrix(self, X, Y=None):
        if self.kernel=='linear':
            return linear_kernel(X,Y)
        elif self.kernel=='poly':
            return polynomial_kernel(X, Y, degree=self.d, gamma=1, coef0=1)
        else:
            return rbf_kernel(X, Y, gamma=self.gamma)

    def _diag_kernel(self, X):
        K = self._kernel_matrix(X)
        return np.diag(K)

    def fit(self, train_X):
        #内積行列を作成
        K = self._kernel_matrix(train_X)
        self.train_X=train_X
        
        #cvxopt用の変数
        P = matrix(2*K, K.shape, 'd')
        q = matrix(np.diag(K).reshape(-1,1), (K.shape[0],1), 'd')
        A = matrix(np.ones(K.shape[0]).reshape(1,-1),(1,K.shape[0]),'d')
        b = matrix(np.array([1]),(1,1),'d')
        G1 = (-1)*np.identity(K.shape[0])
        h1 = np.zeros(K.shape[0]).reshape(-1,1)
        G2 = np.identity(K.shape[0])
        h2 = self.C*np.ones(K.shape[0]).reshape(-1,1)
        G = matrix(np.r_[G1, G2],(2*K.shape[0], K.shape[0]), 'd')
        h = matrix(np.r_[h1, h2],(2*K.shape[0],1),'d')

        self.sol = cvxopt.solvers.qp(P=P, q=q, A=A, b=b, G=G, h=h)
        self.alpha = np.array(self.sol['x']).reshape(-1)
        if(self.C > 1):
            idx = np.where(self.alpha<1e-3)
        else:
            idx = np.where(self.alpha<self.C*1e-4)
        self.alpha[idx] = 0
        self.sv_index = np.where(self.alpha>0)
        self.sv = train_X[self.sv_index]
        self.norm_a = self.alpha.reshape(1,-1)@K@self.alpha.reshape(-1,1)
        
    def decision_function(self,X):
        test_Kii = self._diag_kernel(X)
        test_K = self._kernel_matrix(X, self.train_X)
        return test_Kii.reshape(-1,1)-2*test_K@self.alpha.reshape(-1,1)-self.norm_a

from sklearn.svm import OneClassSVM
class OCSVM(object):
    def __init__(self, kernel='rbf', d = 2, gamma = 3.0, nu = 0.1):
        self.kernel = kernel
        self.d = d
        self.gamma = gamma
        if(self.kernel == 'poly'):
            self.gamma = 1
        self.nu = nu

    def fit(self, train_X):
        if self.gamma=='auto':
            self.model = OneClassSVM(kernel = self.kernel,
                                    degree=self.d,
                                    gamma='scale',
                                    coef0=1
                                    )
        else:
            self.model = OneClassSVM(kernel = self.kernel,
                                    degree=self.d,
                                    gamma=self.gamma,
                                    coef0=1
                                    )
        self.model.fit(train_X)
    
    def decision_function(self, X):
        return (-1)*self.model.score_samples(X)#OCSVMは±0が識別面でプラス側が学習データ(正常)になるため

class VaeModel(nn.Module):
    def __init__(self, *dims):
        super().__init__()

        self.depth=len(dims)
        self.units=dims

        #Encode layers
        for i in range(self.depth-2):
            layer=nn.Linear(self.units[i], self.units[i+1])
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            setattr(self, f'encoder{i+1}', layer)
        layer_mu=nn.Linear(self.units[self.depth-2], self.units[self.depth-1])

        #Bottle neck
        nn.init.kaiming_normal_(layer_mu.weight, mode='fan_in', nonlinearity='relu')
        setattr(self, f'encoder{self.depth-1}{1}', layer_mu)
        
        layer_sigma=nn.Linear(self.units[self.depth-2], self.units[self.depth-1])
        nn.init.kaiming_normal_(layer_sigma.weight, mode='fan_in', nonlinearity='relu')
        setattr(self, f'encoder{self.depth-1}{2}', layer_sigma)


        #Decode layers
        for i in range(self.depth-1):
            layer=nn.Linear(self.units[self.depth-i-1], self.units[self.depth-i-2])
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            setattr(self, f'decoder{i+1}', layer)

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x=None, layer=None):
        if layer is None:
            mark_pos = self.depth-1
            encode_layers=[f'encoder{i+1}' for i in range(mark_pos-1)]
        else:
            mark_pos = layer
            encode_layers=[f'encoder{i+1}' for i in range(mark_pos)]

        for layer in encode_layers:
            x = getattr(self, layer)(x)
            x = F.relu(x)
        
        if(mark_pos != self.depth-1):
            return x
        return getattr(self, f'encoder{self.depth-1}{1}')(x), getattr(self,f'encoder{self.depth-1}{2}')(x)

    def decode(self, x=None, layer=None):
        if layer is None:
            mark_pos=self.depth-1
        else:
            mark_pos=layer

        decode_layers=[f'decoder{i+1}' for i in range(mark_pos)]

        for layer in decode_layers[:-1]:
            x=getattr(self, layer)(x)
            x=F.relu(x)
        x=getattr(self, decode_layers[-1])(x)
        return torch.sigmoid(x)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VAE(DAE):
    def __init__(self, max_epoch=20, lr=1e-5, batch_size=128, weight_decay=0, dims=[784, 100, 10], seed=-1):
        super().__init__()
        if(seed>=0):
            np.random.seed(seed)
            torch.manual_seed(seed)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device("cpu")

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model = VaeModel(*dims).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _epoch_procedure(self, X):
        loader = torch.utils.data.DataLoader(torch.from_numpy(X),batch_size=self.batch_size, shuffle=True)

        running_loss = 0.0
        for idx, (inputs) in enumerate(loader):
            torch.autograd.set_detect_anomaly(True)
            inputs = Variable(inputs.to(self.device)).float()

            output, mu, logvar = self.model(inputs)
            loss = loss_function(output, inputs, mu, logvar)
            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return running_loss/len(loader)

    def reconstruction(self, X):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs = Variable(inputs.to(self.device)).float()
                output, _, _ = self.model(inputs)
                output = output.cpu().data.numpy()
                if batch_idx==0:
                    output_X = output.reshape(1,-1)
                else:
                    output_X = np.r_[output_X, output.reshape(1,-1)]
        return output_X


    def _cal_loss(self, X):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=1, shuffle=False)
        output_loss = []
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                inputs = Variable(inputs.to(self.device)).float()
                output, _, _ = self.model(inputs)
                loss = loss_function(output, inputs)
                output_loss.append(loss.item())
        return np.array(output_loss)

    def decision_function(self, X):
        recon_X = self.reconstruction(X)
        S = np.linalg.norm(X-recon_X, axis=1)**2
        return S

