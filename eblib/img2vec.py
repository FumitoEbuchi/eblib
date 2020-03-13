##############################################
#############画像データの特徴抽出#############
##############################################

import numpy as np
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
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

class CNNFeature(object):
    def __init__(self, normalize = False, seed=-1):
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
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.normalize = normalize

    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            dataset = MyDatasets(X, transform=self.data_transforms)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            for batch_idx, inputs in enumerate(loader):
                inputs = Variable(inputs.to(self.device)).float()
                outputs = self.model(inputs).cpu().data.numpy().reshape(1,-1)
                if batch_idx==0:
                    transformed_X = outputs
                else:
                    transformed_X = np.r_[transformed_X, outputs]
                

                ############################################
                ##画像を変換できているか確認するスクリプト##
                ############################################
                """
                if batch_idx == 0:
                    plt.figure()
                    plt.gray()
                    plt.imshow(inputs.cpu().data.numpy()[0].transpose(1,2,0))
                    plt.savefig(f'./tmp.png')
                """
        if self.normalize is True:
            norm_transformed_X = np.linalg.norm(transformed_X, axis=1).reshape(-1, 1)
            transformed_X/=norm_transformed_X
        return transformed_X

