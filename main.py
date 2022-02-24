from dataset import TreeCutDataset
from eval import evaluate_model
from models import LumberjackNet
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn as nn
import argparse

from train import train

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


def main(training,n_epochs, n_mfcc,n_fft, hop_length, batch_size,lr=0.00001,momentum=0.9, weight_decay=0.01, p=0.5, exp='exp01', chan1=32, kernel=4):
    dataset_train = TreeCutDataset(is_train=True, n_mfcc=n_mfcc,n_fft=n_fft, hop_length=hop_length)
    dataset_test = TreeCutDataset(is_train=False, n_mfcc=n_mfcc,n_fft=n_fft, hop_length=hop_length)
    

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LumberjackNet(p=p, chan1=chan1, kernel=kernel, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft).to(device)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    
    # show_sample(model, device, dataloader_train)
    # show_mfcc(dataloader_train)

    if(training):
        train(model, device,criterion,optimizer,n_epochs,dataloader_train, dataloader_test,exp)
    
    # EVALUATE WITH BEST MODEL
    model.load_state_dict(torch.load(f'./{exp}/model/checkpoint.pth'))
    evaluate_model(model,dataloader_test,device,criterion,exp)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.000001, help="rmsprop: learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="rmsprop: momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.9, help="rmsprop: weight decay l2-norm regularization")
    # parser.add_argument("--p", type=float, default=0.25, help="dropout rate")
    parser.add_argument("--n_mfcc", type=int, default=40, help="number of coefficients")
    parser.add_argument("--n_fft", type=int, default=1024, help="size of the fft")
    parser.add_argument("--hop_length", type=int, default=512, help="window hop for mfcc")
    parser.add_argument("--exp", type=str, default='exp01', help="experiment name")
    parser.add_argument("--chan1", type=int, default=32, help="channels in cnn")
    parser.add_argument("--kernel", type=int, default=8, help="kernel size")
    opt = parser.parse_args()
    print(opt)

    os.makedirs(f'{opt.exp}', exist_ok=True)
    os.makedirs(f'{opt.exp}/logs', exist_ok=True)
    os.makedirs(f'{opt.exp}/model', exist_ok=True)
    with open(f'{opt.exp}/config.txt', 'w') as cfg:
        cfg.write(str(opt))

    main(training=True,
        n_epochs=opt.n_epochs,
        n_mfcc=opt.n_mfcc, 
        n_fft=opt.n_fft, 
        hop_length=opt.hop_length, 
        batch_size=opt.batch_size,
        lr=opt.lr,
        momentum=opt.momentum,
        exp=opt.exp,
        weight_decay=opt.weight_decay,
        # p=opt.p,
        chan1=opt.chan1,
        kernel=opt.kernel)
