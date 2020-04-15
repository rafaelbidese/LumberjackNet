from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from sklearn.metrics import r2_score
from time import strftime
from tqdm import tqdm
import json, os, csv
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import argparse

from scipy import signal
from scipy.io import wavfile

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def _json_object_hook(d):
    return namedtuple('tree', d.keys())(*d.values())

def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def calcMeanStdDataset(dataset):
    trees = np.empty([1,1])
    for tree in dataset:
        trees = np.concatenate((trees, tree.mfccs.reshape(1,-1)), axis=None)
    print(np.mean(trees), np.std(trees))


class TreeData():
    def __init__(self, dbh, filepath, mfccs):
        self.dbh = dbh
        self.filepath = filepath
        self.mfccs = mfccs

class TreeCutDataset(Dataset):
    def __init__(self, is_train=True, n_mfcc=40, n_fft=1024, hop_length=512, transform=None):
        path = 'train_split.json' if is_train else 'val_split.json'
        with open(path, 'r') as json_file:
            json_data = json_file.read()
            trees = json.loads(json_data)
            dataset = json2obj(json.dumps(trees))
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.dataset = []

        for i,sample in enumerate(dataset):
            dbh = sample.DBH
            filepath = ''.join(['data',os.sep,sample.directory,os.sep,sample.fileName])
            data, sample_rate = librosa.load(filepath)
            # print(sample_rate)
            mfccs = librosa.feature.mfcc(y=data, sr=48000, n_fft=self.n_fft, n_mfcc = self.n_mfcc,hop_length=self.hop_length)
            # mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, dct_type=3, n_fft=self.n_fft, n_mfcc = self.n_mfcc,hop_length=self.hop_length)
            # mfccs = np.abs(librosa.core.stft(y=data, n_fft=self.n_fft, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect'))

            # sample_rate, samples = wavfile.read(filepath)
            # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

            # plt.pcolormesh(times, frequencies, spectrogram)
            # plt.imshow(spectrogram)
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.show()

            self.dataset.append(TreeData(dbh, filepath, mfccs))
        
        # calcMeanStdDataset(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform is not None:
            data.mfccs = self.transform(data.mfccs)
        return data.dbh, data.filepath, data.mfccs

def show_files(dataloader):
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, (dbhs, files, mfccs) in enumerate(dataloader):
        for j, (dbh, file, mfcc) in enumerate(zip(dbhs, files, mfccs)):
            fig.add_subplot(2, 2, j+1)
            plt.title(''.join([file, ',', str(dbh.item())]))
            data, sample_rate = librosa.load(file)
            librosa.display.waveplot(data, sr=sample_rate)
        plt.savefig('signal_examples.png')
        break

def show_mfcc(dataloader):
    plt.figure(figsize=(10,6))
    for i, (dbhs, files, mfccs) in enumerate(dataloader):
        for j, (dbh, file, mfcc) in enumerate(zip(dbhs, files, mfccs)):
            # print(mfcc.numpy().size)
            data, sample_rate = librosa.load(file)
            librosa.display.specshow(mfcc.numpy(), sr=22050, x_axis='time')
            plt.title(''.join([file, ',', str(dbh.item())]))
            plt.savefig('MFCCs.png')
        break

def show_sample(model, device, dataloader_train):
    dbh, filepath, mfccs = next(iter(dataloader_train))
    mfccs = mfccs[0].to(device)
    pred = model(mfccs)
    show_mfcc(dataloader_train)
    show_files(dataloader_train)

class LumberjackNet(nn.Module):
    def __init__(self,p=0.5,chan1=32,kernel=5, hop_length=512, n_mfcc=40, n_fft=512):
        super(LumberjackNet, self).__init__()
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.signal_length = 72050
        self.inputH = 141
        self.p = p
        self.kernel = kernel
        # channels, output, kernels 
        self.conv1 = nn.Conv2d(1,chan1,kernel)
        # kernel size, stride
        self.pool = nn.MaxPool2d(2,2) # chan1, n_mfcc-k+1, 141-k+1
        # # channels, output, kernels # 32,18,67  
        self.conv2W     = int((self.n_mfcc-self.kernel+1) / 2)
        self.conv2H     = int((self.inputH-self.kernel+1) / 2)
        self.conv2      = nn.Conv2d(chan1,self.conv2W,kernel) # channels, 40 - kernel%2 / pool, 141 - kernel%2 / pool
        self.fc1mul2    = int((self.conv2W-self.kernel+1)/2)
        self.fc1uml3    = int((self.conv2H-self.kernel+1)/2)
        self.fc1        = nn.Linear(self.conv2W*self.fc1mul2*self.fc1uml3,100)
        self.fc2        = nn.Linear(100,10)
        self.fc3        = nn.Linear(10,1)
        self.drop       = nn.Dropout(p=p)

    def forward(self,x):
        x=x.view(-1,1,self.n_mfcc,self.inputH)                # -> n, 1, 40, 141
        x = self.pool(F.relu(self.conv1(x))) # -> n, 6, 18, 68
        x = self.pool(F.relu(self.conv2(x))) # -> n, 18, 7, 32
        x = x.view(-1,self.conv2W*self.fc1mul2*self.fc1uml3)               # -> n, 18*7*32
        x = F.relu(self.fc1(x))              # -> n, 100
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 1
        return x

def train(model, device,criterion,optimizer,n_epochs,dataloader_train, dataloader_test, exp):
    # TODO: add train loss to plot
    all_mse = []
    all_r2 = []

    all_train_mse = []
    all_train_r2 = []
    best_model_r2 = -100;
    mse = 0
    for epoch in tqdm(range(n_epochs), desc="TRAIN"):
        model.train()
        accloss = 0
        all_train_pred = np.zeros([])
        all_train_dbhs = np.zeros([])
        for i, (dbhs, files, mfccs) in enumerate(dataloader_train):
            optimizer.zero_grad()
            # forward pass and loss
            dbhs = dbhs.view(-1,1).float()
            dbhs = dbhs.to(device)
            mfccs = mfccs.to(device)
            pred = model(mfccs)
            loss = criterion(pred, dbhs)
            # backward
            loss.backward()
            # updates
            optimizer.step()

            accloss += loss.cpu().detach().numpy()

            all_train_dbhs = np.append(all_train_dbhs, dbhs.cpu().detach().numpy()[:,0])
            all_train_pred = np.append(all_train_pred,  pred.cpu().detach().numpy()[:,0])
    
            

        mse = accloss.sum() / len(dataloader_train)
        r2score = r2_score(all_train_dbhs[1:], all_train_pred[1:])
        all_train_r2.append(r2score)
        all_train_mse.append(mse)
        time = strftime("%Y-%m-%d %H:%M:%S")
        if (epoch+1)%1 == 0:
            print(f'\n{time} [TRAIN] {epoch+1}] mse = {mse:.4f} r2score = {r2score:.4f}')
    
    # torch.save(model.state_dict(), f'./{exp}/model/checkpoint.pth')

        accloss = 0
        all_pred = np.zeros([])
        all_dbhs = np.zeros([])
        ## VALIDATION PER EPOCH
        with torch.no_grad():
            model.eval()
            for i, (dbhs, files, mfccs) in enumerate(dataloader_test):
                dbhs = dbhs.view(-1,1).float()
                dbhs = dbhs.to(device)
                mfccs = mfccs.to(device)
                pred = model(mfccs)
                loss = criterion(pred, dbhs)
                accloss += loss.cpu().numpy()
                all_dbhs = np.append(all_dbhs, dbhs.cpu().numpy()[:,0])
                all_pred = np.append(all_pred,  pred.cpu().numpy()[:,0])

            mse = accloss.sum() / len(dataloader_test)
            r2score = r2_score(all_dbhs[1:], all_pred[1:])
            # all_losses.append(mse.cpu().numpy())
            all_r2.append(r2score)
            all_mse.append(mse)

        if (r2score > best_model_r2):
            torch.save(model.state_dict(), f'./{exp}/model/checkpoint.pth')
            best_model_r2 = r2score

        time = strftime("%Y-%m-%d %H:%M:%S")
        if (epoch+1)%1 == 0:
            print(f'{time} [TRAIN VAL] {epoch+1}] mse = {mse:.4f} r2score = {r2score:.4f}')

    # LOG VAL R2 SCORES
    with open(f'./{exp}/logs/train_val_r2_loss_log.txt', 'w') as log:
        wr = csv.writer(log)
        wr.writerows(zip(all_r2))
    plt.figure()
    plt.plot(all_r2, 'bo')
    plt.plot(all_train_r2, 'ro')
    plt.title("TRAIN VAL R2 LOSS")
    plt.savefig(f'./{exp}/logs/train_val_r2_loss_log.png')

    # LOG VAL MSE SCORES
    with open(f'./{exp}/logs/train_val_mse_loss_log.txt', 'w') as log:
        wr = csv.writer(log)
        wr.writerows(zip(all_mse))
    plt.figure()
    plt.plot(all_mse, 'bo')
    plt.plot(all_train_mse, 'ro')
    plt.title("TRAIN VAL MSE LOSS")
    plt.savefig(f'./{exp}/logs/train_mse_loss_log.png')

def evaluate_model(model,dataloader_test,device,criterion,exp):
    accloss = 0
    all_pred = np.zeros([])
    all_dbhs = np.zeros([])
    mse = 0
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(dataloader_test)), desc="EVAL"):
            dbhs, files, mfccs = next(iter(dataloader_test))
            dbhs = dbhs.view(-1,1).float()
            dbhs = dbhs.to(device)
            mfccs = mfccs.to(device)
            pred = model(mfccs)
            loss = criterion(pred, dbhs)
            accloss += loss
            all_dbhs = np.append(all_dbhs, dbhs.cpu().numpy()[:,0])
            all_pred = np.append(all_pred, pred.cpu().numpy()[:,0])

        mse = accloss.sum() / len(dataloader_test)
        r2score = r2_score(all_dbhs[1:], all_pred[1:])
        print(f"Best model evaluation scores R2: {r2score:.4f} MSE: {mse:.4f}")

    with open(f'./{exp}/logs/eval_log.txt', 'w') as log:
        gt_pred = [all_dbhs[1:], all_pred[1:]]
        wr = csv.writer(log)
        wr.writerows(zip(*gt_pred))

    x = np.linspace(1,20,10)
    plt.figure()
    plt.plot(all_dbhs[1:], all_pred[1:], 'bo')
    plt.plot(x,x)
    plt.savefig(f'./{exp}/logs/best_model_pred_log.png')

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
