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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def _json_object_hook(d):
    return namedtuple('tree', d.keys())(*d.values())

def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)

class AudioDataset(Dataset):
    def __init__(self, is_train=True, n_mfcc=40, n_fft=1024, hop_length=512):
        path = 'train_split.json' if is_train else 'val_split.json'
        with open(path, 'r') as json_file:
            json_data = json_file.read()
            trees = json.loads(json_data)
            self.dataset = json2obj(json.dumps(trees))
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.dbh = self.dataset[index].DBH
        self.filepath = ''.join([self.dataset[index].directory,os.sep,self.dataset[index].fileName])
        data, sample_rate = librosa.load(self.filepath)
        self.mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_fft=self.n_fft, n_mfcc = self.n_mfcc,hop_length=self.hop_length)
        return self.dbh, self.filepath, self.mfccs

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
    plt.figure(figsize=(8,8))
    for i, (dbhs, files, mfccs) in enumerate(dataloader):
        for j, (dbh, file, mfcc) in enumerate(zip(dbhs, files, mfccs)):
            # print(mfcc.numpy().size)
            data, sample_rate = librosa.load(file)
            librosa.display.specshow(mfcc.numpy(), sr=sample_rate, x_axis='time')
            plt.title(''.join([file, ',', str(dbh.item())]))
            plt.savefig('MFCCs.png')
        break

def show_sample(model, device, dataloader_train):
    dbh, filepath, mfccs = next(iter(dataloader_train))
    mfccs = mfccs[0].to(device)
    pred = model(mfccs)
    show_mfcc(dataloader_train)
    show_files(dataloader_train)

class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        # channels, output, kernels 
        self.conv1 = nn.Conv2d(1,6,5)
        # kernel size, stride
        self.pool = nn.MaxPool2d(2,2)
        # # channels, output, kernels 
        self.conv2 = nn.Conv2d(6,18,5)
        self.fc1 = nn.Linear(18*7*32,100)
        self.fc2 = nn.Linear(100,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        x=x.view(-1,1,40,141)
        # -> n, 1, 40, 141
        x = self.pool(F.relu(self.conv1(x))) # -> n, 6, 18, 68
        x = self.pool(F.relu(self.conv2(x))) # -> n, 18, 7, 32
        x = x.view(-1,18*7*32)               # -> n, 18*7*32
        x = F.relu(self.fc1(x))              # -> n, 100
        x = F.relu(self.fc2(x))              # -> n, 84
        x = self.fc3(x)                      # -> n, 1
        return x

def train(model, device,criterion,optimizer,n_epochs,dataloader_train, dataloader_test):

    all_mse = []
    all_r2 = []
    best_model_r2 = 0;
    mse = 0
    for epoch in tqdm(range(n_epochs), desc="TRAIN"):
        for i, (dbhs, files, mfccs) in enumerate(dataloader_train):
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
            optimizer.zero_grad()

        accloss = 0
        all_pred = np.zeros([])
        all_dbhs = np.zeros([])
        ## VALIDATION PER EPOCH
        with torch.no_grad():
            for i, (dbhs, files, mfccs) in enumerate(dataloader_test):
                dbhs = dbhs.view(-1,1).float()
                dbhs = dbhs.to(device)
                mfccs = mfccs.to(device)
                pred = model(mfccs)
                loss = criterion(pred, dbhs)
                accloss += loss
                all_dbhs = np.append(all_dbhs, dbhs.cpu().numpy()[:,0])
                all_pred = np.append(all_pred,  pred.cpu().numpy()[:,0])

        mse = accloss.sum() / len(dataloader_test)
        r2score = r2_score(all_dbhs[1:], all_pred[1:])
        # all_losses.append(mse.cpu().numpy())
        all_r2.append(r2score)
        all_mse.append(mse)

        if (r2score > best_model_r2):
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), './model/checkpoint.pth')
            best_model_r2 = r2score

        time = strftime("%Y-%m-%d %H:%M:%S")
        if (epoch+1)%1 == 0:
            print(f'\n{time} [EPOCH {epoch+1}] mse = {mse:.4f} r2score = {r2score:.4f}')

    os.makedirs('logs', exist_ok=True)
    # LOG R2 SCORES
    with open('./logs/train_r2_loss_log.txt', 'w') as log:
        wr = csv.writer(log)
        wr.writerows(zip(all_r2))
    plt.figure()
    plt.plot(all_r2, 'bo')
    plt.title("TRAIN R2 LOSS")
    plt.savefig('./logs/train_r2_loss_log.png')

    # LOG MSE SCORES
    with open('./logs/train_mse_loss_log.txt', 'w') as log:
        wr = csv.writer(log)
        wr.writerows(zip(all_mse))
    plt.figure()
    plt.plot(all_mse, 'bo')
    plt.title("TRAIN MSE LOSS")
    plt.savefig('./logs/train_mse_loss_log.png')

def evaluate_model(model,dataloader_test,device,criterion):
    accloss = 0
    all_pred = np.zeros([])
    all_dbhs = np.zeros([])
    mse = 0
    with torch.no_grad():
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

    os.makedirs('logs', exist_ok=True)
    with open('./logs/eval_log.txt', 'w') as log:
        gt_pred = [all_dbhs[1:], all_pred[1:]]
        wr = csv.writer(log)
        wr.writerows(zip(*gt_pred))

    x = np.linspace(1,20,10)
    plt.figure()
    plt.plot(all_dbhs[1:], all_pred[1:], 'bo')
    plt.plot(x,x)
    plt.savefig('./logs/best_model_pred_log.png')

def main(training,n_epochs, n_mfcc,n_fft, hop_length, batch_size,lr=0.00001,momentum=0.9):
    dataset_train = AudioDataset(is_train=True, n_mfcc=n_mfcc,n_fft=n_fft, hop_length=hop_length)
    dataset_test = AudioDataset(is_train=False, n_mfcc=n_mfcc,n_fft=n_fft, hop_length=hop_length)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AudioNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    
    # show_sample(model, device, dataloader_train)
    if(training):
        model.train()
        train(model, device,criterion,optimizer,n_epochs,dataloader_train, dataloader_test)
    
    # EVALUATE WITH BEST MODEL
    model.load_state_dict(torch.load('./model/checkpoint.pth'))
    model.eval()
    evaluate_model(model,dataloader_test,device,criterion)
    

if __name__ == '__main__':
    main(training=True,n_epochs=100,n_mfcc=40, n_fft=1024, hop_length=512, batch_size=16,lr=0.00001,momentum=0.9)
