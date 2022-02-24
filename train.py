from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from time import strftime
import torch
import csv
import matplotlib.pyplot as plt


def train(model, device,criterion,optimizer,n_epochs,dataloader_train, dataloader_test, exp):
    # TODO: add train loss to plot
    all_mse = []
    all_r2 = []

    all_train_mse = []
    all_train_r2 = []
    best_model_r2 = -100
    mse = 0
    for epoch in tqdm(range(n_epochs), desc="TRAIN"):
        model.train()
        accloss = 0
        all_train_pred = np.zeros([])
        all_train_dbhs = np.zeros([])
        for i, (dbhs, files, mfccs, spectrograms) in enumerate(dataloader_train):
            optimizer.zero_grad()
            # forward pass and loss
            dbhs = dbhs.view(-1,1).float()
            dbhs = dbhs.to(device)
            # mfccs = mfccs.to(device)
            # pred = model(mfccs)
            spectrograms = spectrograms.to(device)
            pred = model(spectrograms)
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
            for i, (dbhs, files, mfccs, spectrograms) in enumerate(dataloader_test):
                dbhs = dbhs.view(-1,1).float()
                dbhs = dbhs.to(device)
                # mfccs = mfccs.to(device)
                # pred = model(mfccs)
                spectrograms = spectrograms.to(device)
                pred = model(spectrograms)
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
