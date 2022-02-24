from tqdm import tqdm
import torch
from sklearn.metrics import r2_score
import numpy as np
import csv
import matplotlib.pyplot as plt


def evaluate_model(model,dataloader_test,device,criterion,exp):
    accloss = 0
    all_pred = np.zeros([])
    all_dbhs = np.zeros([])
    mse = 0
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(dataloader_test)), desc="EVAL"):
            dbhs, files, mfccs, spectrograms = next(iter(dataloader_test))
            dbhs = dbhs.view(-1,1).float()
            dbhs = dbhs.to(device)
            # mfccs = mfccs.to(device)
            # pred = model(mfccs)
            spectrograms = spectrograms.to(device)
            pred = model(spectrograms)
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