from collections import namedtuple
import json

import librosa
import matplotlib.pyplot as plt

def _json_object_hook(d):
    return namedtuple('tree', d.keys())(*d.values())

def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)


def show_files(dataloader):
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, (dbhs, files, mfccs, spectrograms) in enumerate(dataloader):
        for j, (dbh, file, mfcc, spectrogram) in enumerate(zip(dbhs, files, mfccs, spectrograms)):
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
    dbh, filepath, mfccs, spectrograms = next(iter(dataloader_train))
    mfccs = mfccs[0].to(device)
    pred = model(mfccs)
    show_mfcc(dataloader_train)
    show_files(dataloader_train)