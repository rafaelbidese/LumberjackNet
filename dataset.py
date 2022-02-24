
import json, os
from utils import json2obj
import numpy as np
import librosa
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile

def calcMeanStdDataset(dataset):
    trees = np.empty([1,1])
    for tree in dataset:
        trees = np.concatenate((trees, tree.mfccs.reshape(1,-1)), axis=None)
    print(np.mean(trees), np.std(trees))

class TreeData():
    def __init__(self, dbh, filepath, mfccs, spectrogram):
        self.dbh = dbh
        self.filepath = filepath
        self.mfccs = mfccs
        self.spectrogram = spectrogram

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

            _, samples = wavfile.read(filepath)
            frequencies, times, spectrogram = signal.spectrogram(
                samples,
                fs=48000,
                nfft=4096)


            # plt.pcolormesh(times, frequencies, spectrogram)
            # print(spectrogram.shape)
            # plt.imshow(spectrogram[:200,:])
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.show()

            self.dataset.append(TreeData(dbh, filepath, mfccs, spectrogram[:200,:]))
        
        # calcMeanStdDataset(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform is not None:
            data.mfccs = self.transform(data.mfccs)
        return data.dbh, data.filepath, data.mfccs, data.spectrogram

