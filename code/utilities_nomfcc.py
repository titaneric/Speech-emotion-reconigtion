"""

author: harry-7

This file contains functions to read the data files from the given folders and generate the data interms of features
"""
import numpy as np
import torch
import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

dataset_folder = "dataset_chunk_nonsilence/"

class_labels = ["Neutral", "Angry", "Happy", "Sad"]

mslen = 320  # Empirically calculated for the given dataset


def get_data(flatten=True, mfcc_len=39):
    """
    Read the files get the data perform the test-train split and return them to the caller
    :param mfcc_len: Number of mfcc features to take for each frame
    :param flatten: Boolean specifying whether to flatten the data or not
    :return: 4 arrays, x_train x_test y_train y_test
    """
    data = []
    labels = []
    max_fs = 0
    min_sample = int('9' * 10)
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    os.chdir('..')
    os.chdir(dataset_folder)
    for i, directory in enumerate(class_labels):
        print("started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):  
            fs, signal = read_wav(filename) 
            max_fs = max(max_fs, fs)
            s_len = len(signal)
            #print(s_len)
            #if s_len < 6000:
                #continue
            if s_len < mslen:
                pad_len = mslen - s_len
                #print("pad_len: ",pad_len)
                pad_rem = pad_len % 2
                #print("pad_rem: ",pad_rem)
                pad_len //= 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
            else:
                pad_len = s_len - mslen
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = signal[pad_len:pad_len + mslen]
            min_sample = min(len(signal), min_sample)
            #print(signal.shape) # (32000,)
            #mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
            #print(mfcc.shape) # (198,39)

            if flatten:
                # Flatten the data
                mfcc = mfcc.flatten()
            data.append(signal)
            #data.append(mfcc)
            labels.append(i)
            cnt += 1
        print("ended reading folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    #print("total number of data: ", cnt)  339
    #training : testing = 8:2
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return torch.tensor(x_train), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_test)


def display_metrics(y_pred, y_true):
    print(accuracy_score(y_pred=y_pred, y_true=y_true))
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))


def read_wav(filename):
    """
    Read the wav file and return corresponding data
    :param filename: name of the file
    :return: return tuple containing sampling frequency and signal
    """
    return wav.read(filename)
