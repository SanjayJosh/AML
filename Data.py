import csv
import numpy as np
#from python_speech_features import mfcc
from split_list import getclasses,category_to_digit,digit_to_category,make_split
from keras.utils import to_categorical
#import scipy.io.wavfile as wav
import librosa
import os
class Data():
    all_classes=None
    class_dict=None
    trainlength= None
    testlength= None
    trainfile="trainlist-00.csv"
    testfile="testlist-00.csv"
    trainlist=None
    testlist=None
    def __init__(self,isinit=False):
        if isinit == True:
            self.trainfile,self.testfile=make_split(80,20,"11")
        self.all_classes = getclasses()
        self.class_num = len(self.all_classes)
        self.class_dict = digit_to_category()
    def make_path_lists(self):
        self.trainlist=list()
        with open(self.trainfile,"r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                self.trainlist.append([row[0],int(row[1])])
        self.trainlength=len(self.trainlist)
        self.testlist=list()
        with open(self.testfile,"r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                self.testlist.append([row[0],int(row[1])])
        self.testlength=len(self.testlist)
    def load_all_in_memory(self,listname):
        X=[]
        y=[]
        for each_file in listname:
            (sig,rate) = librosa.load(each_file[0]);
            mfcc_feat =librosa.feature.mfcc(sig,rate,n_mfcc=26)[:,:1024];
            chrome=librosa.feature.chroma_stft(sig,rate)[:,:1024];
            spectral_contrast=librosa.feature.spectral_contrast(sig,rate)[:,:1024]
            # contrast=librosa.feature.spectral_contrast(sig,rate)[:,:1290];
            #print(mfcc_feat.shape,type(mfcc_feat))
            X.append(np.concatenate((mfcc_feat,chrome,spectral_contrast)).T)
            y.append(to_categorical(each_file[1],self.class_num).squeeze())
        print(np.array(X).shape, np.array(y).shape)
        return np.array(X), np.array(y)
