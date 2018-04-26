from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import os
for i in os.listdir('dataset'):
	for j in os.listdir(os.path.join('dataset',i)):
		v=os.path.join('dataset',i,j)
		(sig,rate) = librosa.load(v);
		mfcc_feat = librosa.feature.mfcc(sig,rate,n_mfcc=26);
		chrome=librosa.feature.chroma_stft(sig,rate)
		contrast=librosa.feature.spectral_contrast(sig,rate)
		print(rate,mfcc_feat.shape,chrome.shape,contrast.shape)
