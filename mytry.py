from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
for i in os.listdir('dataset'):
	for j in os.listdir(os.path.join('dataset',i)):
		v=os.path.join('dataset',i,j)
		(rate,sig) = wav.read(v);
		mfcc_feat = mfcc(sig,rate);
		print(rate,mfcc_feat.shape)
