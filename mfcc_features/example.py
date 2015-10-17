from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy

(rate,sig) = wav.read("FS01-10-AUTO-CRASH001.wav")
mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=20,nfilt=40,nfft=400,lowfreq=0,highfreq=8000,preemph=0.97,ceplifter=22,appendEnergy=True)

print mfcc_feat
#numpy.savetxt('xxcheck.txt',mfcc_feat)
#mfcc_feat = mfcc(sig,rate)
#fbank_feat = logfbank(sig,rate)

#print fbank_feat[1:3,:]
