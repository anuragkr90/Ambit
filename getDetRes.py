from features import mfcc
from features.sigproc import framesig
import scipy.io.wavfile as wav
import numpy as np
import os
import sys
from svmutil import *
from sklearn import metrics
from sklearn.externals import joblib
import math

# Contains a lot of redundant parts

fs = 44100 #keeping same as of now
wlen = 0.020 # in seconds
wstep = 0.010 # in seconds
ncep = 21 #
fftsz = int(2**(math.ceil(math.log(wlen*fs,2))))
numfilt = 40

seglen = 1.0 # in seconds
segstp = 0.5 # in seconds same

GMMPath='GMMModels'
gModnm='pythonGMM128'
SVMPath='classModls'
sModnm='pythonMod128'

gmmmixt = joblib.load(os.path.join(GMMPath,gModnm+'.pkl'))
nComp=128
svmMod = svm_load_model(os.path.join(SVMPath,sModnm))
trPath = 'trainD'
trdata = np.loadtxt(os.path.join(trPath,'TrainDataSv'))

kergama = 1.0

opPoint = 0.0285 # operating point
pOrc = 0 # probability 

def computeKernel(X,Y,kergama):
    kerdis = metrics.pairwise.chi2_kernel(X,Y,kergama)
    return kerdis

def doDet(fileIn):
    (fsr,sig) = wav.read(fileIn)
    segChunks,segTimes = framesig(sig,seglen*fs,segstp*fs,'box',1,fs)
    
    if len(segChunks.shape) == 1:
        segChunks = np.reshape(segChunks,1,segChunks.reshape[0])
        
    allOut = np.zeros((segChunks.shape[0],1))
    #print allOut.shape
    for t in range(segChunks.shape[0]):
        seg = segChunks[t,:]
        mfcc_feat,mspec,logmelspec = mfcc(seg,fs,winlen=wlen,winstep=wstep,numcep=ncep,nfilt=numfilt,nfft=fftsz,lowfreq=0,highfreq=fs/2,preemph=0.97,ceplifter=22,appendEnergy=True)

        #if (math.isnan(numpy.sum(numpy.sum(mfcc_feat)))) or (math.isinf(numpy.sum(numpy.sum(mfcc_feat)))):
        #    print 'Escaping this Seg -- NaN or Inf occurres'
        #else:
        #    numpy.savetxt(fltoread.replace('AllData',mfccset).rstrip('.wav')+'_POSITIVE_'+tlist[0]+'_'+tlist[1]+'_'+str(t)+'.mfcc',mfc\
        #                      c_feat,delimiter=' ') 

        cdist=gmmmixt.predict_proba(mfcc_feat)
        hist = np.sum(cdist,axis=0)
        histfeat = hist/float(hist.shape[0])
        histfeat = histfeat.reshape(1,histfeat.shape[0])
        
        histKer = computeKernel(histfeat,trdata,1.0)
        kerId = np.arange(histfeat.shape[0])+1
        kerId = np.reshape(kerId,(kerId.shape[0],1))
        teKer = np.hstack((kerId,histKer))
        teKer = map(list,teKer)
        telax = [0]*len(teKer)
        plb,acc,probab = svm_predict(telax,teKer,svmMod,'-b 1 -q')
        
        
        lbs = svmMod.get_labels()
        #print str(probab) + str(lbs)
        probab = np.array(probab)
        if lbs[0] == 1:
            prob_f=probab[:,0]
        elif lbs[1] == 1:
            prob_f=probab[:,1]
        else:
            print 'Not possible'
            sys.exit()

        if pOrc == 1:
            allOut[t,0] = prob_f
        elif pOrc == 0:
            allOut[t,0] = int(prob_f > opPoint)
    #print np.hstack((allOut,segTimes))
    np.savetxt(fileIn.rstrip('.wav')+'_res'+'.txt',allOut)
    return np.hstack((allOut,segTimes))
                 
    
if __name__ == "__main__":
    res=doDet(sys.argv[1])
    print res
