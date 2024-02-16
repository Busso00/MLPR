import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from MVG.pdf import *

def MVGNaiveBayes(trainDataV,trainLabelV):#compute only variances
    N_ATTRS = trainDataV.shape[0]
    #muc = mean for all classes (must differ)
    muc=numpy.zeros((N_ATTRS,1,N_LABELS))
    for i in range(N_LABELS):
        muc[:,0,i]=numpy.mean(trainDataV[:,trainLabelV==i],axis=1)
    #Cc = covariance matrix (only diagonal)
    Cc=numpy.zeros((N_ATTRS,N_LABELS))
    for i in range(N_LABELS):#cost N_ATTRxN_DATA
        Cc[:,i]=numpy.var(trainDataV[:,trainLabelV==i],axis=1)
    return (muc,Cc)
    
def inferClassMVGNaiveBayes(testDataV,testLabelV,muc,Cc,priorProb=PC):#more accurate features of my data are uncorrelated (converge with less training data)
    N_RECORDS = testDataV.shape[1]
    
    NormalPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) pdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalPdfV[i,:]=GAU_ND_pdf_naiveBayes(testDataV,muc[:,:,i],Cc[:,i])
    
    SJointV=NormalPdfV*vcol(numpy.array(priorProb))
    SMargV=vrow(numpy.sum(SJointV,axis=0))    
    SPostV=SJointV/SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)

    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)

def inferClassMVGLogNaiveBayes(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]
    
    NormalLogPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) logpdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalLogPdfV[i,:]=GAU_ND_logpdf_naiveBayes(testDataV,muc[:,:,i],Cc[:,i])
    
    SJointV=NormalLogPdfV+numpy.log(vcol(numpy.array(priorProb)))
    SMargV=scipy.special.logsumexp(SJointV,axis=0)
    SPostV=SJointV-SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)
    
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)