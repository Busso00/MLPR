import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from MVG.pdf import *

def MVGTied(trainDataV,trainLabelV):#compute only one (unique for each class) covariance matrix
    N_ATTRS = trainDataV.shape[0]
    #muc = mean for all classes (must differ)
    muc=numpy.zeros((N_ATTRS,1,N_LABELS))
    for i in range(N_LABELS):
        muc[:,0,i]=numpy.mean(trainDataV[:,trainLabelV==i],axis=1)
    #Cc = covariance matrix (shared across all classes)
    Cc=withinClassCovarianceM(trainDataV,trainLabelV)#cost N_ATTRxN_ATTRxN_DATA maybe already calculated
    return (muc,Cc) 

def inferClassMVGTied(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]

    NormalPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) pdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalPdfV[i,:]=GAU_ND_pdf(testDataV,muc[:,:,i],Cc)
    
    SJointV=NormalPdfV*vcol(numpy.array(priorProb))
    SMargV=vrow(numpy.sum(SJointV,axis=0))    
    SPostV=SJointV/SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)

    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)

def inferClassMVGLogTied(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]

    NormalLogPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) logpdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalLogPdfV[i,:]=GAU_ND_logpdf(testDataV,muc[:,:,i],Cc)
    
    SLogJointV=NormalLogPdfV+numpy.log(vcol(numpy.array(priorProb)))
    SLogMargV=scipy.special.logsumexp(SLogJointV,axis=0)
    SLogPostV=SLogJointV-SLogMargV
    predictedLabelV=numpy.argmax(SLogPostV,axis=0)
    
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SLogPostV,predictedLabelV,acc)
