import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from MVG.pdf import *

def MVGNaiveBayesTied(trainData,trainLabel):#compute only one (unique for each class) variances
    N_ATTRS=trainData.shape[0]
    N_RECORDS=trainData.shape[1]

    #muc = mean for all classes (must differ)
    muc=numpy.zeros((N_ATTRS,1,N_LABELS))
    for i in range(N_LABELS):
        muc[:,0,i]=numpy.mean(trainData[:,trainLabel==i],axis=1)
    #Cc = covariance matrix (only diagonal, shared across all classes)
    Cc=numpy.zeros((N_ATTRS))
    for i in range(N_LABELS):#cost N_ATTRxN_DATA
        recordsOfC=trainData[:,trainLabel==i]
        N_RECORDS_CLASS=recordsOfC.shape[1]
        Cc+=numpy.var(recordsOfC,axis=1)*N_RECORDS_CLASS/N_RECORDS
    
    return (muc,Cc)


def inferClassMVGNaiveBayesTied(testDataV,testLabelV,muc,Cc,priorProb=PC):#more accurate features of my data are uncorrelated (converge with less training data)
    N_RECORDS = testDataV.shape[1]

    NormalPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) pdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalPdfV[i,:]=GAU_ND_pdf_naiveBayes(testDataV,muc[:,:,i],Cc)
    
    SJointV=NormalPdfV*vcol(numpy.array(priorProb))
    SMargV=vrow(numpy.sum(SJointV,axis=0))    
    SPostV=SJointV/SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)

    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)

def inferClassMVGLogNaiveBayesTied(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]

    NormalLogPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) logpdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalLogPdfV[i,:]=GAU_ND_logpdf_naiveBayes(testDataV,muc[:,:,i],Cc)

    SJointV=NormalLogPdfV+numpy.log(vcol(numpy.array(priorProb)))
    SMargV=scipy.special.logsumexp(SJointV,axis=0)
    SPostV=SJointV-SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)
    
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)
