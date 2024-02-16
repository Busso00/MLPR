import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from MVG.pdf import *

#no weight on prior of dataset (can only result in less accurate mean and covariance estimate,
#but not in a change of decision boundaries witch can be only influenced by the later chosed threshold)
def MVG(trainDataV,trainLabelV):
    N_ATTRS = trainDataV.shape[0]
    #muc = mean for all classes
    muc=numpy.zeros((N_ATTRS,1,N_LABELS))
    for c in range(N_LABELS):
        muc[:,0,c]=numpy.mean(trainDataV[:,trainLabelV==c],axis=1)
    #Cc = covariance matrix for all classes
    Cc=numpy.zeros((N_ATTRS,N_ATTRS,N_LABELS))
    for c in range(N_LABELS):#cost N_ATTRxN_ATTRxN_DATA
        Cc[:,:,c]=numpy.cov(trainDataV[:,trainLabelV==c],bias=True)

    return (muc,Cc)

def inferClassMVG(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]

    NormalPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for c in range(N_LABELS):
        #return a normal (multivariate) pdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalPdfV[c,:]=GAU_ND_pdf(testDataV,muc[:,:,c],Cc[:,:,c])
    
    SJointV=NormalPdfV*vcol(numpy.array(priorProb))/vcol(numpy.array(PC))
    SMargV=vrow(numpy.sum(SJointV,axis=0))    
    SPostV=SJointV/SMargV
    predictedLabelV=numpy.argmax(SPostV,axis=0)

    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SPostV,predictedLabelV,acc)

def inferClassMVGLog(testDataV,testLabelV,muc,Cc,priorProb=PC):
    N_RECORDS = testDataV.shape[1]

    NormalLogPdfV=numpy.zeros((N_LABELS,N_RECORDS))
    for i in range(N_LABELS):
        #return a normal (multivariate) logpdf N(x) where x are test samples (testDataV) given mu and Cov of class c
        NormalLogPdfV[i,:]=GAU_ND_logpdf(testDataV,muc[:,:,i],Cc[:,:,i])
    
    SLogJointV=NormalLogPdfV+numpy.log(vcol(numpy.array(priorProb)))
    SLogMargV=scipy.special.logsumexp(SLogJointV,axis=0)
    SLogPostV=SLogJointV-SLogMargV
    predictedLabelV=numpy.argmax(SLogPostV,axis=0)
    
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SLogPostV,predictedLabelV,acc)