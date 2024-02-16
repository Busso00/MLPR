import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def QlogregSolve(trainDataV,trainLabelV,l,p0=PC[0]):#account for different priors
    N_ATTRS=trainDataV.shape[0]
    N_RECORDS=trainDataV.shape[1]
    
    expandedTrainDataV=numpy.zeros((N_ATTRS**2+N_ATTRS,N_RECORDS))
    expandedTrainDataV[-N_ATTRS:,:]+=trainDataV
    for i in range(N_RECORDS):
        expandedTrainDataV[0:-N_ATTRS,i]+=numpy.outer(trainDataV[:,i],trainDataV[:,i]).ravel()

    #numpy.outer(a, b)
    #Compute the outer product of two vectors.
    #Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the outer product is:

    #[[a0*b0  a0*b1 ... a0*bN ]
    #[a1*b0    .
    #[ ...          .
    #[aM*b0            aM*bN ]]

    #update
    logregObj=priorLogregObjWrap(expandedTrainDataV,trainLabelV,l,p0)#closure with expanded feature vector
    Awbopt = numpy.zeros((N_ATTRS**2+N_ATTRS+1,))
    (Awbopt,_,_)=scipy.optimize.fmin_l_bfgs_b(logregObj,Awbopt,approx_grad=True)#optimize paramiters, returns Awb,JAwbin 
    

    return (Awbopt[0:-N_ATTRS-1].reshape((N_ATTRS,N_ATTRS)),Awbopt[-N_ATTRS-1:-1], Awbopt[-1])# return w,b

def priorLogregObjWrap(trainDataV,trainLabelV,l,p0=PC[0]):#useful closure for defining at runtime parameters that we don't vary in order to maximize

    zV=-(2*trainLabelV-1)#already done *-1 for faster computation
    p1=1-p0
    n0 = trainDataV[:,trainLabelV==0].shape[1]
    n1 = trainDataV[:,trainLabelV==1].shape[1]
    
    def logregObj(wb):
        w, b = wb[0:-1], wb[-1]#unpacking from 1-d array
        s = w.T@trainDataV + b#not oriented distance in 1-d remapped space
        distV = numpy.logaddexp(0,zV*s) #oriented distance
        JAwb = l/2 * (numpy.linalg.norm(w) ** 2) + p0 / n0 * distV[trainLabelV == 0].sum() + p1 / n1 * distV[trainLabelV == 1].sum()

        return JAwb
    
    def logregGradient(wb): #overflow
        w, b = wb[0:-1], wb[-1]  # unpacking from 1-d array
        s = w.T @ trainDataV + b  # not oriented distance in 1-d remapped space
        p = 1 / (1 + numpy.exp(zV*s)) # 1 x n_samples
        gradW = l * w 
        #dot product X @ Y = sum over i Xrow(:)col(i) * Yrow(i) 
        #                                                        (n_attr,n_samples) @ (n_samples,1) -> (n_attr,1)
        gradW += p0 / n0 * (zV[trainLabelV == 0] * trainDataV[:, trainLabelV == 0] ) @ (1 - p[trainLabelV == 0]).T
        gradW += p1 / n1 * (zV[trainLabelV == 1] * trainDataV[:, trainLabelV == 1] ) @ (1 - p[trainLabelV == 1]).T
        gradB = p0 / n0 * numpy.sum(zV[trainLabelV == 0] * (1 - p[trainLabelV == 0])) + p1 / n1 * numpy.sum(zV[trainLabelV == 1] * (1 - p[trainLabelV == 1]))
        gradWB = numpy.concatenate((gradW, [gradB])).ravel()
        #concatenate same ndim vectors along axis 0 
        return gradWB

    return logregObj



def inferClassQlogreg(testDataV,testLabelV,A,w,b):#use for calibration
    N_RECORDS=testDataV.shape[1]

    #calculate score
    scoreV = numpy.zeros((1,N_RECORDS))
    for i in range (N_RECORDS):
        scoreV[0,i]=(vrow(testDataV[:,i])@A)@vcol(testDataV[:,i])+vrow(w)@testDataV[:,i]+b
    #infer class
    predictedLabelV=numpy.where(scoreV>0,1,0)
    
    #calculate accuracy
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)
