import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def logregSolve(trainDataV,trainLabelV,l,p0=PC[0]):#doesn't account for different priors since on training data (whose are balanced) has't given good results
    N_ATTRS=trainDataV.shape[0]

    logregObj=logregObjWrap(trainDataV,trainLabelV,l,p0=p0)#function
    wbopt = numpy.zeros(N_ATTRS+1)
    (wbopt,_,_)=scipy.optimize.fmin_l_bfgs_b(logregObj,wbopt,approx_grad=True)#optimize paramiters
    
    return (wbopt[0:-1], wbopt[-1])# return w,b

def logregObjWrap(trainDataV,trainLabelV,l,p0=PC[0]):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    
    zV=-(2*trainLabelV-1)#already done *-1 for faster computation
    p1=1-p0
    n0 = trainDataV[:,trainLabelV==0].shape[1]
    n1 = trainDataV[:,trainLabelV==1].shape[1]

    def logregObj(wb):
        w, b = wb[0:-1], wb[-1]#unpacking from 1-d array
        logpostV=w.T@trainDataV + b
        distV = numpy.logaddexp(0,zV*logpostV) #oriented distance
        Jwb = l/2 * (numpy.linalg.norm(w) ** 2) + p0 / n0 * distV[trainLabelV == 0].sum() + p1 / n1 * distV[trainLabelV == 1].sum()

        return Jwb

    return logregObj


def inferClassLogreg(testDataV,testLabelV,w,b):
    N_RECORDS=testDataV.shape[1]
    #calculate score
    scoreV=vrow(w)@testDataV+b
    #infer class
    predictedLabel=numpy.where(scoreV>0,1,0)
    
    #calculate accuracy
    A=predictedLabel==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabel,acc)
