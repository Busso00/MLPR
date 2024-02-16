import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def SVMPrimalSolve(trainDataV,trainLabelV,K,C,p0=PC[0]):
    #C, K are hyperparameters of the model
    N_ATTRS=trainDataV.shape[0]

    SVMobj=SVMPrimalObjWrap(trainDataV,trainLabelV,K,C,p0)#function
    wbopt=numpy.zeros((N_ATTRS+1,))#params (to modify)
    (wbopt,Jwbmin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj,wbopt,approx_grad=True,factr=1.0)#optimize paramiters
    
    return (wbopt[0:-1], wbopt[-1])# return w,b

def SVMPrimalObjWrap(trainDataV,trainLabelV,K,C,p0=PC[0]):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    #C, K are hyperparameters of the model
    N_ATTRS=trainDataV.shape[0]
    N_RECORDS=trainDataV.shape[1]

    p1=1-p0
    expandedDataV=numpy.zeros((N_ATTRS+1,N_RECORDS))
    expandedDataV[0:-1,:]=trainDataV
    expandedDataV[-1,:]=numpy.ones((N_RECORDS,))*K
    zV=2*trainLabelV-1
    #no use of H
    
    def SVMPrimalObj(wb):
        regterm=(numpy.linalg.norm(wb)**2)/2#is actually objective function if class are perfectly separable (I assume they are not)
        hingelossV=numpy.maximum(numpy.zeros(expandedDataV.shape[1]),(1-zV*(vrow(wb)@expandedDataV)).ravel())

        return regterm+C*(p0/PC[0]*numpy.sum(hingelossV[trainLabelV==0])+p1/PC[1]*numpy.sum(hingelossV[trainLabelV==1]))
    
    return SVMPrimalObj

def inferClassLinearSVM(w,b,testDataV,testLabelV,K):
    N_RECORDS=testDataV.shape[1]

    scoreV=vrow(w)@testDataV+b*K
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)