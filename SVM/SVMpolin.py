import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def polinKernelSVMSolve(trainDataV,trainLabelV,C,eps=0.0,degree=2,const=0.0,p0=PC[0]):
    #C, eps, const are hyperparameters of the model
    N_RECORDS=trainDataV.shape[1]
    p1=1-p0
    SVMobj=polinKernelSVMWrap(trainDataV,trainLabelV,eps=eps,degree=degree,const=const)#function
    alfaVopt=numpy.zeros((N_RECORDS,))#params (to modify)
    alfaLimits=[]
    rebalanceTermV=numpy.where(trainLabelV==0,p0/PC[0],p1/PC[1])

    for i in range(N_RECORDS):
        alfaLimits.append((0,C*rebalanceTermV[i]))
        

    (alfaVopt,Jalfamin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj[0],alfaVopt,fprime=SVMobj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jalfamin)# return alfaV,loss (dual) , also JAlfamin

def polinKernelSVMWrap(trainDataV,trainLabelV,eps=0,degree=2,const=0):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    #eps, const, degree are hyperparameters of the model

    kernelM=(trainDataV.T@trainDataV + const)**degree + eps
    zV=2*trainLabelV-1
    H=kernelM*vrow(zV)*vcol(zV)
    
    def SVMObj(alfaV):
        
        return (vrow(alfaV)@H)@vcol(alfaV)/2 - alfaV.sum()
    
    def SVMGradient(alfaV):

        return (H@vcol(alfaV) - 1 ).ravel()
    
    return (SVMObj,SVMGradient)

def inferClassPolinSVM(trainDataV,trainLabelV,testDataV,testLabelV,alfaV,eps=0.0,degree=2,const=0.0):
    N_RECORDS=testDataV.shape[1]

    zV=2*trainLabelV-1
    
    scoreV = (vcol(zV)*vcol(alfaV)*((trainDataV.T@testDataV + const)**degree + eps)).sum(axis=0)
    
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)
