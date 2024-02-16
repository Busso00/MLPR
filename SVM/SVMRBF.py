import sys
sys.path.append('..')
from const import *
from Utility.utility import *


def expKernelSVMSolve(trainDataV,trainLabelV,C,eps=0.0,gamma=0.0,p0=PC[0]):
    #C, eps, const are hyperparameters of the model
    N_RECORDS=trainDataV.shape[1]
    p1=1-p0
    SVMobj=expKernelSVMWrap(trainDataV,trainLabelV,eps=eps,gamma=gamma)#function
    alfaVopt=numpy.zeros((N_RECORDS,))#params (to modify)
    alfaLimits=[]
    rebalanceTermV=numpy.where(trainLabelV==0,p0/PC[0],p1/PC[1])

    for i in range(N_RECORDS):
        alfaLimits.append((0,C*rebalanceTermV[i]))
    (alfaVopt,Jalfamin,_)=scipy.optimize.fmin_l_bfgs_b(SVMobj[0],alfaVopt,fprime=SVMobj[1],bounds=alfaLimits,factr=1.0)#optimize paramiters
    
    return (alfaVopt,Jalfamin)# return alfaV,loss (dual) , also JAlfamin


def wKfunRBF(gamma,eps):

    def kFunRBF(dataV1,dataV2):
        dist= vcol((dataV1**2).sum(0))+vrow((dataV2**2).sum(0))-2*dataV1.T@dataV2
        return numpy.exp(-gamma*dist)+eps
    return kFunRBF

def expKernelSVMWrap(trainDataV,trainLabelV,eps=0.0,gamma=1.0):#useful closure for defining at runtime parameters that we don't vary in order to maximize
    #N_RECORDS=trainDataV.shape[1]
    
    #kernelM=numpy.zeros((N_RECORDS,N_RECORDS))
    #for row in range(N_RECORDS):
        #modify kernel
    #    kernelM[row,:]=numpy.exp(-gamma*numpy.linalg.norm(trainDataV-vcol(trainDataV[:,row]),axis=0)**2)+eps

    kFun = wKfunRBF(gamma,eps)
    kernelM=kFun(trainDataV,trainDataV)
    zV=2*trainLabelV-1
    H=kernelM*vrow(zV)*vcol(zV)

    def SVMObj(alfaV):
        
        return vrow(alfaV)@H@vcol(alfaV)/2 - alfaV.sum()
    
    def SVMGradient(alfaV):

        return (H@vcol(alfaV) - 1 ).ravel()
    
    return (SVMObj,SVMGradient)
    

def inferClassExpSVM(trainDataV,trainLabelV,testDataV,testLabelV,alfaV,eps=0.0,gamma=1.0):
    N_RECORDS=testDataV.shape[1]

    zV=2*trainLabelV-1

    #scoreV = numpy.zeros((N_RECORDS,))

    #for t in range(N_RECORDS):
        #modify kernel
    #    kernelVT=numpy.exp(-gamma*numpy.linalg.norm(trainDataV-vcol(testDataV[:,t]),axis=0)**2)+eps
    #    scoreV[t] = (vrow(zV)*vrow(alfaV)*kernelVT).sum(axis=1)
    kFun=wKfunRBF(gamma,eps)
    kernelM=kFun(trainDataV,testDataV)
    scoreV=(vrow(zV)*vrow(alfaV)*kernelM.T).sum(1)
    
    predictedLabelV=numpy.where(scoreV>0,1,0)
    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (scoreV,predictedLabelV,acc)
