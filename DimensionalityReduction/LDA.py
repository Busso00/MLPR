import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def LDA(dataV,labelV):#supervised
    m=N_LABELS-1#other directions are random
    Sw=withinClassCovarianceM(dataV,labelV)
    Sb=betweenClassCovarianceM(dataV,labelV)
    _,U=scipy.linalg.eigh(Sb,Sw)#solve the generalized eigenvalue problem
    #eigenvalue Sw^-1@Sb , only C-1 eigenvectors are !=0 -> useless random eigenvectors
    W=U[:,::-1][:,0:m]
    projectedDataV=W.T@dataV

    return (projectedDataV,W)

def LDA2proj(dataV,labelV):#supervised
    m=N_LABELS-1#other directions are random
    Sw=withinClassCovarianceM(dataV,labelV)
    Sb=betweenClassCovarianceM(dataV,labelV)
    
    U,eigv1,_=numpy.linalg.svd(Sw)
    P1=U@numpy.diag(1.0/(eigv1**0.5))@U.T
    SBT=P1@Sb@P1.T#transformed between class covariance
    P2,_,_=numpy.linalg.svd(SBT)
    #eigenvalue Sw^-1@Sb , only C-1 eigenvectors are !=0 -> useless random eigenvectors
    W=P1.T@P2[:,0:m]
    projectedDataV=W.T@dataV

    return (projectedDataV,W)

def testLDA(data,labels):#keep at maximum 1 dimension
    
    projectedDataV,P_not_T=LDA(data,labels)
    print("LDA Transform matrix:")
    print(P_not_T)
    print("\nvisualize dataV after LDA:")
    visualizeData(projectedDataV,labels)
    
    return projectedDataV,P_not_T

