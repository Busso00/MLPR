import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def GAU_ND_pdf(X,mu,C):#compute only one 
    XC=X-mu
    M=X.shape[0]
    const=(numpy.pi*2)**(-0.5*M)
    det=numpy.linalg.det(C)
    L=numpy.linalg.inv(C)
    #for i in range(data.shape[1]):
    #    Pdata[i]=-1/2*XC[:,i].T@L@XC[:,i]
    #efficient way
    v=(XC*(L@XC)).sum(axis=0)
    return const*(det**-0.5)*numpy.exp(-0.5*v)

def GAU_ND_logpdf(X,mu,C):
    XC=X-mu
    M=X.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.linalg.slogdet(C)[1]
    L=numpy.linalg.inv(C)
    #for i in range(data.shape[1]):
    #    Pdata[i]=-1/2*XC[:,i].T@L@XC[:,i]
    #efficient way
    v=(XC*(L@XC)).sum(axis=0)
    return const-0.5*logdet-0.5*v

def GAU_ND_pdf_naiveBayes(X,mu,C):#C is diagonal (diagonal vector)-> less computational expensive
    M=X.shape[0]
    const=(numpy.pi*2)**(-0.5*M)
    det=C.prod(axis=0)*((-1)**M)
    XC=X-mu
    L=1/C
    v=numpy.zeros(X.shape[1])
    #for i in range(data.shape[1]):#better way than explicit iteration?
    #    for j in range(data.shape[0]):#cost N_ATTRxN_DATA
    #        v[i]+=XC[j,i]*L[j]*XC[j,i]
    #efficient way
    v=(XC**2*vcol(L)).sum(axis=0)
    return const*(det**-0.5)*numpy.exp(-0.5*v)

def GAU_ND_logpdf_naiveBayes(X,mu,C):#C is diagonal (diagonal vector)-> less computational expensive
    M=X.shape[0]
    const=-0.5*M*numpy.log(2*numpy.pi)
    logdet=numpy.log(C).sum(axis=0)
    XC=X-mu
    L=1/C
    v=numpy.zeros(X.shape[1])
    #for i in range(data.shape[1]):
    #    for j in range(data.shape[0]):
    #        v[i]+=(XC[j,i]*L[j]*XC[j,i])
    #efficient way
    v=(XC**2*vcol(L)).sum(axis=0)
    return const-0.5*logdet-0.5*v
