import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from MVG.pdf import *

def GMM_logpdf(X,gmm):#gmm=[(w1, mu1, C1), (w2, mu2, C2), ...]
    
    N=X.shape[1]
    M=gmm.__len__()
    S=numpy.zeros((M,N))#S shape is N_RECORDxN_COMPONENTS and those are the gamma

    for g in range(M):
        S[g,:] = GAU_ND_logpdf(X,gmm[g][1],gmm[g][2])
        S[g,:] += numpy.log(gmm[g][0])

    return (S,scipy.special.logsumexp(S, axis=0))#log density


def GMM_logpdf_diagonal(X,gmm):#gmm=[(w1, mu1, C1), (w2, mu2, C2), ...]
    
    N=X.shape[1]
    M=gmm.__len__()
    
    S=numpy.zeros((M,N))#S shape is N_RECORDxN_COMPONENTS and those are the gamma

    for g in range(M):
        S[g,:] = GAU_ND_logpdf_naiveBayes(X,gmm[g][1],gmm[g][2])
        S[g,:] += numpy.log(gmm[g][0])

    return (S,scipy.special.logsumexp(S, axis=0))#log density