import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def PCA(dataV,m_req_dim):#unsupervised
    covarianceM=numpy.cov(dataV,bias=True)#already center the data, but normalize by n-1
    _,eigenvectors=numpy.linalg.eigh(covarianceM)#eigenvalues aren't necessary
    P = eigenvectors[:, ::-1][:,0:m_req_dim] 
    projectedData=P.T@dataV
    
    return (projectedData,P)

def PCATreshold(dataV,treshold):#unsupervised
    covarianceM=numpy.cov(dataV,bias=True)#already center the data, but normalize by n-1
    eigenvalues,eigenvectors=numpy.linalg.eigh(covarianceM)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    required_dim=1
    explained_var=eigenvalues[0]
    tot_explained_var=eigenvalues.sum()
    while (explained_var/tot_explained_var)<treshold:
        explained_var+=eigenvalues[required_dim]
        required_dim+=1

    print("Selected dimensions : %d"%required_dim)
    P=eigenvectors[:,0:required_dim]
    projectedData=P.T@dataV
    
    return (projectedData,P)

def PCAsvd(data,m_req_dim):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    U,_,_=numpy.linalg.svd(covarianceM)#singularvalues aren't necessary
    P = U[:,0:m_req_dim]
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def PCATresholdsvd(data,treshold):#unsupervised
    covarianceM=numpy.cov(data,bias=True)#already center the data, but normalize by n-1
    U,singularvalues,_=numpy.linalg.svd(covarianceM)

    required_dim=1
    explained_var=singularvalues[0]
    tot_explained_var=singularvalues.sum()
    while (explained_var/tot_explained_var)<treshold:
        explained_var+=singularvalues[required_dim]
        required_dim+=1

    print("Selected dimensions : %d"%required_dim)
    P=U[:,0:required_dim]
    projectedData=P.T@data
    print("Transform matrix:")
    print(P)
    return (projectedData,P)

def testPCA(data,labels,nDim,visualize=False):#the version of PCA selected is the one with threshold: keeps 95% information -> 6 dimensions

    (projectedData,P_not_T)=PCA(data,nDim)
    if (visualize):
        print("PCA Transform matrix:")
        print(P_not_T)
        print("\nvisualize data after PCA:")
        visualizeData(projectedData,labels)
        
    return projectedData,P_not_T
