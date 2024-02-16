import sys
sys.path.append('..')
from const import *
from Utility.utility import *
from GMM.pdf import GMM_logpdf

def eStepTied(trainDataV,gau):
    
    (JointLogPdfV,MargLogPdfV)=GMM_logpdf(trainDataV,gau)
    gammaV=numpy.exp(JointLogPdfV-MargLogPdfV)
    return gammaV

def mStepTied(trainDataV,gammaV,psi=0.01):
    N_RECORDS = trainDataV.shape[1]
    N_ATTRS= trainDataV.shape[0]
    N_COMP = gammaV.shape[0]#number of components
 
    
    newgau=[]
    newC=numpy.zeros((N_ATTRS,N_ATTRS))
    for g in range(N_COMP):
        Zstat=gammaV[g,:].sum()
        Fstat=((vrow(gammaV[g,:]))*trainDataV).sum(axis=1)
        Sstat=((vrow(gammaV[g,:])*trainDataV)@trainDataV.T)
        newwg=Zstat/N_RECORDS
        newmug=Fstat/Zstat #final c mean class (component in this case) #N_attrs,g
        newCg=Sstat/Zstat-(vcol(newmug)@vrow(newmug))
        
        newgau.append((newwg,vcol(newmug),newCg))

        newC+=Zstat*newCg

    for g in range(N_COMP):

        U, s, _ = numpy.linalg.svd(newC/N_RECORDS)
        s[s<psi] = psi
        newCg = U@(vcol(s)*U.T)

        newgau[g]=(newgau[g][0],newgau[g][1],newCg)

    return newgau

def GMMTied(trainDataV,gmm,minIncr=10**-6,psi=0.01):
    N_SAMPLES = trainDataV.shape[1]
    start=True
    newgmm=gmm
    llpast=0.0
    llact=0.0
    
    while (start or (((llact-llpast)/N_SAMPLES)>minIncr)):
        gmm=newgmm
        gammaV=eStepTied(trainDataV,gmm)
        newgmm=mStepTied(trainDataV,gammaV,psi=psi)
        if start:
            start=False
            llpast=GMM_logpdf(trainDataV,gmm)[1].sum()
            llact=GMM_logpdf(trainDataV,newgmm)[1].sum()
        else:
            llpast=llact
            llact=GMM_logpdf(trainDataV,newgmm)[1].sum()
    
    return newgmm

def LBGTied(trainDataV,alpha=0.1,nClusters=2,psi=0.01):#nClusters must be the log2 of the power of 2 nClusters + 1
    N_ATTRS= trainDataV.shape[0]
    mu=vcol(trainDataV.mean(axis=1))
    C=numpy.cov(trainDataV,bias=True).reshape((N_ATTRS,N_ATTRS))#overcome 1 dimensional data
    gmm = [(1.0, mu, C)]
    gmmV = [gmm]
    newgmm=False
    for _ in range(int(numpy.log2(nClusters)+1.0)): 
        if newgmm:
            gmm=newgmm
            gmm=GMMTied(trainDataV,gmm,psi=psi)
            gmmV.append(gmm)
        
        newgmm=[]
        for g in range(gmm.__len__()):

            U, s, _ = numpy.linalg.svd(gmm[g][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha

            newgmm.append((gmm[g][0]/2,gmm[g][1]-d,gmm[g][2]))
            newgmm.append((gmm[g][0]/2,gmm[g][1]+d,gmm[g][2]))
    
    return gmmV


def allClassLBGTied(trainDataV,trainLabelV,alpha=0.1,nComp=2,psi=0.01):
    gmmcV=[]
    for i in  range(N_LABELS):
        gmmV=LBGTied(trainDataV[:,trainLabelV==i],nClusters=nComp,alpha=alpha,psi=psi)
        gmmcV.append(gmmV)
    
    gmmc=[]#now i have it divided by class
    for j in range (int(numpy.log2(nComp))+1):
        gmmcTemp=[]
        for i in range(N_LABELS):
            gmmcTemp.append(gmmcV[i][j])
        gmmc.append(gmmcTemp)
            
    return gmmc
        

def inferClassGMMTied(testDataV,testLabelV,gmmc,priorProb=PC):
    N_RECORDS=testDataV.shape[1]

    SLogGMMV=numpy.zeros((N_LABELS,N_RECORDS))
    for c in range(N_LABELS):
        SLogGMMV[c,:]=GMM_logpdf(testDataV,gmmc[c])[1] #doesn't change to use exp
    
    SLogJointV=SLogGMMV+numpy.log(vcol(numpy.array(priorProb)))
    SLogMargV=scipy.special.logsumexp(SLogJointV,axis=0)
    SLogPostV=SLogJointV-SLogMargV
    predictedLabelV=numpy.argmax(SLogPostV,axis=0)

    A=predictedLabelV==testLabelV
    acc=A.sum()/float(N_RECORDS)

    return (SLogPostV,predictedLabelV,acc)