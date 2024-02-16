##Progetto Biometric Identity Verification per il corso Machine Learning and Pattern Recognition di Federico Bussolino e Francine Ombala

#initial consideration: from corcoefficient matrix seen high correlation bw attr:
#lot of attributes are higly correlated
import sys
sys.path.append('')
from const import *
from Utility.utility import *
from DimensionalityReduction.PCA import testPCA
from DimensionalityReduction.LDA import testLDA
from MVG.MVG import *
from MVG.MVG_NaiveBayes import *
from MVG.MVG_tied import *
from MVG.MVG_NaiveBayes_tied import *
from LR.linear import logregSolve,inferClassLogreg
from LR.expanded import QlogregSolve, inferClassQlogreg
from Evaluation.eval import *
from SVM.SVMlinear import SVMPrimalSolve, inferClassLinearSVM
from SVM.SVMpolin import polinKernelSVMSolve, inferClassPolinSVM
from SVM.SVMRBF import  expKernelSVMSolve, inferClassExpSVM
from GMM.GMM import *
from GMM.GMM_Diag import *
from GMM.GMM_tied import *
from GMM.GMM_Diag_tied import *
from GMM.GMM_Mixed import *

def KFold(D,L,k,type=0,l=0.0,K=1.0,C=1.0,degree=2,gamma=1.0,nComp=2,roc=False,be=False,color=0,maxShowedCost=1.0,prior=0.1,det=False,seed=0):#TODO: add another step to KFoldCal for calibration of scores (training on all scores)
    ## D = dataV, L = labelV
    ## seed = RNG seed for random shuffling the data
    ## type: 0 = MVG, 1 = Naive-Bayes, 2 = tied, 3 = naive-Bayes tied
    ## next four types are log versions of same model (less numerical issues -> used)
    ## type: 8 = Logistic Regression --> l = lambda of Logistic Regression

    nFold=int(D.shape[1]/k)#n elements in a fold
    numpy.random.seed(seed)
    acc=0.0
    idx=numpy.random.permutation(D.shape[1])
    scoreV=numpy.zeros((len(nToLabel),D.shape[1]))
    testLabelV=numpy.zeros((D.shape[1],))#all actual labels
    predLabelV=numpy.zeros((D.shape[1],))#all predicted labels
    gmmcV=[]
    costV=[]
    costV1=[]
    costV2=[]

    for i in range(k):

        idxTrain=numpy.zeros(D.shape[1]-nFold,dtype=numpy.int32)
        idxTrain[:i*nFold]=idx[:i*nFold]
        idxTrain[i*nFold:]=idx[(i+1)*nFold:]
        idxTest=idx[i*nFold:(i+1)*nFold]
        trainData=D[:,idxTrain]
        testData=D[:,idxTest]
        trainLabel=L[idxTrain]
        testLabel=L[idxTest]
        
        match type:
            case 0:#not used
                (muc,Cc)=MVG(trainData,trainLabel)#Cc.shape=(n_attr,n_attr,n_class)
                (partialScoreV,predLabel,partialAcc)=inferClassMVG(testData,testLabel,muc,Cc)
            case 1:#not used
                (muc,Cc)=MVGNaiveBayes(trainData,trainLabel)#Cc.shape=(n_attr,n_class)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGNaiveBayes(testData,testLabel,muc,Cc)
            case 2:#not used
                (muc,Cc)=MVGTied(trainData,trainLabel)#Cc.shape=(n_attr,n_attr)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGTied(testData,testLabel,muc,Cc)
            case 3:#not used
                (muc,Cc)=MVGNaiveBayesTied(trainData,trainLabel)#Cc.shape=(n_attr)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGNaiveBayesTied(testData,testLabel,muc,Cc)
            case 4:
                (muc,Cc)=MVG(trainData,trainLabel)#Cc.shape=(n_attr,n_attr,n_class)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGLog(testData,testLabel,muc,Cc)
            case 5:
                (muc,Cc)=MVGNaiveBayes(trainData,trainLabel)#Cc.shape=(n_attr,n_class)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGLogNaiveBayes(testData,testLabel,muc,Cc)
            case 6:
                (muc,Cc)=MVGTied(trainData,trainLabel)#Cc.shape=(n_attr,n_attr)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGLogTied(testData,testLabel,muc,Cc)
            case 7:
                (muc,Cc)=MVGNaiveBayesTied(trainData,trainLabel)#Cc.shape=(n_attr)
                (partialScoreV,predLabel,partialAcc)=inferClassMVGLogNaiveBayesTied(testData,testLabel,muc,Cc)
            case 8:
                if not FUSION:
                    (w,b)=logregSolve(trainData,trainLabel,l)
                    (partialScoreV,predLabel,partialAcc)=inferClassLogreg(testData,testLabel,w,b)
                else:
                    (w,b)=logregSolve(trainData,trainLabel,l,p0=2000/3500)
                    (partialScoreV,predLabel,partialAcc)=inferClassLogreg(testData,testLabel,w,b)
            case 9:
                (A,w,b)=QlogregSolve(trainData,trainLabel,l)
                (partialScoreV,predLabel,partialAcc)=inferClassQlogreg(testData,testLabel,A,w,b)
            case 10:
                (w,b)=SVMPrimalSolve(trainData,trainLabel,K,C)
                (partialScoreV,predLabel,partialAcc)=inferClassLinearSVM(w,b,testData,testLabel,K)
            case 11:
                (alfaV,_)=polinKernelSVMSolve(trainData,trainLabel,C,degree=degree)
                (partialScoreV,predLabel,partialAcc)=inferClassPolinSVM(trainData,trainLabel,testData,testLabel,alfaV,degree=degree)
            case 12:
                (alfaV,_)=expKernelSVMSolve(trainData,trainLabel,C,gamma=gamma)
                (partialScoreV,predLabel,partialAcc)=inferClassExpSVM(trainData,trainLabel,testData,testLabel,alfaV,gamma=gamma)
            case 13:
                gmmcV.append(allClassLBG(trainData,trainLabel,nComp=nComp))
                (partialScoreV,predLabel,partialAcc)=inferClassGMM(testData,testLabel,gmmcV[i][1])
            case 14:
                gmmcV.append(allClassLBGDiagonal(trainData,trainLabel,nComp=nComp))
                (partialScoreV,predLabel,partialAcc)=inferClassGMMDiagonal(testData,testLabel,gmmcV[i][1])
            case 15:
                gmmcV.append(allClassLBGTied(trainData,trainLabel,nComp=nComp))
                (partialScoreV,predLabel,partialAcc)=inferClassGMMTied(testData,testLabel,gmmcV[i][1])
            case 16:
                gmmcV.append(allClassLBGDiagonalTied(trainData,trainLabel,nComp=nComp))
                (partialScoreV,predLabel,partialAcc)=inferClassGMMDiagonalTied(testData,testLabel,gmmcV[i][1])
            case 17:
                gmmcV=allClassLBGMixed(trainData,trainLabel,nComp=nComp)
                (partialScoreV,predLabel,partialAcc)=inferClassGMMMixed(testData,testLabel,gmmcV[1])
            case 18:
                gmmcV=allClassLBGMixed2(trainData,trainLabel,nComp=nComp)
                (partialScoreV,predLabel,partialAcc)=inferClassGMMMixed2(testData,testLabel,[gmmcV[0][0],gmmcV[1][1]])
            
        acc+=partialAcc
        
        #make a score vector for each fold
        scoreV[:,i*nFold:(i+1)*nFold]+=partialScoreV
        testLabelV[i*nFold:(i+1)*nFold]+=testLabel
        predLabelV[i*nFold:(i+1)*nFold]+=predLabel.ravel()

    acc/=float(k)
    print("total accuracy:%f"%acc)
    
    llrV = []

    #just to uniform the score between the returned by different models
    if(type >= 0 and type < 8):
        llrV = binScoreMVG(scoreV)
        
    if(type >=8 and type < 10):
        llrV = binScoreLR(scoreV)
        
    if(type >=10 and type <13):
        llrV = binScoreSVM(scoreV)#actually is not well calibrated log-likehood ratio since there is no probabilistic interpretation

    if(type >= 13):
        llrV = binScoreMVG(scoreV)  
    
    #predLabelV = binaryTaskEvaluation(llrV,0.1,1,1)

    #confM = confusionMatrix(testLabelV,predLabelV)
    (cost,_,_,_) = incrMinDCF(llrV,testLabelV,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)
    (cost1,_,_,_) = incrMinDCF(llrV,testLabelV,0.5,1,1)
    print("cost incrMinDCF @ 0.5:%f" % cost1)
    (cost2,_,_,_) = incrMinDCF(llrV,testLabelV,0.05,1,1)
    print("cost incrMinDCF @ 0.05:%f" % cost2)

    if roc :
        calcROC(llrV,testLabelV,0.5,1,1,color=color,modelN=type)
       
    if be:
        #since i added it to model in gaussian (and also LR) i may have to subtract log(1500/2000)
        #but since the score is already a ratio the division already subtract this quantity
        threshold=-numpy.log(0.1/0.9) 
        predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
        confM = confusionMatrix(testLabelV,predLabelV)

        print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))

        threshold=-numpy.log(0.5/0.5)
        predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
        confM = confusionMatrix(testLabelV,predLabelV)
        
        print("cost actDCF @ 0.5:%f" % normalizedDCF(confM,0.5,1,1))
        
        threshold=-numpy.log(0.05/0.95)
        predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
        confM = confusionMatrix(testLabelV,predLabelV)

        print("cost actDCF @ 0.05:%f" % normalizedDCF(confM,0.05,1,1))

        calcBayesError(llrV,testLabelV,color=color,maxShowedCost=maxShowedCost,modelN=type)

    if det:
        calcDET(llrV,testLabelV,0.5,1,1,color=color,modelN=type)
    if(type >= 13): #try to 8 components only for diagoal since is cheaper
        #gmm n components
        costV.append(cost)
        costV1.append(cost1)
        costV2.append(cost2)

        for n in range(2,int(numpy.log2(nComp))+1):
            
            nFold=int(D.shape[1]/k)#n elements in a fold
            acc=0.0
            scoreV=numpy.zeros((len(nToLabel),D.shape[1]))
            testLabelV=numpy.zeros((D.shape[1],))#all actual labels
            predLabelV=numpy.zeros((D.shape[1],))#all predicted labels

            for i in range(k):

                
                idxTest=idx[i*nFold:(i+1)*nFold]
                testData=D[:,idxTest]
                testLabel=L[idxTest]

                match type:
                    case 13:
                        (partialScoreV,predLabel,partialAcc)=inferClassGMM(testData,testLabel,gmmcV[i][n])
                    case 14:
                        (partialScoreV,predLabel,partialAcc)=inferClassGMMDiagonal(testData,testLabel,gmmcV[i][n])
                    case 15:
                        (partialScoreV,predLabel,partialAcc)=inferClassGMMTied(testData,testLabel,gmmcV[i][n])
                    case 16:
                        (partialScoreV,predLabel,partialAcc)=inferClassGMMDiagonalTied(testData,testLabel,gmmcV[i][n])
                
                acc+=partialAcc
            
                #make a score vector for each fold
                scoreV[:,i*nFold:(i+1)*nFold]+=partialScoreV
                testLabelV[i*nFold:(i+1)*nFold]+=testLabel
                predLabelV[i*nFold:(i+1)*nFold]+=predLabel.ravel()
                
            acc/=float(k)
            print("total accuracy:%f"%acc)
            
            llrV = binScoreMVG(scoreV)  

            (cost,_,_,_) = incrMinDCF(llrV,testLabelV,0.1,1,1)
            print("cost incrMinDCF %d @ WP:%f" % (2**n,cost))
            (cost1,_,_,_) = incrMinDCF(llrV,testLabelV,0.5,1,1)
            print("cost incrMinDCF %d @ 0.5:%f" % (2**n,cost1))
            (cost2,_,_,_) = incrMinDCF(llrV,testLabelV,0.05,1,1)
            print("cost incrMinDCF %d @ 0.05:%f" % (2**n,cost2))

            costV.append(cost)
            costV1.append(cost1)
            costV2.append(cost2)

            if roc :
                calcROC(llrV,testLabelV,0.5,1,1,color=color,modelN=type)
            
            if be:
                calcBayesError(llrV,testLabelV,color=color,maxShowedCost=maxShowedCost,modelN=type)
            
            
        if FUSION:
            return (llrV,testLabelV)
        
        return (costV,costV1,costV2)
        
    if FUSION:
        return (llrV,testLabelV)
    
    return (cost,cost1,cost2)


def split_db_2tol(D,L,seed=0):

    nTrain=int(D.shape[1]*4.0/5.0)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    idxTrain=idx[:nTrain]
    idxTest=idx[nTrain:]

    trainData=D[:,idxTrain]
    testData=D[:,idxTest]
    trainLabel=L[idxTrain]
    testLabel=L[idxTest]

    return (trainData,trainLabel),(testData,testLabel)

def trainModel(trainData, trainLabel):
    
    PDTR,P=testPCA(trainData,trainLabel,10)
    ((DTR,LTR),(DTE,LTE))=split_db_2tol(PDTR,trainLabel)
    (mvgmuC,mvgCC)=MVGNaiveBayes(DTR[0:8,:],LTR)
    (score1,_,_)=inferClassMVGLogNaiveBayes(DTE[0:8,:],LTE,mvgmuC,mvgCC,priorProb=[0.9,0.1])
    llrV1 = binScoreMVG(score1)
    gmmC=allClassLBGTied(DTR[0:7,:],LTR,nComp=2)[1]
    (score2,_,_)=inferClassGMMTied(DTE[0:7,:],LTE,gmmC,priorProb=[0.9,0.1])
    llrV2 = binScoreMVG(score2)
    (w,b)=logregSolve(numpy.vstack((llrV1.ravel(),llrV2.ravel())),LTE,10**-2,p0=0.9)

    return P,mvgmuC,mvgCC,gmmC,w,b

def applyModel(testData,LTE,P,mvgmuC,mvgCC,gmmC,w,b,be=False):
    
    PDTE=P.T@testData
    (score1,_,_)=inferClassMVGLogNaiveBayes(PDTE[0:8,:],LTE,mvgmuC,mvgCC,priorProb=[0.9,0.1])
    llrV1 = binScoreMVG(score1)
    (score2,_,_)=inferClassGMMTied(PDTE[0:7,:],LTE,gmmC,priorProb=[0.9,0.1])
    llrV2 = binScoreMVG(score2)
    (score,_,acc)=inferClassLogreg(numpy.vstack((llrV1.ravel(),llrV2.ravel())),LTE,w,b)

    llrV = binScoreLR(score)
    threshold=0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(LTE,predLabelV)
    
    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

    if be:
        calcBayesError(llrV-numpy.log(0.1/0.9),LTE,color=colorPalette[0],maxShowedCost=0.5,modelN='fusion 1(selected)')

    

def trainModel2(trainData, LTR):
    
    PDTR,P=testPCA(trainData,LTR,10)
    
    gmmC=allClassLBGTied(PDTR[0:7,:],LTR,nComp=2)[1]
    
    return P,gmmC

def applyModel2(testData,LTE,P,gmmC):
    
    DTE=P.T@testData
    (score2,_,acc)=inferClassGMMTied(DTE[0:7,:],LTE,gmmC,priorProb=[0.9,0.1])
    llrV = binScoreMVG(score2)
    
    threshold=0.0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(LTE,predLabelV)

    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

def trainModel3(trainData, trainLabel):
    
    PDTR,P=testPCA(trainData,trainLabel,10)
    
    (mvgmuC,mvgCC)=MVGNaiveBayes(PDTR[0:8,:],trainLabel)
    
    return P,mvgmuC,mvgCC

def applyModel3(testData,LTE,P,mvgmuC,mvgCC):
    
    DTE=P.T@testData
    (score1,_,acc)=inferClassMVGLogNaiveBayes(DTE[0:8,:],LTE,mvgmuC,mvgCC,priorProb=[0.9,0.1])
    llrV = binScoreMVG(score1)
    
    threshold=0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(LTE,predLabelV)

    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

def trainModel4(trainData, trainLabel):
    
    PDTR,P=testPCA(trainData,trainLabel,10)
    ((DTR,LTR),(DTE,LTE))=split_db_2tol(PDTR,trainLabel)
    (mvgmuC,mvgCC)=MVGNaiveBayes(DTR[0:8,:],LTR)
    (score1,_,_)=inferClassMVGLogNaiveBayes(DTE[0:8,:],LTE,mvgmuC,mvgCC,priorProb=[0.9,0.1])
    llrV1 = binScoreMVG(score1)
    gmmC=allClassLBGDiagonal(DTR[0:6,:],LTR,nComp=4)[2]
    (score2,_,_)=inferClassGMMDiagonal(DTE[0:6,:],LTE,gmmC,priorProb=[0.9,0.1])
    llrV2 = binScoreMVG(score2)
    (w,b)=logregSolve(numpy.vstack((llrV1.ravel(),llrV2.ravel())),LTE,5*10**-2,p0=0.9)

    return P,mvgmuC,mvgCC,gmmC,w,b

def applyModel4(testData,LTE,P,mvgmuC,mvgCC,gmmC,w,b,be=False):
    
    PDTE=P.T@testData
    (score1,_,_)=inferClassMVGLogNaiveBayes(PDTE[0:8,:],LTE,mvgmuC,mvgCC,priorProb=[0.9,0.1])
    llrV1 = binScoreMVG(score1)
    (score2,_,_)=inferClassGMMDiagonal(PDTE[0:6,:],LTE,gmmC,priorProb=[0.9,0.1])
    llrV2 = binScoreMVG(score2)
    (score,_,acc)=inferClassLogreg(numpy.vstack((llrV1.ravel(),llrV2.ravel())),LTE,w,b)

    llrV = binScoreLR(score)
    threshold=0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(LTE,predLabelV)

    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

    if be:
        calcBayesError(llrV-numpy.log(0.1/0.9),LTE,color=colorPalette[3],maxShowedCost=0.5,modelN='fusion 2')


def trainModel5(trainData, trainLabel,gamma=10**-4,C=1000):
    PDTR,P=testPCA(trainData,trainLabel,10)

    (alfaV,_)=expKernelSVMSolve(PDTR[0:8,:],trainLabel,C,gamma=gamma,p0=0.9)
    return (P,alfaV)
            
def applyModel5(trainData,trainLabel,testData,LTE,P,alfaV,gamma=10**-4,be=False):
    trainData=P.T@trainData
    testData=P.T@testData
    (score,predLabel,acc)=inferClassExpSVM(trainData[0:8,:],trainLabel,testData[0:8,:],LTE,alfaV,gamma=gamma)
    

    confM = confusionMatrix(LTE,predLabel)
    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(score,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

    if be:
        calcBayesError(score-numpy.log(0.1/0.9),LTE,color=colorPalette[6],maxShowedCost=0.5,modelN='SVM RBF')


def trainModel6(trainData,trainLabel,l=10**-2):
    PDTR,P=testPCA(trainData,trainLabel,10)
    (A,w,b)=QlogregSolve(PDTR[0:9,:],trainLabel,l,p0=0.9)
    return P,A,w,b

def applyModel6(testData,testLabel,P,A,w,b):
    PDTE,P=testPCA(testData,testLabel,10)
    (score,_,acc)=inferClassQlogreg(PDTE[0:9,:],testLabel,A,w,b)

    llrV = binScoreLR(score)
    threshold=0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(testLabel,predLabelV)

    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,testLabel,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)



def trainModel7(trainData, LTR):
    
    PDTR,P=testPCA(trainData,LTR,10)
    
    gmmC=allClassLBGTied(trainData,LTR,nComp=2)[1]
    
    return (P,gmmC)

def applyModel7(testData,LTE,P,gmmC,be=False):
    
    DTE=P.T@testData
    (score2,_,acc)=inferClassGMMTied(testData,LTE,gmmC,priorProb=[0.9,0.1])
    llrV = binScoreMVG(score2)
    
    threshold=0.0
    predLabelV = numpy.where(llrV>threshold,1,0)#threshold is already included into llr
    confM = confusionMatrix(LTE,predLabelV)

    print("cost actDCF @ WP:%f" % normalizedDCF(confM,0.1,1,1))
    print("accuracy %f"%acc)
    (cost,_,_,_) = incrMinDCF(llrV,LTE,0.1,1,1)
    print("cost incrMinDCF @ WP:%f" % cost)

    if be:
        calcBayesError(llrV-numpy.log(1/9),LTE,color=colorPalette[5],maxShowedCost=0.5,modelN='GMM 2 comp')


if __name__ == '__main__':

    labeledData=load('Train.txt')
    testData=load('Test.txt')
    #flag EVALUATING @ 1 means that I want to execute the code used to analyze different model performances
    if EVALUATING:
        # Visualization of data
        NO_SCATTER=True #too much combination of data to visualize
        print("visualize original data")
        visualizeData(labeledData.dsAttributes,labeledData.dsLabel)
        NO_SCATTER=False
        testLDA(labeledData.dsAttributes,labeledData.dsLabel)#too much information loss, can't find a linear division in resultig data, and since are 1-d can't apply quadratic classifier ->don't use result of LDA, data are still gaussian distributed but doesn't show any particular discriminating shape
        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,6,visualize=True)#the discarded dimension seems to be quite irrelevant but the 6 retained dimension show quite well separed data, with ellipses shape nested into other ellipses
        #95% explained variance
        if not FUSION:
            #K-fold test (6-dimensions 95+% explained variance)
            if GAUSSIAN:

                print("\nMVG")
                print("NO PCA")
                KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=4)
                #***************************************COST: incrMinDCF***************************************
                
                plt.title("PCA_6_10_MVG")
                costV=[]
                costV1=[]
                costV2=[]
                nDimV=[]
                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nMVG")
                    print("ndim PCA %d" % nDim)
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=4,color=colorPalette[0])#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                    costV.append(cost)
                    costV1.append(cost1)
                    costV2.append(cost2)
                    nDimV.append(nDim)
                
                plt.legend(loc='upper left' )
                plt.plot(nDimV, costV, color=colorPalette[0],label='0.1')
                plt.plot(nDimV, costV1, color=colorPalette[1],label='0.5')
                plt.plot(nDimV, costV2, color=colorPalette[2],label='0.05')
                plt.xlim([6, 10])
                plt.ylim([0, 0.4])
                plt.xlabel("PCA dimension")
                plt.ylabel("min DCF")
                plt.plot()
                plt_opt()

                plt.title("PCA_6_10_MVGNaiveBayes")
                costV=[]
                costV1=[]
                costV2=[]
                nDimV=[]
                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)    
                    print("\nMVG Naive-Bayes")
                    print("ndim PCA %d" % nDim)
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=5,color=nDim-6)#slightly lower accuracy lot faster training, good independency assumption: see per class heatmaps
                    costV.append(cost)
                    costV1.append(cost1)
                    costV2.append(cost2)
                    nDimV.append(nDim)
               
                plt.legend(loc='upper left' )
                plt.plot(nDimV, costV, color=colorPalette[0],label='0.1')
                plt.plot(nDimV, costV1, color=colorPalette[1],label='0.5')
                plt.plot(nDimV, costV2, color=colorPalette[2],label='0.05')
                plt.xlim([6, 10])
                plt.ylim([0, 0.4])
                plt.xlabel("PCA dimension")
                plt.ylabel("min DCF")
                plt_opt()

                #I just expect that the tied covariance hypotesis is not true (and not separable linearly like in LDA) -> try just to show I have implemented the model (not plotted)    
                print("\nMVG tied")
                for nDim in range (6,11):
                    print("ndim PCA %d" % nDim)
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=6)#expected low accuracy
                #1.000
                print("\nMVG Naive-Bayes tied")
                for nDim in range (6,11):
                    print("ndim PCA %d" % nDim)
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=7)#expected low accuracy
                #1.000
                
                #best Naive-Bayes PCA 8 (0.260)
            
            if LR:
                if LIN:
                    #I just expect that the hypotesis of linear separable classes is not true -> try just to show I have implemented the model (not plotted)    
                    
                    for i in range (-2,5):
                        for nDim in range (8,11):
                            print("\nLinear Logistic Regression l=%f"%(10**-i))
                            print("PCA %d"%nDim)
                            projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                            KFold(projectedData,labeledData.dsLabel,5,type=8,l=10**-i)
   
                if QUAD:
                    if not PART:
                        costV = []
                        costV1=[]
                        costV2=[]
                        #nDim -> different colors, function of lambda 
                        plt.title("PCA_6_10_NOPCA_Quadratic_LogReg")
                        for nDim in range (6,11):
                            projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                            for i in range(-2,7):#out of this range lr accuracy doesn't change anymore for this dataset
                                l=10**-i
                                print("\nQuadratic Logistic Regression l=%f" % l)
                                print("ndim PCA %d" % nDim)
                                (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=l)
                                costV.append(cost)
                                costV1.append(cost1)
                                costV2.append(cost2) 
                        
                        lambdaV=[]  
                        for i in range(-2,7):#out of this range lr accuracy doesn't change anymore for this dataset
                            l=10**-i
                            print("\nQuadratic Logistic Regression l=%f" % l)
                            print("NO PCA")
                            (cost,cost1,cost2)=KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=9,l=l)
                            costV.append(cost)
                            costV1.append(cost1)
                            costV2.append(cost2)
                            lambdaV.append(i)
                    
                        
                        plt.title("PCA_6_10_NOPCA_LogReg_0_1")
                        
                        for nDim in range (6,11):
                            plt.plot(lambdaV,costV[(nDim-6)*9:(nDim-6+1)*9], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                        plt.plot(lambdaV,costV[5*9:6*9], label='NO PCA', color=colorPalette[5])
                            
                        plt.legend(loc='upper left' )
                        plt.xlim([-2, 6])
                        plt.ylim([0.0, 0.8])
                        plt.xlabel("lambda = 10^-x ")
                        plt.ylabel("min DCF")
                        plt_opt()

                        plt.title("PCA_6_10_NOPCA_LogReg_0_5")
                        
                        for nDim in range (6,11):
                            plt.plot(lambdaV,costV1[(nDim-6)*9:(nDim-6+1)*9], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                        plt.plot(lambdaV,costV1[5*9:6*9], label='NO PCA', color=colorPalette[5])
                            
                        plt.legend(loc='upper left' )
                        plt.xlim([-2, 6])
                        plt.ylim([0.0, 0.8])
                        plt.xlabel("lambda = 10^-x ")
                        plt.ylabel("min DCF")
                        plt_opt()

                        plt.title("PCA_6_10_NOPCA_LogReg_0_05")
                        
                        for nDim in range (6,11):
                            plt.plot(lambdaV,costV2[(nDim-6)*9:(nDim-6+1)*9], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                        plt.plot(lambdaV,costV2[5*9:6*9], label='NO PCA', color=colorPalette[5])
                            
                        plt.legend(loc='upper left' )
                        plt.xlim([-2, 6])
                        plt.ylim([0.0, 0.8])
                        plt.xlabel("lambda = 10^-x ")
                        plt.ylabel("min DCF")
                        plt_opt()

                        #nDim -> different colors, function of lambda
                        #perform bad probably because of rescaling principal component (M and F division)
                    
                        zNormData = zNorm(labeledData.dsAttributes)
                        nDim=8
                        projectedData,_=testPCA(zNormData,labeledData.dsLabel,nDim) #need to provide temporary data array for the structure of test PCA
                        l=10**-2
                        print("\nQuadratic Logistic Regression l=%f" % l)
                        print("ndim PCA %d" % nDim)
                        KFold(zNormData,labeledData.dsLabel,5,type=9,l=l)
                        #0.855

                        l=10**-2
                        print("\nQuadratic Logistic Regression l=%f" % l)
                        print("NO PCA")
                        KFold(zNormData,labeledData.dsLabel,5,type=9,l=l)
                        #0.855


                    #Best Quad LogReg PCA 9 l=10**-2 (0.307)
                if PART:
                    print("\nPCA 6 l=10")
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,6)
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=10)
                    print("\nPCA 6 l=100")
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=100)
                    print("\nPCA 8 l=10**-8")
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-8)
                    print("\nPCA 8 l=10**-9")
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-9)
                    print("\nPCA 9 l=10**-8")
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,9)
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-8)
                    print("\nPCA 9 l=10**-9")
                    (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-9)

            if SVM:

                #put later z-score attempt -> performed poorly
                
                if LIN:
                    #as in the previous model linear model is not able to separate classes -> 1 just to show the implementation
                    C=1.0
                    K=1.0
                    print("\nlinear SVM K=%f, C=%f" % (K,C))
                    KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=10,K=K,C=C)
                    for i in range (-2,5):
                        for nDim in range (8,11):
                            print("\nLinear SVM C=%f"%(10**-i))
                            print("PCA %d"%nDim)
                            projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                            KFold(projectedData,labeledData.dsLabel,5,type=8,C=10**-i,K=K)
                    #1.000
                
                if QUAD:
                    degree=2
                   
                    #nDim -> different colors, function of lambda 
                    
                    costV = []
                    costV1 = []
                    costV2 = []
                    
                    for nDim in range (6,11): 
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                        for i in range(1,5):#out of this range lr accuracy doesn't change anymore for this dataset
                            C=10**-i
                            print("\nQuadratic SVM C=%f" % C)
                            print("ndim PCA %d" % nDim)
                            (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=11,degree=degree,C=C)#balanced version, little error on C
                            costV.append(cost)
                            costV1.append(cost1)
                            costV2.append(cost2)
                            
                    CV=[]
                    for i in range(1,5):#out of this range lr accuracy doesn't change anymore for this dataset
                        C=10**-i
                        print("\nQuadratic SVM C=%f" % C)
                        print("NO PCA")
                        (cost,cost1,cost2)=KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=11,degree=degree,C=C)#balanced version, little error on C
                        costV.append(cost)
                        costV1.append(cost1)
                        costV2.append(cost2)
                        CV.append(i)
                    

                    plt.title("PCA_6_10_NOPCA_QuadraticKernelSVM")
                        
                    for nDim in range (6,11):
                        plt.plot(CV,costV[(nDim-6)*4:(nDim-6+1)*4], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                    plt.plot(CV,costV[5*4:6*4], label='NO PCA', color=colorPalette[5])
                            
                    plt.legend(loc='upper left' )
                    plt.xlim([0, 4])
                    plt.ylim([0.0, 0.8])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_6_10_NOPCA_QuadraticKernelSVM_0_5")
                        
                    for nDim in range (6,11):
                        plt.plot(CV,costV1[(nDim-6)*4:(nDim-6+1)*4], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                    plt.plot(CV,costV1[5*4:6*4], label='NO PCA', color=colorPalette[5])
                            
                    plt.legend(loc='upper left' )
                    plt.xlim([0, 4])
                    plt.ylim([0.0, 0.8])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_6_10_NOPCA_QuadraticKernelSVM_0_05")
                        
                    for nDim in range (6,11):
                        plt.plot(CV,costV2[(nDim-6)*4:(nDim-6+1)*4], label='PCA %d' %nDim, color=colorPalette[nDim-6])
                    plt.plot(CV,costV2[5*4:6*4], label='NO PCA', color=colorPalette[5])
                            
                    plt.legend(loc='upper left' )
                    plt.xlim([0, 4])
                    plt.ylim([0.0, 0.8])
                    plt.xlabel("lambda = 10^-x ")
                    plt.ylabel("C = 10^-x ")
                    plt_opt()
                    

                    #best quad PCA 9  C=10**-2 0.331

                if RBF:
                    
                    if PART:
                        print("\nRBF SVM C=%f gamma=%f" % (10**4,10**-4))
                        print("ndim PCA 8")
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                        KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=10**-4,C=10**4)

                        print("\nRBF SVM C=%f gamma=%f" % (10**3,10**-4))
                        print("ndim PCA 8")
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                        KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=10**-4,C=10**3)


                    #PCA 8
                    CV=[-2,-1,0,1,2]

                    costV = []
                    costV1 = []
                    costV2 = []
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                    for j in range (0,5):
                        gamma=10**-j
                        for i in range(-2,3):#
                            C=10**-i
                            print("\nRBF SVM C=%f gamma=%f" % (C,gamma))
                            print("ndim PCA 8")
                            (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=gamma,C=C)#balanced version, little error on C
                            costV.append(cost)
                            costV1.append(cost1)
                            costV2.append(cost2)
                    
                    
                    plt.title("PCA_8_RBFSVM")
                    for j in range (0,5):
                        plt.plot(CV,costV[(j)*5:(j+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j])

                    plt.legend(loc='upper left' )
                    plt.xlim([-2, 2])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_8_RBFSVM_0_5")
                    for j in range (0,5):
                        plt.plot(CV,costV1[(j)*5:(j+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j])

                    plt.legend(loc='upper left' )
                    plt.xlim([-2, 2])
                    plt.ylim([0.0, 0.5])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_8_RBFSVM_0_05")
                    for j in range (0,5):
                        plt.plot(CV,costV2[(j)*5:(j+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j])

                    plt.legend(loc='upper left' )
                    plt.xlim([-2, 2])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()
                    

                    #PCA 9
                    CV=[-3,-2,-1,0,1]

                    costV = []
                    costV1 = []
                    costV2 = []
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,9)
                    for j in range (2,6):
                        gamma=10**-j
                        for i in range(-3,2):#
                            C=10**-i
                            print("\nRBF SVM C=%f gamma=%f" % (C,gamma))
                            print("ndim PCA 9")
                            (cost,cost1,cost2)=KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=gamma,C=C)#balanced version, little error on C
                            costV.append(cost)
                            costV1.append(cost1)
                            costV2.append(cost2)
                    
                    
                    plt.title("PCA_9_RBFSVM")
                    for j in range (2,6):
                        plt.plot(CV,costV[(j-2)*5:(j-2+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j-2])

                    plt.legend(loc='upper left' )
                    plt.xlim([-3, 1])
                    plt.ylim([0.0, 0.8])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_9_RBFSVM_0_5")
                    for j in range (2,6):
                        plt.plot(CV,costV1[(j-2)*5:(j-2+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j-2])

                    plt.legend(loc='upper left' )
                    plt.xlim([-3, 1])
                    plt.ylim([0.0, 0.5])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()

                    plt.title("PCA_9_RBFSVM_0_05")
                    for j in range (2,6):
                        plt.plot(CV,costV2[(j-2)*5:(j-2+1)*5], label=('gamma %f' % (10**-j)), color=colorPalette[j-2])

                    plt.legend(loc='upper left' )
                    plt.xlim([-3, 1])
                    plt.ylim([0.0, 1.0])
                    plt.xlabel("C = 10^-x ")
                    plt.ylabel("min DCF")
                    plt_opt()
                    
                    #best RBF SVM PCA 8 gamma=10**-3, C=10: 0.304
            
            if GMM:
               
                if not PART:
                    
                    print("\nGMM diag")
                    print("NO PCA")
                    KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=14,nComp=32)

                    plt.title("PCA_6_10_GMM_diagonal")
                    costM=[]
                    costM1=[]
                    costM2=[]
                    for nDim in range (6,11):
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                        print("\nGMM diagonal")
                        print("ndim PCA %d" % nDim)
                        (costV,costV1,costV2)=KFold(projectedData,labeledData.dsLabel,5,type=14,color=colorPalette[0],nComp=32)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                        costM.append(costV)
                        costM1.append(costV1)
                        costM2.append(costV2)
                        
                    x = numpy.array([1,2,3,4,5])  # Bins on the x-axis
                    y = numpy.array([6,7,8,9,10])  # Bins on the y-axis
                    X, Y = numpy.meshgrid(x, y)  # Create a grid of x and y values
                    Z = numpy.array(costM)

                    # Create the Figure and Axes
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Create the 3D Histogram
                    dx = dy = 1  # Width and depth of each bin
                    dz = Z.flatten()  # Flatten the Z matrix
                    xpos, ypos = X.flatten(), Y.flatten()  # Flatten the X and Y matrices
                    num_elements = len(xpos)  # Total number of elements

                    # Create the 3D bars
                    colormap = plt.cm.get_cmap('viridis')
                    norm = Normalize(vmin=dz.min(), vmax=dz.max())
                    dz_normalized = norm(dz)

                    ax.bar3d(xpos, ypos, numpy.zeros(num_elements), dx, dy, dz, color=colormap(dz_normalized), alpha=0.8,linewidth=1, edgecolor='black')

                    sm = ScalarMappable(cmap=colormap)
                    sm.set_array(dz)
                    cbar = plt.colorbar(sm)
                    cbar.set_label('Value')

                    # Add labels and title
                    ax.set_xlabel('2^x components')
                    ax.set_ylabel('PCA dim')
                    ax.set_zlabel('incrMinDCF')
                    ax.set_title('GMM_diag_incrMinDCF(PCA_dim,N_comp)')

                    # Adjust the viewing angle
                    ax.view_init(azim=120, elev=30)
                    plt.title('GMM_diag_incrMinDCF(PCA_dim,N_comp)')
                    plt_opt()

                    #my code stops here

                    print("\nGMM")
                    print("NO PCA")
                    KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=13,nComp=32)
                    #***************************************COST: incrMinDCF***************************************
                    #0.309/0.286
                    costM=[]
                    costM1=[]
                    costM2=[]
                    for nDim in range (6,11):
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                        print("\nGMM")
                        print("ndim PCA %d" % nDim)
                        (costV,costV1,costV2)=KFold(projectedData,labeledData.dsLabel,5,type=13,color=colorPalette[0],nComp=32)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                        costM.append(costV)
                        costM1.append(costV1)
                        costM2.append(costV2)
                        
                    x = numpy.array([1,2,3,4,5])  # Bins on the x-axis
                    y = numpy.array([6,7,8,9,10])  # Bins on the y-axis
                    X, Y = numpy.meshgrid(x, y)  # Create a grid of x and y values
                    Z = numpy.array(costM)
                
                    # Create the Figure and Axes
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Create the 3D Histogram
                    dx = dy = 1  # Width and depth of each bin
                    dz = Z.flatten()  # Flatten the Z matrix
                    xpos, ypos = X.flatten(), Y.flatten()  # Flatten the X and Y matrices
                    num_elements = len(xpos)  # Total number of elements

                    # Create the 3D bars
                    colormap = plt.cm.get_cmap('viridis')
                    norm = Normalize(vmin=dz.min(), vmax=dz.max())
                    dz_normalized = norm(dz)

                    ax.bar3d(xpos, ypos, numpy.zeros(num_elements), dx, dy, dz, color=colormap(dz_normalized), alpha=0.8,linewidth=1, edgecolor='black')

                    sm = ScalarMappable(cmap=colormap)
                    sm.set_array(dz)
                    cbar = plt.colorbar(sm)
                    cbar.set_label('Value')

                    # Add labels and title
                    ax.set_xlabel('2^x components')
                    ax.set_ylabel('PCA dim')
                    ax.set_zlabel('incrMinDCF')
                    ax.set_title('GMM_incrMinDCF(PCA_dim,N_comp)')

                    # Adjust the viewing angle
                    ax.view_init(azim=120, elev=30)
                    plt.title('GMM_incrMinDCF(PCA_dim,N_comp)')
                    plt_opt()
                    
                    #GMM tied
                    print("\nGMM tied")
                    print("NO PCA")
                    KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=15,nComp=32)
                    #***************************************COST: incrMinDCF***************************************
                    #0.270/0.270
                    costM=[]
                    costM1=[]
                    costM2=[]
                    for nDim in range (6,11):
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                        print("\nGMM tied")
                        print("ndim PCA %d" % nDim)
                        (costV,costV1,costV2)=KFold(projectedData,labeledData.dsLabel,5,type=15,color=colorPalette[0],nComp=32)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                        costM.append(costV)
                        costM1.append(costV1)
                        costM2.append(costV2)
                        
                    x = numpy.array([1,2,3,4,5])  # Bins on the x-axis
                    y = numpy.array([6,7,8,9,10])  # Bins on the y-axis
                    X, Y = numpy.meshgrid(x, y)  # Create a grid of x and y values
                    Z = numpy.array(costM)
                
                    # Create the Figure and Axes
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Create the 3D Histogram
                    dx = dy = 1  # Width and depth of each bin
                    dz = Z.flatten()  # Flatten the Z matrix
                    xpos, ypos = X.flatten(), Y.flatten()  # Flatten the X and Y matrices
                    num_elements = len(xpos)  # Total number of elements

                    # Create the 3D bars
                    colormap = plt.cm.get_cmap('viridis')
                    norm = Normalize(vmin=dz.min(), vmax=dz.max())
                    dz_normalized = norm(dz)
                    ax.bar3d(xpos, ypos, numpy.zeros(num_elements), dx, dy, dz, color=colormap(dz_normalized), alpha=0.8,linewidth=1, edgecolor='black')

                    sm = ScalarMappable(cmap=colormap)
                    sm.set_array(dz)
                    cbar = plt.colorbar(sm)
                    cbar.set_label('Value')

                    # Add labels and title
                    ax.set_xlabel('2^x components')
                    ax.set_ylabel('PCA dim')
                    ax.set_zlabel('incrMinDCF')
                    ax.set_title('GMM_tied_incrMinDCF(PCA_dim,N_comp)')

                    # Adjust the viewing angle
                    ax.view_init(azim=120, elev=30)
                    plt.title('GMM_tied_incrMinDCF(PCA_dim,N_comp)')
                    plt_opt()
                    
                    #print("\nGMM tied diagonal")
                    #print("NO PCA")
                    #KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=16,nComp=8)
                    #***************************************COST: incrMinDCF***************************************
                    
                    costM=[]
                    costM1=[]
                    costM2=[]
                    for nDim in range (6,11):
                        projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                        print("\nGMM tied diagonal")
                        print("ndim PCA %d" % nDim)
                        (costV,costV1,costV2)=KFold(projectedData,labeledData.dsLabel,5,type=16,color=colorPalette[0],nComp=32)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                        costM.append(costV)
                        costM1.append(costV1)
                        costM2.append(costV2)
                        
                    x = numpy.array([1,2,3,4,5])  # Bins on the x-axis
                    y = numpy.array([6,7,8,9,10])  # Bins on the y-axis
                    X, Y = numpy.meshgrid(x, y)  # Create a grid of x and y values
                    Z = numpy.array(costM)
                
                    # Create the Figure and Axes
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    # Create the 3D Histogram
                    dx = dy = 1  # Width and depth of each bin
                    dz = Z.flatten()  # Flatten the Z matrix
                    xpos, ypos = X.flatten(), Y.flatten()  # Flatten the X and Y matrices
                    num_elements = len(xpos)  # Total number of elements

                    # Create the 3D bars
                    colormap = plt.cm.get_cmap('viridis')
                    norm = Normalize(vmin=dz.min(), vmax=dz.max())
                    dz_normalized = norm(dz)
                    ax.bar3d(xpos, ypos, numpy.zeros(num_elements), dx, dy, dz, color=colormap(dz_normalized), alpha=0.8,linewidth=1, edgecolor='black')

                    sm = ScalarMappable(cmap=colormap)
                    sm.set_array(dz)
                    cbar = plt.colorbar(sm)
                    cbar.set_label('Value')

                    # Add labels and title
                    ax.set_xlabel('2^x components')
                    ax.set_ylabel('PCA dim')
                    ax.set_zlabel('incrMinDCF')
                    ax.set_title('GMM_tied_diag_incrMinDCF(PCA_dim,N_comp)')

                    # Adjust the viewing angle
                    ax.view_init(azim=120, elev=30)
                    plt.title('GMM_tied_diag_incrMinDCF(PCA_dim,N_comp)')
                    plt_opt()
                    
            if PART:

                    #***************************************COST: incrMinDCF***************************************
                    #0.309/0.286 
                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM Mixed 2 comp")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=17,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                
                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM Mixed 2 comp 1-2")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=18,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)

                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM 2 comp")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=13,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)
                    

                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM diag 2 comp")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=14,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)

                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM tied 2 comp")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=15,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)

                for nDim in range (6,11):
                    projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,nDim)
                    print("\nGMM tied diag 2 comp")
                    print("ndim PCA %d" % nDim)
                    KFold(projectedData,labeledData.dsLabel,5,type=16,color=colorPalette[0],nComp=2)#reach highest accuracy but doesn't account for separation which seems to be a good assumption (on attribute 0 only one ellipses separates data instead of 2)

                
        #-------------------------------------------------------------------

    if CALIBRATION: 
        if not BEST: 
            if ROC:
                print("ROC")
                print("MVG")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=5,color=colorPalette[0],roc=True)
                print("SVM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=10**-4,C=10**3,color=colorPalette[1],roc=True)
                print("GMM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,7)
                KFold(projectedData,labeledData.dsLabel,5,type=15,nComp=2,color=colorPalette[2],roc=True)
                print("QLR")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,9)
                KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-2,color=colorPalette[3],roc=True)

                plt.legend(loc='upper left' )
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt_opt()
            
            if DET:
                print("DET")
                print("MVG")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=5,color=colorPalette[0],det=True)
                print("SVM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=10**-4,C=10**3,color=colorPalette[1],det=True)
                print("GMM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,7)
                KFold(projectedData,labeledData.dsLabel,5,type=15,nComp=2,color=colorPalette[2],det=True)
                print("QLR")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,9)
                KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-2,color=colorPalette[3],det=True)

                plt.legend(loc='upper left' )
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt_opt()

            if BEP:
                print("Bayes Error Plots")
                print("MVG")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=5,color=colorPalette[0],be=True)
                print("SVM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                KFold(projectedData,labeledData.dsLabel,5,type=12,gamma=10**-4,C=10**3,color=colorPalette[1],be=True)
                print("GMM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,7)
                KFold(projectedData,labeledData.dsLabel,5,type=15,nComp=2,color=colorPalette[2],be=True)
                print("QLR")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,9)
                KFold(projectedData,labeledData.dsLabel,5,type=9,l=10**-2,color=colorPalette[3],be=True)

                plt.legend(loc='upper left' )
                plt_opt()

        if BEST:
            if FUSION:
                print("MVG")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,8)
                (score1V,labelV)=KFold(projectedData,labeledData.dsLabel,5,type=5,color=colorPalette[0],be=True)
                print("GMM")
                projectedData,_=testPCA(labeledData.dsAttributes,labeledData.dsLabel,7)
                (score2V,labelV)=KFold(projectedData,labeledData.dsLabel,5,type=15,nComp=2,color=colorPalette[1],be=True)
                #labelV (and order of sample) are the same since random seed is the same
                print("fusing with LR")
                fusionData=numpy.vstack((score1V.reshape((3500,)),score2V.reshape((3500,))))
                
                KFold(fusionData,labelV,5,type=8, seed=1, be=True, color=colorPalette[2], l=10**-2)
                plt.title('fusion')
                plt.legend(loc='upper left' )
                plt_opt()

    if APPLY:
        visualizeData(testData.dsAttributes,testData.dsLabel)
        #print("try GMM")
        #KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=13,color=colorPalette[0],nComp=2)

        #print("try GMM tied")
        #KFold(labeledData.dsAttributes,labeledData.dsLabel,5,type=15,color=colorPalette[0],nComp=2)


        #(P,gmmC)=trainModel7(labeledData.dsAttributes,labeledData.dsLabel)
        #applyModel7(testData.dsAttributes,testData.dsLabel,P,gmmC)

        #testPCA(testData.dsAttributes,testData.dsLabel,10,visualize=True)
        
        #(P,mvgmuC,mvgCC,gmmC,w,b)=trainModel(labeledData.dsAttributes,labeledData.dsLabel)
        #applyModel(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC,gmmC,w,b,be=True)
        #(P,mvgmuC,mvgCC,gmmC,w,b)=trainModel4(labeledData.dsAttributes,labeledData.dsLabel)
        #applyModel4(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC,gmmC,w,b,be=True)
        #(P,alfa)=trainModel5(labeledData.dsAttributes,labeledData.dsLabel,gamma=10**-3,C=100)
        #applyModel5(labeledData.dsAttributes,labeledData.dsLabel,testData.dsAttributes,testData.dsLabel,P,alfa,gamma=10**-3,be=True)
        #plt_opt()
        
        (P,mvgmuC,mvgCC,gmmC,w,b)=trainModel(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC,gmmC,w,b,be=True)
        (P,gmmC)=trainModel7(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel7(testData.dsAttributes,testData.dsLabel,P,gmmC,be=True)
        plt_opt()

        
        print("\n selected")
        (P,mvgmuC,mvgCC,gmmC,w,b)=trainModel(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC,gmmC,w,b)
        print("\n")
        (P,gmmC)=trainModel2(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel2(testData.dsAttributes,testData.dsLabel,P,gmmC)
        print("\n")
        (P,mvgmuC,mvgCC)=trainModel3(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel3(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC)
        print("\n")
        (P,mvgmuC,mvgCC,gmmC,w,b)=trainModel4(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel4(testData.dsAttributes,testData.dsLabel,P,mvgmuC,mvgCC,gmmC,w,b)
        print("\n SVM")
        (P,alfa)=trainModel5(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel5(labeledData.dsAttributes,labeledData.dsLabel,testData.dsAttributes,testData.dsLabel,P,alfa)
        print("\n QLR")
        (P,A,w,b)=trainModel6(labeledData.dsAttributes,labeledData.dsLabel)
        applyModel6(testData.dsAttributes,testData.dsLabel,P,A,w,b)

        for i in range (0,7):
            l=10**-i
            print("\nlambda: %f" %l)
            (P,A,w,b)=trainModel6(labeledData.dsAttributes,labeledData.dsLabel,l=l)
            applyModel6(testData.dsAttributes,testData.dsLabel,P,A,w,b)

        for i in range (0,6):
            gamma=10**-i
            for j in range (-2,5):
                C=10**-j 
                print("\ngamma: %f, C: %f" %(gamma,C))
                (P,alfa)=trainModel5(labeledData.dsAttributes,labeledData.dsLabel,gamma=gamma,C=C)
                applyModel5(labeledData.dsAttributes,labeledData.dsLabel,testData.dsAttributes,testData.dsLabel,P,alfa,gamma=gamma)


       