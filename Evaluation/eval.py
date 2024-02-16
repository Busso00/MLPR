import sys
sys.path.append('..')
from const import *
from Utility.utility import *

def confusionMatrix(testLabelV, predictedLabelV):
    N_ATTRS=testLabelV.shape[0]

    confM=numpy.zeros((len(nToLabel),len(nToLabel)))
    for i in range(N_ATTRS):
        confM[int(predictedLabelV[i])][int(testLabelV[i])]+=1
    return confM

def binaryTaskEvaluation(binScoreV,p1,CostFN,CostFP):
    t=-numpy.log(p1*CostFN/((1-p1)*CostFP))
    return numpy.where(binScoreV>t,1,0)

def binScoreMVG(binScoreV):
    return binScoreV[1,:]-binScoreV[0,:]

def binScoreLR(binScoreV):
    return binScoreV[0,:]

def binScoreSVM(binScoreV):
    return binScoreV[0,:]

def DCF(ConfM,p1,CostFN,CostFP):
    fnr=ConfM[0][1]/(ConfM[0][1]+ConfM[1][1])
    fpr=ConfM[1][0]/(ConfM[1][0]+ConfM[0][0])
    return p1*CostFN*fnr+(1-p1)*CostFP*fpr

def normalizedDCF(ConfM,p1,CostFN,CostFP):
    return DCF(ConfM,p1,CostFN,CostFP)/min(p1*CostFN,(1.0-p1)*CostFP)

def DCFrates (ConfM,p1,CostFN,CostFP):
    fnr=ConfM[0][1]/(ConfM[0][1]+ConfM[1][1])
    fpr=ConfM[1][0]/(ConfM[1][0]+ConfM[0][0])
    return (p1*CostFN*fnr+(1-p1)*CostFP*fpr,fnr,fpr)

def normalizedDCFrates(ConfM,p1,CostFN,CostFP):
    (DCF,fnr,fpr)=DCFrates(ConfM,p1,CostFN,CostFP)
    return (DCF/min(p1*CostFN,(1.0-p1)*CostFP),fnr,fpr)

def minDCF(binScoreV,testLabelV,p1,CostFN,CostFP): #left just for comprehension of incrMinDCF
    minThreshold = -1
    minDCF = numpy.finfo(numpy.float64).max #modify
    #threshold is variable meaning that i compare the best result that model can give provided his hyperparameters/preprocessing steps
    #-> the metrics measure also non probabilistic model
    for threshold in numpy.sort(numpy.copy(binScoreV)):
        predLabelV = numpy.where(binScoreV>threshold,1,0)
        confM = confusionMatrix(testLabelV,predLabelV)
        normDCF = normalizedDCF(confM,p1,CostFN,CostFP)
        if(normDCF<minDCF):
            minDCF=normDCF
            minThreshold=threshold
    
    return (minDCF,minThreshold) #doesn't return fnrV fprV 


def incrMinDCF(binScoreV,testLabelV,p1,CostFN,CostFP):#incremental more efficient version minDCF
    minDCF = numpy.finfo(numpy.float64).max
    N_SCORES=len(binScoreV)
    binScoreLabels = numpy.zeros((2,N_SCORES))

    for i in range(N_SCORES):
        binScoreLabels[:,i] = numpy.asarray([binScoreV[i], testLabelV[i]])
    fnrV=numpy.zeros((N_SCORES,))
    fprV=numpy.zeros((N_SCORES,))

    ind = numpy.argsort( binScoreLabels[0,:] ); #binScoreV needed to sort testLabel
    binScoreLabels = binScoreLabels[:,ind]
    
    predLabelV = numpy.ones(N_SCORES)
    confM = confusionMatrix(binScoreLabels[1,:],predLabelV)
    minThreshold = binScoreLabels[0][0]
    fnrV[0]=confM[0][1]
    fprV[0]=confM[1][0]

    for i in range(0,binScoreLabels.shape[1]):
        (normDCF,fnrV[i],fprV[i]) = normalizedDCFrates(confM,p1,CostFN,CostFP)
        
        if(normDCF<minDCF):
            minDCF=normDCF
            minThreshold=binScoreLabels[1][i]
        
        #update confusion matrix (change only 2 values)
        confM[0][int(binScoreLabels[1][i])] += 1 
        confM[1][int(binScoreLabels[1][i])] -= 1

    #error with p1 low -->all labeled as 0
    return (minDCF,minThreshold,fnrV,fprV) #return fnrV  & fprV

def plotROC(fnrV,fprV,color,modelN=0):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fprV, 1-fnrV , color, label=('roc %d'%modelN))
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    

def calcROC(binScoreV, testLabelV, p1, CostFN, CostFP, color=0,modelN=0):
    (_,_,fnrV,fprV)=incrMinDCF(binScoreV,testLabelV,p1,CostFN,CostFP)
    plotROC(fnrV,fprV,color,modelN=modelN)

def plotDET(fnrV,fprV,color,modelN=0):
    plt.title('Detection Error Trade-off')
    plt.plot(fprV, fnrV , color, label=('roc %d'%modelN))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('log10(FPR)')
    plt.ylabel('log10(FNR)')

def calcDET(binScoreV, testLabelV, p1, CostFN, CostFP, color=0,modelN=0):
    (_,_,fnrV,fprV)=incrMinDCF(binScoreV,testLabelV,p1,CostFN,CostFP)
    plotDET(fnrV,fprV,color,modelN=modelN)

def calcBayesError(binScoreV,testLabelV,color=0,maxShowedCost=1.0,modelN=0):
    effPriorLogOdds = numpy.linspace(-4, 4, 51)
    effPriors = numpy.exp(-numpy.logaddexp(0,-effPriorLogOdds))#ln(pi0/p1)
    N_EVAL_PRIORS=len(effPriors)
    normDCFV = numpy.zeros((N_EVAL_PRIORS,))
    minDCFV = numpy.zeros((N_EVAL_PRIORS,))

    for i in range(N_EVAL_PRIORS):
        ep=effPriors[i]
        
        predLabelV = binaryTaskEvaluation(binScoreV,ep,1,1)
        confM = confusionMatrix(testLabelV,predLabelV)
        normDCFV[i] = normalizedDCF(confM,ep,1,1)
        
        (minDCFV[i],_,_,_)=incrMinDCF(binScoreV,testLabelV,ep,1,1)

    plt.title('Bayes Error')
    plt.plot(effPriorLogOdds, normDCFV, label=('act DCF %s'%modelN), color=color)
    plt.plot(effPriorLogOdds, minDCFV, label=('min DCF %s'%modelN), color=color, linestyle='--' ) 
    plt.ylim([0, 0.8])
    plt.xlim([-4, 4])#can plot wider renge to shom more umbalanced working points
    plt.ylabel("norm DCF")
    plt.xlabel("ln(p1/p0)")
    
    