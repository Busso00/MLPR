
import numpy
import matplotlib.pyplot as plt
import numpy.linalg
import scipy.linalg
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

attributeToN={'attr1':0,'attr2':1,'attr3':2, 'attr4':3, 'attr5':4, 'attr6':5, 'attr7':6, 'attr8':7, 'attr9':8, 'attr10':9}
nToAttribute=['attr1','attr2','attr3','attr4','attr5','attr6','attr7','attr8','attr9','attr10']
labelToN={'0':0,'1':1}
nToLabel=['0','1']

#N_ATTRS = len(nToAttribute) #can change with dimensionality reduction
N_LABELS = len(nToLabel) #generally doesn't change

#N_RECORDS = 3500 #can change between train and test

FILENAME="Train.txt"

SHOW = True #if false no image display
NO_HIST = False #if true no histogram display
NO_SCATTER = False #if true no scatter display
NO_HEATMAP = False #if true no heatmap display

EVALUATING = False #if true model is not definitive
CALIBRATION = False
#ZSCORE = True #if true z-score is applied 


FUSION = True

GAUSSIAN = False #if true gaussians models are evaluated
LR = False #if true logistic regression models are evaluated
SVM = False #if true support vector machine models are evaluated

LIN = False
QUAD = False
RBF = True

PART = True

GMM = True

ROC = False
DET = False
BEP = True
BEST = True

APPLY = True #if true use final model

PC=[2000/3500,1500/3500]#PC=[1/2,1/2]#prior class probabilities around 50/50 but for a more accurate model I prefer to use real prior

colorPalette = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'cyan']