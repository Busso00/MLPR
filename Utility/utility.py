import sys
import os
sys.path.append('..')
from const import *

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

def plt_opt():
    if SHOW:
        # Set the desired filename and file format
        filename = "./figure/%s.png"%plt.gca().get_title()

        # Check if the file already exists
        if os.path.exists(filename):
            # If the file exists, modify the filename by adding a number suffix
            suffix = 1
            while os.path.exists(f"{os.path.splitext(filename)[0]}_{suffix}{os.path.splitext(filename)[1]}"):
                suffix += 1
            filename = f"{os.path.splitext(filename)[0]}_{suffix}{os.path.splitext(filename)[1]}"

        # Save the figure with the modified filename
        plt.savefig(filename)
        plt.clf()

class DataList:
    def __init__(self):
        self.dsAttributes=[]
        self.dsLabel=[]

class DataArray:

    def __init__(self,listAttr,listLabel):
        self.dsAttributes=numpy.vstack(listAttr).T
        self.dsLabel=numpy.array(listLabel,dtype=numpy.int32)
   
        
def load(filename):
    try:
        f=open(filename,'r')
    except:
        print("error opening Dataset")
        exit(-1)
    
    labeledData=DataList()
    for line in f:
        try:
            record=line.split(',')
            record[-1]=record[-1].replace('\n', '')
            attributes=numpy.array([float(i) for i in record[0:-1]])
            label=labelToN[record[-1]]
            labeledData.dsAttributes.append(attributes)
            labeledData.dsLabel.append(label)
        except:
            print("error parsing line")

    labeledData=DataArray(labeledData.dsAttributes,labeledData.dsLabel)
    return labeledData

def plotHist(dataV,labelV,useUnnamed=False):
    if NO_HIST:
        return
    
    N_ATTRS=dataV.shape[0]
    
    for i in range(N_ATTRS):
        plt.title("Histogram %d "%i)
        if (useUnnamed):
            plt.xlabel("attribute%d"%i)
        else:
            plt.xlabel(nToAttribute[i])
        for j in range(N_LABELS):
            plt.hist(dataV[:,labelV==j][i,:],label=nToLabel[j],alpha = 0.3,bins=20,density=True) #alpha = trasparenza, bins=numero divisioni, density=true->normalizza t.c sum(numero_valori_bin*ampiezza_bin)=1  ->scala altezza e mostra circa una gaussiana
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('hist_%d_%s.pdf' % (i,nToAttribute[i]))

        plt_opt()

def plotScatter(dataV,labelV,useUnnamed=False):
    if NO_SCATTER:
        return
    
    N_ATTRS=dataV.shape[0]

    for i in range(N_ATTRS):
        for j in range(i):#doesn't print the same figure with inverted axis
            plt.title("Scatter %d-%d "%(i,j))
            if (useUnnamed):
                plt.xlabel("attribute%d"%i)
                plt.ylabel("attribute%d"%j)
            else:
                plt.xlabel(nToAttribute[i])
                plt.ylabel(nToAttribute[j])
            for k in range(N_LABELS):
                plt.scatter(dataV[:,labelV==k][i,:],dataV[:,labelV==k][j,:],label=nToLabel[k],alpha=0.3)
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%s_%s.pdf' % (nToAttribute[i],nToAttribute[j]))

            plt_opt()

def meanAttr(dataV):
    N_ATTRS=dataV.shape[0]

    mean=vcol(dataV.mean(axis=1))
    
    print("Mean:")
    for i in range(N_ATTRS):
        print("%s:\t%.2f" % (nToAttribute[i],mean[i,0]))
    print("\n")

    return mean


def stdDevAttr(dataV):
    N_ATTRS=dataV.shape[0]

    stdDev=vcol(dataV.std(axis=1))

    print("Standard Deviation:")
    for i in range(N_ATTRS):
        print("%s:\t%.2f" % (nToAttribute[i],stdDev[i,0]))
    print("\n")

    return stdDev

def corrM(dataV):
    N_ATTRS=dataV.shape[0]

    corrM=numpy.corrcoef(dataV)
    if corrM.shape == ():
        corrM = numpy.array([[corrM]])

    print("Correlation coefficient (Pearson):")
    for i in range(N_ATTRS+1):
        for j in range(N_ATTRS+1):
            if  (i==0):
                if(j==0):
                    print("\t",end="")
                else:
                    print(nToAttribute[j-1]+"\t\t",end="")
            else:
                if(j==0):
                    print(nToAttribute[i-1]+"\t",end="")
                else:
                    print("%.2f\t\t"%(corrM[i-1][j-1]),end="")

        print("")
    print("\n")

    if NO_HEATMAP:
        return corrM
    #print a heatmap to better visualize correlatins between data

    plt.title("Heatmap")
    plt.imshow(corrM, cmap='coolwarm', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Correlation Coefficient')

    # Add ticks and labels
    plt.xticks(range(len(corrM)), nToAttribute[0:dataV.shape[0]] )
    plt.yticks(range(len(corrM)), nToAttribute[0:dataV.shape[0]] )

    # Add gridlines
    plt.grid(visible=False)

    # Show the plot
    plt_opt()

    return corrM
        
def covM(dataV):
    N_ATTRS=dataV.shape[0]
    covM=numpy.cov(dataV,bias=True)
    if covM.shape == ():
        covM = numpy.array([[covM]])

    print("Covariance:")
    for i in range(N_ATTRS+1):
        for j in range(N_ATTRS+1):
            if  (i==0):
                if(j==0):
                    print("\t",end="")
                else:
                    print(nToAttribute[j-1]+"\t\t",end="")
            else:
                if(j==0):
                    print(nToAttribute[i-1]+"\t",end="")
                else:
                    print("%.2f\t\t"%(covM[i-1][j-1]),end="")

        print("")
    print("\n")

    return covM


def classCorrelation(dataV,labelV):
    N_ATTRS=dataV.shape[0]

    Ccv=numpy.zeros((N_ATTRS,N_ATTRS,N_LABELS))
    for c in range(N_LABELS):
        elementOfC=dataV[:,labelV==c]
        print("correlation matrix of class %d:" % c)
        Ccv[:,:,c]+=corrM(elementOfC)
        #print(Ccv[:,:,c]) #no --> already done in corrM
    return Ccv

def classCovariance(dataV,labelV):
    N_ATTRS=dataV.shape[0]

    Swv=numpy.zeros((N_ATTRS,N_ATTRS,N_LABELS))
    for c in range(N_LABELS):
        elementOfC=dataV[:,labelV==c]
        print("covariance matrix of class %s"%nToLabel[c])
        Swv[:,:,c]+=covM(elementOfC)
        # print(Swv[:,:,c]) #no --> already done in corrM
    return Swv

def withinClassCovarianceM(dataV,labelV):
    N_RECORDS=dataV.shape[1]
    N_ATTRS=dataV.shape[0]


    Sw=numpy.zeros((N_ATTRS,N_ATTRS))
    for c in range(N_LABELS):
        elementOfC=dataV[:,labelV==c]
        nc=elementOfC.shape[1]
        Sw+=(numpy.cov(elementOfC,bias=True)*nc)/N_RECORDS
    return Sw

def betweenClassCovarianceM(dataV,labelV):
    N_RECORDS=dataV.shape[1]
    N_ATTRS=dataV.shape[0]

    avg=vcol(dataV.mean(axis=1))
    Sb=numpy.zeros((N_ATTRS,N_ATTRS))
    for c in range(N_LABELS):
        elementOfC=dataV[:,labelV==c]
        avgOfC=vcol(elementOfC.mean(axis=1))
        nc=elementOfC.shape[1]
        Sb+=(((avgOfC-avg)@(avgOfC-avg).T)*nc)/N_RECORDS
    return Sb

def SbSw(D,L):
    Sb=0
    Sw=0
    mu=vcol(D.mean(axis=1))
    for i in range(N_LABELS):
        DCIs=D[:,L==i]
        muCIs=vcol(DCIs.mean())
        Sw+=numpy.dot(DCIs-muCIs,(DCIs-muCIs).T)
        Sb+=DCIs.shape[1]*numpy.dot(muCIs-mu,(muCIs-mu).T)
    Sw/=D.shape[1]
    Sb/=D.shape[1]
    return (Sb,Sw)

def visualizeData(data,labels):
    print()
    print("Attributes matrix shape:")
    print(data.shape)
    print("Label vector shape:")
    print(labels.shape)
    plotHist(data,labels)
    plotScatter(data,labels)
    print()
    meanAttr(data)
    stdDevAttr(data)
    covM(data)
    corrM(data)#1
    classCorrelation(data,labels)#2,3
    classCovariance(data,labels)

def zNorm(data):
    return (data-vcol(data.mean(axis=1)))/vcol(data.std(axis=1))