from json.encoder import INFINITY
import random
import numpy as np
import os
import sys
import statistics as stat
import math

path_to_txt_files=os.getcwd() + "/files"

sys.argv.pop(0)
arguments=sys.argv


name_of_training_file=arguments[arguments.index("--train")+1]
name_test_file=arguments[arguments.index("--test")+1]
sructure_of_neural_net=arguments[arguments.index("--nn")+1]
popsize=int(arguments[arguments.index("--popsize")+1])
numOf_elitist=int(arguments[arguments.index("--elitism")+1])
probability_of_mutation=float(arguments[arguments.index("--p")+1])
standard_deviation_gaus_mutation=float(arguments[arguments.index("--K")+1])
number_of_iterations_of_genAlgo=int(arguments[arguments.index("--iter")+1])

path_to_training_data=path_to_txt_files+"/"+name_of_training_file
path_to_testing_data=path_to_txt_files+"/"+name_test_file


trainingDataNotSplit=open(path_to_training_data).read().split("\n")
trainingDataNotSplit.pop(0)
trainingDataNotSplit.pop(-1)

testingDataNotSplit=open(path_to_testing_data).read().split("\n")
testingDataNotSplit.pop(0)
testingDataNotSplit.pop(-1)

trainingData=[]

def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

for row in trainingDataNotSplit:
    innerList=[]
    values=row.split(",")
    for number in values:
        innerList.append(float(number))
    trainingData.append(innerList)

testingData=[]
for row in testingDataNotSplit:
    innerList=[]
    values=row.split(",")
    for number in values:
        innerList.append(float(number))
    testingData.append(innerList)






allPopulation=[]

def goThrough5sNeuronNet1(weightsArray,ulaz):
    fistWeights=weightsArray[0:5]
    secondWeights=weightsArray[5:10]
    freeWeightHiddenLayer=weightsArray[10:15]
    freeWeightLastNeuron=weightsArray[15]
    
    outputHiddenLayer=np.add(np.multiply(fistWeights,ulaz),freeWeightHiddenLayer)
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer[i]))
    out=np.matmul(newArr,secondWeights)+freeWeightLastNeuron

    return out
    


def goThrough5sNeuronNet2(weightsArray,inputs):
    fistWeights=weightsArray[0:10]
    secondWeights=weightsArray[10:15]
    freeWeightHiddenLayer=weightsArray[15:20]
    freeWeightLastNeuron=weightsArray[20]

    
    fistWeights=np.array(fistWeights).reshape(5, 2)
    inputs=np.array(inputs).reshape(2,1)
    
    outputHiddenLayer=np.add(np.matmul(fistWeights,inputs),np.array(freeWeightHiddenLayer).reshape(5,1))
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer[i]))

    output=np.matmul(newArr,secondWeights)+freeWeightLastNeuron

    return output

def goThrough20sNeuronNet1(weightsArray,ulaz):
    fistWeights=weightsArray[0:20]
    secondWeights=weightsArray[20:40]
    freeWeightHiddenLayer=weightsArray[40:60]
    freeWeightLastNeuron=weightsArray[60]
    
    outputHiddenLayer=np.add(np.multiply(fistWeights,ulaz),freeWeightHiddenLayer)
    newArr=[]
    for i in range(20):
        newArr.append(stable_sigmoid(outputHiddenLayer[i]))
    output=np.matmul(newArr,secondWeights)+freeWeightLastNeuron


    return output

def goThrough20sNeuronNet2(weightsArray,inputs):
    fistWeights=weightsArray[0:40]
    secondWeights=weightsArray[40:60]
    freeWeightHiddenLayer=weightsArray[60:80]
    freeWeightLastNeuron=weightsArray[80]
    
    fistWeights=np.array(fistWeights).reshape(20, 2)
    inputs=np.array(inputs).reshape(2,1)
    
    outputHiddenLayer=np.add(np.matmul(fistWeights,inputs),np.array(freeWeightHiddenLayer).reshape(20,1))
    newArr=[]
    for i in range(20):
        newArr.append(stable_sigmoid(outputHiddenLayer[i]))

    output=np.matmul(newArr,secondWeights)+freeWeightLastNeuron

    return output

def goThrough5s5sNeuronNet1(weightsArray,ulaz):
    fistWeights=weightsArray[0:5]
    secondWeights=np.array(weightsArray[5:30]).reshape(5,5)
    thirdWeights=np.array(weightsArray[30:35]).reshape(5,1)
    freeWeights1Layer=weightsArray[35:40]
    freeWeights2Layer=np.array(weightsArray[40:45]).reshape(5,1)
    freeWeightLastNeuron=weightsArray[45]

    outputHiddenLayer1=np.add(np.multiply(fistWeights,ulaz),freeWeights1Layer)
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer1[i]))
    newArr=np.array(newArr).reshape(5,1)
    
    outputHiddenLayer2=np.add(np.matmul(secondWeights,newArr),freeWeights2Layer)
    
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer2[i]))
    
    output=np.matmul(newArr,thirdWeights)+freeWeightLastNeuron
    return output

def goThrough5s5sNeuronNet2(weightsArray,inputs):
    fistWeights=weightsArray[0:10]
    secondWeights=np.array(weightsArray[10:35]).reshape(5,5)
    thirdWeights=np.array(weightsArray[35:40]).reshape(5,1)
    freeWeights1Layer=weightsArray[40:45]
    freeWeights2Layer=np.array(weightsArray[45:50]).reshape(5,1)
    freeWeightLastNeuron=weightsArray[50]

    fistWeights=np.array(fistWeights).reshape(5, 2)
    inputs=np.array(inputs).reshape(2,1)

    outputHiddenLayer1=np.add(np.matmul(fistWeights,inputs),np.array(freeWeights1Layer).reshape(5,1))
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer1[i]))
    newArr=np.array(newArr).reshape(5,1)
    
    outputHiddenLayer2=np.add(np.matmul(secondWeights,newArr),freeWeights2Layer)
    
    newArr=[]
    for i in range(5):
        newArr.append(stable_sigmoid(outputHiddenLayer2[i]))
    
    output=np.matmul(newArr,thirdWeights)+freeWeightLastNeuron
    return output





def getNSmallestValues(list, n):
    newList=[]
    for x in list:
        newList.append(x)
    returnList=[]
     
    for i in range(n):
        smallestValue=INFINITY
        indexToPop=0
        lenOfArray=len(newList)
        for i in range(lenOfArray):
            if(smallestValue > newList[i]):
                smallestValue=newList[i]
                indexToPop=i
        newList.pop(indexToPop)
        returnList.append(smallestValue)
    
    return returnList

def getIdexesOfValues(bigList, smallList):
    returnList=[]
    for value in smallList:
        returnList.append(bigList.index(value))
    return returnList

##we get a rulet, example; [0.2, 0.6, 1.0]
##argument: deviationArray
def getRoulette(list):
    invertedValues=[]
    rouletteList=[]
    sumList=sum(list)
    for elem in list:
        invertedValues.append(sumList-elem)
    
    sumInvertedList=sum(invertedValues)
    sumUntilNow=0
    for e in invertedValues:

        rouletteList.append(e/sumInvertedList + sumUntilNow)
        sumUntilNow=sumUntilNow+e/sumInvertedList
    return rouletteList

##spin rulet
##argument: ruletList
##output: index of a
def spinRoulette(rouletteList):
    
    randFloat=random.random()
    rouletteSize=len(rouletteList)
    for i in range(rouletteSize):
        if(randFloat<rouletteList[i]):
            break
    return i



if(sructure_of_neural_net=="5s"):
    
    ##get our starting population
    ## [10 "dendrit" weights, 6 free weights]
    if(len(trainingData[0])==2):
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 16)
            allPopulation.append(x)
            numOfWeights=16
    else:
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 21)
            allPopulation.append(x)
            numOfWeights=21

    for g in range(number_of_iterations_of_genAlgo+1):


        ##we get succes stats of one unit with a coresponding array
        deviationArray=[]
        for i in range(len(allPopulation)):
            allDeviationsOfunit =0
            for j in range(len(trainingData)):
                if(numOfWeights==16):
                    output=goThrough5sNeuronNet1(allPopulation[i],trainingData[j][0])
                else:
                    output=goThrough5sNeuronNet2(allPopulation[i],[trainingData[j][0],trainingData[j][1]])
                expectedOutput=trainingData[j][-1]
                difference=math.pow(output-expectedOutput,2)
                allDeviationsOfunit =allDeviationsOfunit +difference
            deviationArray.append(allDeviationsOfunit /len(trainingData))
        
        
        if(g%10 == 0 and g!=0):
            print("[Train error @"+str(g)+"]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")
        
        ##MAKE A NEW POPULATION
        ##!!!
        newPopulation=[]


        ##add elitsits
        smallestValues=getNSmallestValues(deviationArray, numOf_elitist)
        allIndexesOfElitists=getIdexesOfValues(deviationArray, smallestValues)
        for indexOfOneElitist in allIndexesOfElitists:
            newPopulation.append(allPopulation[indexOfOneElitist])
        
    
        ##parents selection
        roulette=getRoulette(deviationArray)
        for i in range(popsize-numOf_elitist):
            parentIndex1=spinRoulette(roulette)
            parentIndex2=spinRoulette(roulette)
            
            parent1=allPopulation[parentIndex1]
            parent2=allPopulation[parentIndex2]
            
            child=[]

            ##mutation
            for i in range(numOfWeights):
                randFloat=random.random()
                mutation=0
                if(randFloat<probability_of_mutation):
                    mutation=np.random.normal(0,standard_deviation_gaus_mutation,1)[0]
                    
                child.append(((parent1[i]+parent2[i]) / 2) + mutation)

            newPopulation.append(np.array(child))
        allPopulation=newPopulation

    ##testing
    deviationArray=[]
    for i in range(len(allPopulation)):
        allDeviationsOfunit =0
        for j in range(len(testingData)):
            if(numOfWeights==16):
                output=goThrough5sNeuronNet1(allPopulation[i],testingData[j][0])
            else:
                output=goThrough5sNeuronNet2(allPopulation[i],[testingData[j][0],testingData[j][1]])
            expectedOutput=testingData[j][-1]
            difference=math.pow(output-expectedOutput,2)
            allDeviationsOfunit =allDeviationsOfunit +difference
        deviationArray.append(allDeviationsOfunit /len(testingData))
    
    print("[Test error]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")
    
    

if(sructure_of_neural_net=="20s"):
    ##get our starting population
    ## [10 "dendrit" weights, 6 free weights]
    if(len(trainingData[0])==2):
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 61)
            allPopulation.append(x)
            numOfWeights=61
    else:
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 81)
            allPopulation.append(x)
            numOfWeights=81

    for g in range(number_of_iterations_of_genAlgo+1):

        ##we get succes stats of one unit with a coresponding array
        deviationArray=[]
        for i in range(len(allPopulation)):
            allDeviationsOfunit =0
            for j in range(len(trainingData)):
                if(numOfWeights==61):
                    output=goThrough20sNeuronNet1(allPopulation[i],trainingData[j][0])
                else:
                    output=goThrough20sNeuronNet2(allPopulation[i],[trainingData[j][0],trainingData[j][1]])
                expectedOutput=trainingData[j][-1]
                difference=math.pow(output-expectedOutput,2)
                allDeviationsOfunit =allDeviationsOfunit +difference
            deviationArray.append(allDeviationsOfunit /len(trainingData))
        
        if(g%10 == 0 and g!=0):
            print("[Train error @"+str(g)+"]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")

        ##CREATE NEW POPULATION
        ##!!!
        newPopulation=[]


        ##add elitsits
        smallestValues=getNSmallestValues(deviationArray, numOf_elitist)
        allIndexesOfElitists=getIdexesOfValues(deviationArray, smallestValues)
        for indexOfOneElitist in allIndexesOfElitists:
            newPopulation.append(allPopulation[indexOfOneElitist])
        
    
        ##parents selection
        roulette=getRoulette(deviationArray)
        for i in range(popsize-numOf_elitist):
            parentIndex1=spinRoulette(roulette)
            parentIndex2=spinRoulette(roulette)
            
            parent1=allPopulation[parentIndex1]
            parent2=allPopulation[parentIndex2]
            
            child=[]

            ##mutation 
            for i in range(numOfWeights):
                randFloat=random.random()
                mutation=0
                if(randFloat<probability_of_mutation):
                    mutation=np.random.normal(0,standard_deviation_gaus_mutation,1)[0]
                    
                child.append(((parent1[i]+parent2[i]) / 2) + mutation)

            newPopulation.append(np.array(child))
        allPopulation=newPopulation
    
    ##testing
    deviationArray=[]
    for i in range(len(allPopulation)):
        allDeviationsOfunit =0
        for j in range(len(testingData)):
            if(numOfWeights==61):
                output=goThrough20sNeuronNet1(allPopulation[i],testingData[j][0])
            else:
                output=goThrough20sNeuronNet2(allPopulation[i],[testingData[j][0],testingData[j][1]])
            expectedOutput=testingData[j][-1]
            difference=math.pow(output-expectedOutput,2)
            allDeviationsOfunit =allDeviationsOfunit +difference
        deviationArray.append(allDeviationsOfunit /len(testingData))
    print("[Test error]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")

if(sructure_of_neural_net=="5s5s"):
    ##get our starting population
    ## [10 "dendrit" weights, 6 free weights]
    if(len(trainingData[0])==2):
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 46)
            allPopulation.append(x)
            numOfWeights=46
    else:
        for i in range (int(popsize)):
        
            x = np.random.normal(0, 0.01, 51)
            allPopulation.append(x)
            numOfWeights=51
    
    for g in range(number_of_iterations_of_genAlgo+1):

        ##we get succes stats of one unit with a coresponding array
        deviationArray=[]
        for i in range(len(allPopulation)):
            allDeviationsOfunit =0
            for j in range(len(trainingData)):
                if(numOfWeights==46):
                    output=goThrough5s5sNeuronNet1(allPopulation[i],trainingData[j][0])
                else:
                    output=goThrough5s5sNeuronNet2(allPopulation[i],[trainingData[j][0],trainingData[j][1]])
                expectedOutput=trainingData[j][-1]
                difference=math.pow(output-expectedOutput,2)
                allDeviationsOfunit =allDeviationsOfunit +difference
            deviationArray.append(allDeviationsOfunit /len(trainingData))
        

        if(g%10 == 0 and g!=0):
            print("[Train error @"+str(g)+"]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")

        ##CREATE NEW POPULATION
        ##!!!
        newPopulation=[]


        ##add elitsits
        smallestValues=getNSmallestValues(deviationArray, numOf_elitist)
        allIndexesOfElitists=getIdexesOfValues(deviationArray, smallestValues)
        for indexOfOneElitist in allIndexesOfElitists:
            newPopulation.append(allPopulation[indexOfOneElitist])
        
    
        ##parents selection
        roulette=getRoulette(deviationArray)
        for i in range(popsize-numOf_elitist):
            parentIndex1=spinRoulette(roulette)
            parentIndex2=spinRoulette(roulette)
            
            parent1=allPopulation[parentIndex1]
            parent2=allPopulation[parentIndex2]
            
            child=[]

            ##mutation 
            for i in range(numOfWeights):
                randFloat=random.random()
                mutation=0
                if(randFloat<probability_of_mutation):
                    mutation=np.random.normal(0,standard_deviation_gaus_mutation,1)[0]
                    
                child.append(((parent1[i]+parent2[i]) / 2) + mutation)

            newPopulation.append(np.array(child))
        allPopulation=newPopulation
    
    deviationArray=[]
    for i in range(len(allPopulation)):
        allDeviationsOfunit =0
        for j in range(len(testingData)):
            if(numOfWeights==46):
                output=goThrough5s5sNeuronNet1(allPopulation[i],testingData[j][0])
            else:
                output=goThrough5s5sNeuronNet2(allPopulation[i],[testingData[j][0],testingData[j][1]])
            expectedOutput=testingData[j][-1]
            difference=math.pow(output-expectedOutput,2)
            allDeviationsOfunit =allDeviationsOfunit +difference
        deviationArray.append(allDeviationsOfunit /len(testingData))
    print("[Test error]: "+str(round(getNSmallestValues(deviationArray,1)[0],6)),end="\n")