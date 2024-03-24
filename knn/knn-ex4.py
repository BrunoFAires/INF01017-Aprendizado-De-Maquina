import pandas as pd
import math

df = pd.read_csv('Dados_Normalizados_2Features/TrainingData_2F_Norm.txt', delimiter='\t')
dd = pd.read_csv('Dados_Normalizados_2Features/TestingData_2F_Norm.txt', delimiter='\t')

trainingSets = df.to_dict(orient='records')
testSets =  dd.to_dict(orient='records')
testDistances = []
knn = 5


def euclidianDistance(firstInstance, secondInstance, actualInstanceIndex):
    sum = 0
    for key in firstInstance.keys():
        if(actualInstanceIndex == 3 and key ==  "citric.acid"):
            sum += pow(0.85 - secondInstance[key], 2)
        
        elif key != "ID":
            sum += pow(firstInstance[key] - secondInstance[key], 2)
    return math.sqrt(sum)

actualInstance = 0
for testSet in testSets:
    distances = []
    for trainingSet in trainingSets:
        distance = euclidianDistance(testSet, trainingSet, actualInstance)
        distances.append({"id": trainingSet["ID"], "distance": distance,"class": trainingSet["Class"]})
        actualInstance += 1
    sortedDistances = sorted(distances, key=lambda x: x['distance'])
    testDistances.append({"set": testSet["ID"], "distances": sortedDistances})

for testDistance in testDistances:
    class0 = 0
    class1 = 0
    for i in range(knn):
        if(testDistance["distances"][i]["class"] == 1):
            class1+=1
        else:
            class0+=1
    if class0 > class1:
        print(testDistance["set"] + " class 0")
    else:
        print(testDistance["set"] + " class 1")
            