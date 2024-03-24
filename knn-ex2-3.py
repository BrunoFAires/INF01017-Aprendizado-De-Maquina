import pandas as pd
import math

df = pd.read_csv('Dados_Originais_2Features/TrainingData_2F_Original.txt', delimiter='\t')
dd = pd.read_csv('Dados_Originais_2Features/TestingData_2F_Original.txt', delimiter='\t')

trainingSets = df.to_dict(orient='records')
testSets =  dd.to_dict(orient='records')
testDistances = []
knn = 5


def euclidianDistance(firstInstance, secondInstance):
    sum = 0
    for key in firstInstance.keys():
        if key != "ID":
            sum += pow(firstInstance[key] - secondInstance[key], 2)
    return math.sqrt(sum)


for testSet in testSets:
    distances = []
    for trainingSet in trainingSets:
        distance = euclidianDistance(testSet, trainingSet)
        distances.append({"id": trainingSet["ID"], "distance": distance,"class": trainingSet["class"]})
    sortedDistances = sorted(distances, key=lambda x: x['distance'])
    testDistances.append({"set": testSet["ID"], "distances": sortedDistances})
    
for i in range(knn):
    print(testDistances[0]["distances"][i])
            