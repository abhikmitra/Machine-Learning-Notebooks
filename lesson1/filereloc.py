from os import makedirs
from os import listdir
from os.path import isfile, join
from shutil import rmtree
from shutil import copy2
import numpy as np
kaggleFolder = "kaggle"
trainFolder = kaggleFolder + "/train"
testFolder = kaggleFolder + "/test"
outputFolder = kaggleFolder + "/output/"
def start():
    primes = [2, 3, 5, 7]
    trainingData = listdir(trainFolder)
    trainingDataSegregated = {}
    for fileName in trainingData:
        positionOfDot = fileName.find(".")
        category = fileName[:positionOfDot]
        if not category in trainingDataSegregated.keys():
            trainingDataSegregated[category] = []
        trainingDataSegregated[category].insert(0, fileName)
    print(trainingDataSegregated.keys())
    create_directories(trainingDataSegregated.keys())
    generateValidFolder(trainingDataSegregated)
    generateValidFolderForSample(trainingDataSegregated)
    generateTrainFolder(trainingDataSegregated)
    generateTrainFolderForSample(trainingDataSegregated)

def create_directories(categories):
    rmtree(outputFolder, True)
    for category in categories:
        makedirs(outputFolder + 'sample/train/'+category+'/')
        makedirs(outputFolder + 'sample/valid/'+category+'/')
        makedirs(outputFolder + 'train/'+category+'/')
        makedirs(outputFolder + 'valid/'+category+'/')

def generateValidFolder(trainingDataSegregated):
    percent = 10
    categories = trainingDataSegregated.keys()
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = trainingDataSegregated[category]
        size = (percent*len(fileNameArray))/100;
        print(size)
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size)))
        for fileName in filesToBeMovedToValid: 
            copy2(trainFolder+"/"+fileName, outputFolder+"valid/" + category + "/" +fileName)

def generateValidFolderForSample(trainingDataSegregated):
    percent = 2
    categories = trainingDataSegregated.keys()
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = trainingDataSegregated[category]
        size = (percent*len(fileNameArray))/100;
        print(size)
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size)))
        for fileName in filesToBeMovedToValid: 
            copy2(trainFolder+"/"+fileName, outputFolder+"/sample/valid/" + category + "/" +fileName)

def generateTrainFolder(trainingDataSegregated):
    categories = trainingDataSegregated.keys()
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = trainingDataSegregated[category]
        filesToBeMovedToValid.extend(fileNameArray)
        for fileName in filesToBeMovedToValid: 
            copy2(trainFolder+"/"+fileName, outputFolder+"train/" + category + "/" +fileName)

def generateTrainFolderForSample(trainingDataSegregated):
    percent = 10
    categories = trainingDataSegregated.keys()
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = trainingDataSegregated[category]
        size = (percent*len(fileNameArray))/100;
        print(size)
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size)))
        for fileName in filesToBeMovedToValid: 
            copy2(trainFolder+"/"+fileName, outputFolder+"sample/train/" + category + "/" +fileName)
start()
