from os import makedirs
from os import listdir
from os.path import isfile, join
from shutil import rmtree
from shutil import copy2
import numpy as np
import shutil as shutil
kaggleFolder = "../data/kaggle_state_farm"
trainFolder = kaggleFolder + "/train"
testFolder = kaggleFolder + "/test"
outputFolder = kaggleFolder + "/output/"
def start():
    primes = [2, 3, 5, 7]
    categories = listdir(trainFolder)
    #categories = categories[1:]
    create_directories(categories)
    generateValidFolder(categories)
    generateValidFolderForSample(categories)
    generateTrainFolderForSample(categories)
    # generateTrainFolderForSample(trainingDataSegregated)

def create_directories(categories):
    rmtree(outputFolder, True)
    for category in categories:
     makedirs(outputFolder + 'sample/train/'+category+'/')
     makedirs(outputFolder + 'sample/valid/'+category+'/')
     makedirs(outputFolder + 'train/'+category+'/')
     makedirs(outputFolder + 'valid/'+category+'/')

def generateValidFolderForSample(categories):
    print("generating valid folder for sample")
    percent = 2
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = listdir(trainFolder+"/"+category)
        size = (percent*len(fileNameArray))/100;
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size), replace=False))
        for fileName in filesToBeMovedToValid: 
            shutil.move(trainFolder+"/" +category+ "/"+fileName, outputFolder+"/sample/valid/" + category + "/" +fileName)

def generateValidFolder(categories):
    print("generating valid folder")
    percent = 20
    for category in categories:
        print("current category", category);
        filesToBeMovedToValid = [];
        fileNameArray = listdir(trainFolder+"/"+category+"/")
        size = (percent*len(fileNameArray))/100;
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size), replace=False))
        for fileName in filesToBeMovedToValid: 
            shutil.move(trainFolder+"/" +category+ "/"+fileName, outputFolder+"/valid/" + category + "/" +fileName)


def generateTrainFolderForSample(categories):
    print("generating train folder for sample")
    percent = 20
    for category in categories:
        filesToBeMovedToValid = [];
        fileNameArray = listdir(trainFolder+"/"+category)
        size = (percent*len(fileNameArray))/100;
        filesToBeMovedToValid.extend(np.random.choice(fileNameArray, int(size), replace=False))
        for fileName in filesToBeMovedToValid: 
            shutil.move(trainFolder+"/" +category+ "/"+fileName, outputFolder+"/sample/train/" + category + "/" +fileName)

start()
