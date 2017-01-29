from keras.preprocessing import image
import bcolz as bcolz

def fileBatchGeneratorWithAugmentedData(path, batchSize):
    gen=image.ImageDataGenerator(rotation_range=90, zoom_range=0.1, 
       vertical_flip=True,  horizontal_flip=True,  width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2)
    return gen.flow_from_directory(path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=batchSize)

def fileBatchGeneratorWithAugmentedData_1(path, batchSize):
    gen=image.ImageDataGenerator(rotation_range=180, zoom_range=0.5, 
       vertical_flip=True,  horizontal_flip=True)
    return gen.flow_from_directory(path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=batchSize)

def fileBatchGeneratorWithAugmentedData_2(path, batchSize):
    gen=image.ImageDataGenerator(rotation_range=180, 
       vertical_flip=True,  horizontal_flip=True)
    return gen.flow_from_directory(path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=batchSize)

def fileBatchGenerator(path, batchSize):
    gen=image.ImageDataGenerator()
    return gen.flow_from_directory(path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=batchSize)

def fileBatchGeneratorHD(path, batchSize):
    gen=image.ImageDataGenerator()
    return gen.flow_from_directory(path, target_size=(720,1280),
                class_mode='categorical', shuffle=False, batch_size=batchSize)


def saveArray(fileName, arr):
     c=bcolz.carray(arr, rootdir=fileName, mode='w')
     c.flush()
    
def loadArray(fname):
    return bcolz.open(fname)[:]
        