from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
FILE_PATH = 'http://www.platform.ai/models/'

def createVGGModelConvolutionBlock(): 
    model = Sequential()
    model.add(Lambda(preprocess, input_shape=(3,224,224), output_shape=(3,224,224), trainable=False))
    addConvolutionBlock(model, 2, 64)
    addConvolutionBlock(model, 2, 128)
    addConvolutionBlock(model, 3, 256)
    addConvolutionBlock(model, 3, 512)
    addConvolutionBlock(model, 3, 512)
    fname = 'vgg16_bn_conv.h5'
    model.load_weights(get_file(fname, FILE_PATH+fname, cache_subdir='models'))
    return model

def preprocess(x): 
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
    newArr = x-vgg_mean
    return newArr[:, ::-1]

def addConvolutionBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(filters,3,3,activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
def markAllLayersAsTrainableFalse(model):
    for layer in model.layers: layer.trainable=False
        
def addNewFCLayers(model):
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.))
    model.add(Dense(8, activation='softmax'))

def compileModel(model, lr):
    model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])

def createVGGModelAllBlock(): 
    model = Sequential()
    model.add(Lambda(preprocess, input_shape=(3,224,224), output_shape=(3,224,224), trainable=False))
    addConvolutionBlock(model, 2, 64)
    addConvolutionBlock(model, 2, 128)
    addConvolutionBlock(model, 3, 256)
    addConvolutionBlock(model, 3, 512)
    addConvolutionBlock(model, 3, 512)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    fname = 'vgg16.h5'
    model.load_weights(get_file(fname, FILE_PATH+fname, cache_subdir='models'))
    return model

def modifyVGGAllModelFor8Output(model):
    model.pop()
    for layer in model.layers: 
        layer.trainable=False
    model.add(Dense(8, activation='softmax'))
    
def modifyVGGAllModelFor6OutputWithTrainable(model):
    model.pop()
    for layer in model.layers: 
        layer.trainable=False
    model.add(Dense(6, activation='softmax'))
    
def getLastConvolutionLayerIndex(model):
    lastIndex = 0;
    for index, layer in enumerate(model.layers[::-1]):
        if type(layer) is Convolution2D:
            lastIndex = index
            break;
            
    return lastIndex;

def getFCModelWith0Dropouts(shape):
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), input_shape=shape))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.))
    model.add(Dense(8, activation='softmax'))
    return model

def copyHalvedWeightsFRomLayers1toLayers2(sourceLayers, targetLayers):
    for sourceLayer,targetLayer in zip(sourceLayers, targetLayers):
        targetLayer.set_weights(getHalveWeightOfLayer(sourceLayer))
        break

def getHalveWeightOfLayer(sourceLayer):
    weights = []
    print(sourceLayer.get_weights())
    for weight in sourceLayer.get_weights():
        weights.append(weight/2)
    print(weights)
    return weights;

def markLayersAsTrainable(layers, boolVal):
    for layer in layers:
        layer.trainable = boolVal
        
def getConvModelFromFullVGGModel(vggModel):
    indexOfLastConvolutedLayer = len(vggModel.layers)-(getLastConvolutionLayerIndex(vggModel) +1)
    convolutedLayers = vggModel.layers[:indexOfLastConvolutedLayer+1]
    convolutedModel = Sequential(convolutedLayers)
    return convolutedModel

def getFCModelFromFullVGGModel(vggModel, numberOfDense):
    indexOfLastConvolutedLayer = len(vggModel.layers)-(getLastConvolutionLayerIndex(vggModel) +1)
    denseLayers = vggModel.layers[indexOfLastConvolutedLayer+1:]
    outputShapeFromConvModel = vggModel.layers[indexOfLastConvolutedLayer].output_shape
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), input_shape=outputShapeFromConvModel[1:]))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfDense, activation='softmax'))
    sourceLayers = denseLayers
    targetLayers = model.layers
    for sourceLayer,targetLayer in zip(sourceLayers, targetLayers):
        targetLayer.set_weights(sourceLayer.get_weights())
        break
    return model

def setFCLayersToTrainable(vggModel):
    indexOfLastConvolutedLayer = len(vggModel.layers)-(getLastConvolutionLayerIndex(vggModel) +1)
    denseLayers = vggModel.layers[indexOfLastConvolutedLayer+1:]
    for idx, layer in  enumerate(denseLayers):
        layer.trainable = True
    return indexOfLastConvolutedLayer
       
def modifyAnyModelFor8Output(model, shouldBeTrainable):
    model.pop()
    for layer in model.layers: 
        layer.trainable=shouldBeTrainable
    model.add(Dense(8, activation='softmax'))