import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine,load_digits
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense , PReLU
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import random
import umap.umap_ as umap
import math

"""

    createEncoder() -> The Function creates a basic encoder
        (Required)
            InputSpace       : Specify the input space
        (Optional)
            EncodeSpace : Specify the encoding Dimensio, default is 2   
            seed            : Specify the seed for random functions, default is 0
    return -> Enocoder Model
"""
['euclidean' , 'hamming' , 'canberra']

def createEncoder(metric , InputSpace : int , EncodeSpace : int = 2, seed : int = 2) :   
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    InputLayer = Input(shape = InputSpace)

    x1 = Dense(math.ceil(InputSpace * 0.75))(InputLayer) 
    x1 = PReLU(alpha_initializer = 'zeros')(x1)
    x2 = Dense(math.ceil(InputSpace * 0.25))(x1)
    x2 = PReLU(alpha_initializer = 'zeros')(x2) 


    if metric == 'euclidean' :
        actFunction = 'linear'
    elif metric == 'hamming' :
        actFunction = 'sigmoid'
    elif metric == 'canberra':
        actFunction = 'softmax'
        
    EncodeLayer = Dense(EncodeSpace , name = 'latent' , activation = actFunction)(x2)

    z1 = Dense(math.ceil(InputSpace * 0.25))(EncodeLayer)
    z1 = PReLU(alpha_initializer = 'zeros')(z1)
    z2 = Dense(math.ceil(InputSpace * 0.75))(z1)
    z2 = PReLU(alpha_initializer = 'zeros')(z2)

    DecodeLayer = Dense(InputSpace , name = 'decoded' , activation = actFunction)(z2)

    Encoder = Model(inputs = InputLayer , outputs = DecodeLayer)

    return Encoder




"""

    constructEnsemble() -> Creates Encoder Models and appends all them to obtain ensemble of encoders and latent space
        (Required) 
            DataSet         : The dataset the model needs to fit
            InputSpaces     : The vector containing the input spaces of all the encoder models, needs to be same dimension as EncodeSpaces and Classes
            EncodeSpaces    : The vector containing the encoding dimension of all the encoder models, need to be same dimension as InputSpaces and Classes
            FeatureClasses  : The list containing the feature devisions of the DataSet, needs to be same dimension as InputSpaces and EncodeSpaces
        (optional)
            SavePath        : Specify the path to save the encoder models
            Epochs          : Specify the epochs for the model to run, default is 20
            BatchSize       : Specify the batch size for the model, default is 5
            seed            : Specify the seed for random functions.
    return -> LatentSpace/Intermediate Space


"""


def constructEnsemble(Metric , InputSpaces , EncodeSpaces , seed : int = 0) :
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    if len(EncodeSpaces) != len(InputSpaces) :
        raise Exception("InputSpaces and EncodeSpaces must be of same dimensions , Given {0} and {1}".format(len(InputSpaces) , len(EncodeSpaces) )) 
    else :
        NClasses = len(EncodeSpaces)

    Ensemble = []

    for i in range(NClasses) :
        
        Ensemble.append(createEncoder(metric = Metric[i] , InputSpace = InputSpaces[i], EncodeSpace = EncodeSpaces[i]))


    return Ensemble


def devideTrainingData(DataSet : pd.DataFrame , FeatureClasses) :
    
    EnsembleDataSet = []

    for i in FeatureClasses :
        EnsembleDataSet.append(DataSet[i])

    return EnsembleDataSet


def TrainEnsemble(Ensemble , EnsembleDataset  , Loss , LearningRate , SavePath = None, Epochs = 30 , BatchSize = 10 ) :
    NClasses = len(EnsembleDataset)

    Encoders = []

    for i in range(NClasses) :

        print('\nTraining and predicting using model {} \n'.format(i + 1))

        Ensemble[i].compile(optimizer = Adam(learning_rate = LearningRate[i]) , loss = Loss[i])
        Ensemble[i].fit(EnsembleDataset[i] , EnsembleDataset[i] , epochs = Epochs , batch_size = BatchSize , shuffle = True)

        Encoder = Model(inputs = Ensemble[i].input , outputs = Ensemble[i].get_layer('latent').output)

        if SavePath != None :
            Encoder.save(SavePath + str(i) + '.keras')

        Encoders.append(Encoder)
    
    return Encoders


def PredictEnsemble(Encoders , EnsembleDataset , EncodeSpaces) :
        
    LatentSpace = []
    EncodeSpace = np.sum(EncodeSpaces)
    NClasses = len(EnsembleDataset)
    
    for i in range(NClasses) :
        LatentFeature = np.transpose(np.array(Encoders[i].predict(EnsembleDataset[i])))
        for i in LatentFeature :
            LatentSpace.append(i)

    #LatentSpace = np.array(LatentSpace).reshape(-1 , EncodeSpace)

    return LatentSpace



"""

    loadEnsemble -> Loads a prebuilt model stored in the system
        (Required)
            LoadPath        : The file path where the models are stored. Since there are more than one model to load only need to specify the folder where all the models are stored.
            DataSet         : The dataset the model needs to fit
            FeatureClasses  : The list containing the feature devisions of the DataSet
    return -> Latent Space


"""


def loadEnsemble(LoadPath : str , NClasses) : 
    
    Encoders = []
    
    for i in range(NClasses) :

        Encoders.append(keras.models.load_model(LoadPath + str(i) + '.keras'))

    return Encoders
"""




"""


def latentEncoder(LatentSpace , EncodeSpace : int = 2 , seed = 0) :
    
    InputNodes = LatentSpace.shape[1]

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    InputLayer = Input(shape = InputNodes)

    x1 = Dense(InputNodes * 8 , activation = 'relu')(InputLayer)

    EncodeLayer = Dense(EncodeSpace , name = 'result')(x1)

    z1 = Dense(InputNodes * 2 , activation = 'relu')(EncodeLayer)

    DecodeLayer = Dense(InputNodes)(z1)

    Encoder = Model(inputs = InputLayer , outputs = DecodeLayer)

    return Encoder
"""




"""


def formInputOutput(FeatureClasses) :

    InputSpaces  = []
    OutputSpaces = []

    for i in FeatureClasses :

        InputSpaces.append(len(i))

        OutputSpaces .append(np.ceil(len(i) / 7))

    return InputSpaces , OutputSpaces