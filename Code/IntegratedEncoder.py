import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense , Input , Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import random





"""

    createEncoder() -> The Function creates a basic encoder without decoder
        (Required)
            InputSpace      : Specify the input space
        (Optional)
            EncodeSpace : Specify the encoding Dimensio, default is 2   
            seed            : Specify the seed for random functions, default is 0
            OutputName      : Specify the name of the encoding layer
    return -> Enocoder Model
    
"""


def createEncoder(InputSpace : int , EncodeSpace : int = 2, seed : int = 2 , NthModel = 'EncoderOutputs') :   
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    InputName = 'Input' + str(NthModel)
    OutputName = 'Output' + str(NthModel)
    HiddenName = 'Hidden' + str(NthModel)

    InputLayer = Input(shape = InputSpace , name = InputName)

    x = Dense(InputSpace * 8 , activation = 'relu' , name = HiddenName)(InputLayer) 

    EncodeLayer = Dense(EncodeSpace , name = OutputName , activation = 'relu')(x)

    Encoder = Model(inputs = InputLayer , outputs = EncodeLayer)

    return Encoder


"""

    constructEnsemble() -> Constructs the ensemble of encoder models
        (Required)
            InputSpaces     : A Vector containing the input spaces for each encoder, need to be the same dimension as EncodeSpaces
            EncodeSpaces    : A vector containing the encode spaces for each encoder, needs to be the same dimension as InputSpaces
        (Optional)
            seed            : Specify the seed for random functions
    return -> An ensemble/list containing encoder models

"""


def constructEnsemble( InputSpaces , EncodeSpaces, seed = 0) :

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if len(InputSpaces) != len(EncodeSpaces) :
        raise Exception("The dimensions of InputSpaces and EncodeSpaces are not same , Given {0} and {1}".format(len(InputSpaces) , len(EncodeSpaces)))
    else :
        NEncoders = len(InputSpaces)

    Ensemble = []

    for i in range(NEncoders) : 
        Ensemble.append(createEncoder(InputSpace = InputSpaces[i] , EncodeSpace = EncodeSpaces[i] , NthModel = (i + 1)))

    return Ensemble



def constructLatentEncoder(InputSpaces , LatentSpaces , seed = 0) :

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if len(InputSpaces) != len(LatentSpaces) :
        raise Exception("The dimensions of InputSpaces and LatentSpaces are not same , Given {0} and {1}".format(len(InputSpaces) , len(LatentSpaces)))
    else :
        NEncoders = len(InputSpaces)


    Ensemble = constructEnsemble(InputSpaces = InputSpaces , EncodeSpaces = LatentSpaces , seed = seed)

    EnsembleInputs  = []
    EnsembleOutputs = []

    for encoder in Ensemble :
        EnsembleInputs.append(encoder.input)
        EnsembleOutputs.append(encoder.output)

    InputSpace = np.sum(InputSpaces)

    SplitLatent = Concatenate(name = 'LatentEncode')(EnsembleOutputs)
    
    Z = Dense(InputSpace * 8 , activation = 'relu' , name = 'HiddenEncode')(SplitLatent)
    
    DecodeLayer = Dense(InputSpace , activation = 'relu' , name = 'Decoded')(Z)

    IntegratedEncoderModel = Model(inputs = EnsembleInputs , outputs = DecodeLayer)

    return IntegratedEncoderModel



"""

    constructIntegratedEncoder() -> constructs an integrated encoder model
        (Required)
            InputSpaces     : A Vector containing the input spaces for each encoder, need to be the same dimension as LatentSpaces
            LatentSpaces    : A vector containing the encode spaces for each encoder, needs to be the same dimension as InputSpaces
        (Optional)
            EncodeSpace     : The final encoding space for the model, default is 2
            seed            : seed value for the random functions
        return -> Integrated Encoder model
"""


def constructIntegratedEncoder(InputSpaces , LatentSpaces  , EncodeSpace : int = 2 , seed = 0) :

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if len(InputSpaces) != len(LatentSpaces) :
        raise Exception("The dimensions of InputSpaces and LatentSpaces are not same , Given {0} and {1}".format(len(InputSpaces) , len(LatentSpaces)))
    else :
        NEncoders = len(InputSpaces)


    Ensemble = constructEnsemble(InputSpaces = InputSpaces , EncodeSpaces = LatentSpaces , seed = seed)

    EnsembleInputs  = []
    EnsembleOutputs = []

    for encoder in Ensemble :
        EnsembleInputs.append(encoder.input)
        EnsembleOutputs.append(encoder.output)

    InputSpace = np.sum(InputSpaces)

    SplitLatent = Concatenate(name = 'LatentEncode')(EnsembleOutputs)
    
    IntegratedEncode = Dense(EncodeSpace , name = 'FinalEncode')(SplitLatent)

    Hidden1 = Dense(InputSpace , activation = 'relu' , name = 'HhiddenEncode1')(IntegratedEncode) 
    
    Hidden2 = Dense(InputSpace * 8 , activation = 'relu' , name = 'HiddenEncode2')(Hidden1)
    
    DecodeLayer = Dense(InputSpace , activation = 'sigmoid' , name = 'Decoded')(Hidden2)

    IntegratedEncoderModel = Model(inputs = EnsembleInputs , outputs = DecodeLayer)

    return IntegratedEncoderModel



"""

    devideTrainingData() -> Support function for creating Integrated Encoder model. Creates subsets of DataSet according to given Feature Classes
        (Required)
            DataSet         : The working Dataset
            FeatureClasses  : A list containing the devisions of features
    
    
"""


def devideTrainingData(DataSet : pd.DataFrame , FeatureClasses) :
    
    EnsembleDataSet = []

    for i in FeatureClasses :
        EnsembleDataSet.append(DataSet[i])

    return EnsembleDataSet



"""

    formInputOutput() -> Support function for constructing the intergrated encoder models. Creates the input and encode dimensions for each latent encoder model.
        (Required)
            FeatureClasses : The list containing devisions of features.

"""


def formInputOutput(FeatureClasses) :

    InputSpaces  = []
    OutputSpaces = []

    for i in FeatureClasses :

        InputSpaces.append(len(i))

        OutputSpaces .append(np.ceil(len(i) / 7))

    return InputSpaces , OutputSpaces