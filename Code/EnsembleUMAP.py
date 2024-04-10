import umap.umap_ as umap
import numpy as np
import seaborn as sns
import pandas as pd



def UMAP(DataSet , Neibhours : int , MinDistance : float , Metric : str , EncodeSpace : int = 2 , seed = 0) :

    np.random.seed(seed)

    Embedding = umap.UMAP(n_neighbors = Neibhours , min_dist = MinDistance , n_components = EncodeSpace , metric = Metric , random_state = seed).fit_transform(DataSet)

    return Embedding


def EnsembleUMAP(DataSet , FeatureClasses , Metrics, LatentSpaces , Neibhours = 30 , MinDistance = 0.1 , seed = 0) :
    
    np.random.seed(seed)

    if len(FeatureClasses) != len(Metrics) != len(LatentSpaces):
        raise Exception(" No ")

    NClasses = len(FeatureClasses)
    EncodeSum = np.sum(LatentSpaces)
    
    LatentFeatures = []
    
    for i in range(NClasses) :
        print("Training Model\t:" , i + 1)
        ClassSet = DataSet[FeatureClasses[i]]
        
        LatentEmbedding = UMAP(DataSet = ClassSet , EncodeSpace = LatentSpaces[i] , Neibhours = Neibhours , MinDistance = MinDistance , Metric = Metrics[i])
        LatentEmbedding = np.transpose(LatentEmbedding)
        for i in LatentEmbedding :
            LatentFeatures.append(list(i))
        print(LatentFeatures)
    
    LatentFeatures = pd.DataFrame(np.array(LatentFeatures).transpose())
    return LatentFeatures


def DevideSpace(FeatureClasses , Threshold = 5) :

    InputSpaces  = []
    OutputSpaces = []

    for i in FeatureClasses :

        InputSpaces.append(len(i))

        OutputSpaces .append(np.ceil(len(i) / Threshold))

    return InputSpaces , OutputSpaces