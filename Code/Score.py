import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

"""

    calculateSignificantFeatures() -> Finds the significants features among the dataset with a label
        (Required)
            DataSet     : The DataSet where the function need to operate
        (Optional)
            Threshold   : Parameter that defines is a feature is signficant or not, default is 0.01
    return -> List of Significant Features

"""



def calculateSignificantFeatures(DataSet: pd.DataFrame , Labels : list , Threshold: float = 0.01):
    ClusterList = []

    for i in np.unique(Labels):
        ClusterList.append(DataSet[Labels == i])

    SignificantFeatures = []

    for feature in DataSet.columns:
        Distribution = [np.array(cluster[feature]) for cluster in ClusterList]
        SignificantResult = f_oneway(*Distribution)

        if SignificantResult[1] < Threshold:
            SignificantFeatures.append(feature)

    return SignificantFeatures

"""

    intersection() -> Finds the elements common to given lists 
        (Required)
            Vector1 : List 1
            Vecotr2 : List 2
    return -> resultant vector containing elements common to both vectors

"""

def intersection(Vector1 : list , Vector2 : list) :

    Resultant = [value for value in Vector1 if value in Vector2]

    return Resultant


"""

    calcualateSignificantPercentage() -> Finds the percentage of features that are significant in each feature classes.
        (Required)  
            SignificantFeatures : List of Significant Features
            FeatureClasses      : List of all Feature Devisions
    return -> List containing Significant Features of each Feature Class


"""


def calculateSignificantPercentage(SignificantFeatures: list, FeatureClasses):
    SignificantPercentage = []

    for i in FeatureClasses:
        ClassSignificant = set(SignificantFeatures) & set(i)
        SignificantPercentage.append((len(ClassSignificant) / len(i)) * 100)

    return SignificantPercentage

    

def DunnIndex(Dataset , Labels) :
    distances = pairwise_distances(Dataset)
    min_inter_cluster_distance = np.min([distances[i, j] for i in range(len(Dataset)) for j in range(len(Dataset)) if Labels[i] != Labels[j]])

    max_intra_cluster_distance = np.max([distances[i, j] for i in range(len(Dataset)) for j in range(len(Dataset)) if Labels[i] == Labels[j]])

    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index

def PValue(Features, SFeatures):
    return ((len(SFeatures) - 1) / (len(Features) - 1)) * 100

def TriScore(Dataset , Labels) :
    dunn_score = davies_bouldin_score(Dataset , Labels)
    print("Dunn Index\t\t: " , dunn_score)
    silhouette_avg = silhouette_score(Dataset, Labels)
    print("Silhouette Score\t: ", silhouette_avg)
    SFeatures = calculateSignificantFeatures(DataSet = Dataset , Labels = Labels , Threshold = 0.05)
    pScore = PValue(Dataset.columns , SFeatures)
    print("PScore\t\t\t: " , pScore)
    print("NClusters\t\t: " , len(np.unique(Labels)))
    print("\n\n")

def Visualize2D(Dataset2D , Labels) :
    plt.scatter(Dataset2D['UMAP1'] , Dataset2D['UMAP2'] , s = 2 , c = Labels , cmap = 'tab20') 
    plt.ylabel('Axis 2')
    plt.xlabel('Axis 1')
    plt.show()



def FindOptimalParameters(Dataset , ReducedSpace, scoreNDSpace = 0 , Algorithm = 'kmeans' , rangeX = 2 , rangeY = 11 , epsX = 0.1 , epsY = 2.0 , MinSamples = 5) :
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans, AgglomerativeClustering , DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
        

    KRange = range(rangeX, rangeY)
    epss = np.linspace(epsX, epsY, 20)

    sse = []  
    silhoetteScores = []
    daviesBouldinScores = []
    
    
    if not scoreNDSpace :
        if Algorithm == 'kmeans' :

            for k in KRange:

                kmeans = KMeans(n_clusters = k , random_state = 42)
                labels = kmeans.fit_predict(ReducedSpace)

                score = silhouette_score(Dataset, labels)
                score = silhouette_score(Dataset, labels)

                silhoetteScores.append(score)

                score = davies_bouldin_score(Dataset, labels)
                score = davies_bouldin_score(Dataset, labels)

                daviesBouldinScores.append(score)

                sse.append(kmeans.inertia_)

        if Algorithm == 'agglo' :

            sse = [0] * len(KRange)

            for k in KRange :

                labels = AgglomerativeClustering(n_clusters=k).fit_predict(ReducedSpace)

                score = silhouette_score(Dataset, labels)
                score = silhouette_score(Dataset, labels)

                silhoetteScores.append(score)

                score = davies_bouldin_score(Dataset, labels)
                score = davies_bouldin_score(Dataset, labels)

                daviesBouldinScores.append(score)

                

        if Algorithm == 'dbscan' :

            sse = [0] * len(epss)

            for epsilon in epss:
                
                labels = DBSCAN(eps = epsilon, min_samples = MinSamples).fit_predict(ReducedSpace)

                if len(np.unique(labels)) > 1:

                    score = silhouette_score(Dataset, labels)
                    score = silhouette_score(Dataset, labels)

                    silhoetteScores.append(score)

                    score = davies_bouldin_score(Dataset, labels)
                    score = davies_bouldin_score(Dataset, labels)

                    daviesBouldinScores.append(score)

                else:

                    silhoetteScores.append(np.nan)
                    
                    daviesBouldinScores.append(np.nan)

    else :

        if Algorithm == 'kmeans' :

            for k in KRange:

                kmeans = KMeans(n_clusters = k , random_state = 42)
                labels = kmeans.fit_predict(ReducedSpace)

                score = silhouette_score(ReducedSpace, labels)
                score = silhouette_score(ReducedSpace, labels)

                silhoetteScores.append(score)

                score = davies_bouldin_score(ReducedSpace, labels)
                score = davies_bouldin_score(ReducedSpace, labels)

                daviesBouldinScores.append(score)

                sse.append(kmeans.inertia_)

        if Algorithm == 'agglo' :

            sse = [0] * len(KRange)

            for k in KRange :

                labels = AgglomerativeClustering(n_clusters=k).fit_predict(ReducedSpace)

                score = silhouette_score(ReducedSpace, labels)
                score = silhouette_score(ReducedSpace, labels)

                silhoetteScores.append(score)

                score = davies_bouldin_score(ReducedSpace, labels)
                score = davies_bouldin_score(ReducedSpace, labels)

                daviesBouldinScores.append(score)

        if Algorithm == 'dbscan' :

            sse = [0] * len(epss)

            for epsilon in epss:
                
                labels = DBSCAN(eps = epsilon, min_samples = MinSamples).fit_predict(ReducedSpace)

                if len(np.unique(labels)) > 1:
                    
                    score = silhouette_score(ReducedSpace, labels)
                    score = silhouette_score(ReducedSpace, labels)
    
                    silhoetteScores.append(score)
    
                    score = davies_bouldin_score(ReducedSpace, labels)
                    score = davies_bouldin_score(ReducedSpace, labels)
    
                    daviesBouldinScores.append(score)

                else:

                    silhoetteScores.append(np.nan)
                    
                    daviesBouldinScores.append(np.nan)




    if Algorithm == 'kmeans' or Algorithm == 'agglo' :

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(KRange, sse, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')

        plt.subplot(1, 3, 2)
        plt.plot(KRange, silhoetteScores, marker='o')
        plt.title('Silhouette Score for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')

        plt.subplot(1, 3, 3)
        plt.plot(KRange, daviesBouldinScores, marker='o')
        plt.title('Davies-Bouldin Index for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Davies-Bouldin Index')

        plt.suptitle(Algorithm)
        plt.tight_layout()
        plt.show()

    elif Algorithm == 'dbscan' :

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epss, sse, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Epsilon')
        plt.ylabel('Sum of Squared Errors (SSE)')

        plt.subplot(1, 3, 2)
        plt.plot(epss, silhoetteScores, marker='o')
        plt.title('Silhouette Score for Optimal k')
        plt.xlabel('Epsilon')
        plt.ylabel('Silhouette Score')

        plt.subplot(1, 3, 3)
        plt.plot(epss, daviesBouldinScores, marker='o')
        plt.title('Davies-Bouldin Index for Optimal k')
        plt.xlabel('Epsilon')
        plt.ylabel('Davies-Bouldin Index')

        plt.suptitle(Algorithm)
        plt.tight_layout()
        plt.show()








import pandas as pd
from scipy import spatial


def Diameter(cluster):
   return spatial.distance.pdist(cluster).max()


import pandas as pd
from scipy import spatial

def dunnIndex(pts, labels, centroids = None):
 

 if centroids == None :
     centroids = compute_centroids(pts , labels)


 # Calculate cluster sizes

 cluster_sizes = pd.Series(labels).value_counts()

 # O(k n log(n)) with k clusters and n points; better performance with more even clusters

 max_intracluster_dist = pd.DataFrame(pts).groupby(labels).apply(Diameter).max()


 # O(k^2) with k clusters; can be reduced to O(k log(k))
 # get pairwise distances between centroids


 cluster_dmat = spatial.distance.cdist(centroids, centroids)
 # fill diagonal with +inf: ignore zero distance to self in "min" computation
 np.fill_diagonal(cluster_dmat, np.inf)
 min_intercluster_dist = cluster_sizes.min()
 return min_intercluster_dist / max_intracluster_dist

    
def compute_centroids(points, labels):
   # Initialize an empty dictionary to hold the centroids
   centroids = {}

   # Iterate over the unique labels
   for label in set(labels):
       # Get all points belonging to the current label
       cluster_points = points[labels == label]
       
       # Compute the centroid of the cluster
       centroid = np.mean(cluster_points, axis=0)
       
       # Add the centroid to the dictionary
       centroids[label] = centroid

   return centroids