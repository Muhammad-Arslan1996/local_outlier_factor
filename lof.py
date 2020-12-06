import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns

#some part of k_neareast_neighbour functtion was implemented using ideas from
#https://stackoverflow.com/questions/57107729/how-to-compute-multiple-euclidean-distances-of-all-points-in-a-dataset
def k_neareast_neighbour(df, k):
    data = [(float(x),float(y)) for x, y in df[['X1', 'X2']].values ]
    nearest_neighbour = []
    nn_distances = []
    knn_distance = []
    for point in data:
        distances = [math.sqrt((point[0]-x[0] )**2+ (point[1]-x[1])**2) for x in data]
        knn = np.argsort(distances)[1:k+1]
        nearest_neighbour.append([i for i in knn])
        nn_distances.append([distances[i] for i in knn])
        knn_distance.append(distances[knn[-1]])
    return nearest_neighbour, nn_distances, knn_distance

def reachability_distance(nearest_neighbour_index, nearest_neighbour_distance, knn_distance):
    rd = 0
    for nni, nnd in zip(nearest_neighbour_index, nearest_neighbour_distance):
        rd = rd + max(knn_distance[nni], nnd)
    return rd
    
def local_outlier_factor(df, k, threshold):
    knn = k_neareast_neighbour(df, k)
    nearest_neighbours_indexes = knn[0]
    nearest_neighbour_distances = knn[1]
    knn_distance = knn[2]
    lrd = []
        
    lrd = np.array([reachability_distance(x, y, knn_distance) 
          for x,y in zip(nearest_neighbours_indexes, nearest_neighbour_distances)]) / k
    
    df['lrd'] = lrd
    lof = []
    for idx, nni in enumerate(nearest_neighbours_indexes):
        lof.append(np.mean([lrd[idx]/lrd[x] for x in nni]))
    df['lof'] = lof
    df_outlier = df[df['lof'] > threshold]

    sns.set(rc={'figure.figsize':(15,10)})
    sns_plot = sns.scatterplot(x="X1", y="X2", data=df, label='inliner')
    sns_plot = sns.scatterplot(x="X1", y="X2", data=df_outlier, s=100, color='r', label='outlier')
    sns_plot.legend(fontsize=20)
    fig = sns_plot.get_figure()
    fig.savefig('lof.png') 
    return df
    
def main():
    if len(sys.argv) != 2:
        print('missing k-value')
        print('run with:\n$ python lof.py <k-value>')
        exit(1)
    df = pd.read_csv('./outliers-3.csv')
    k = int(sys.argv[1])
    threshold = 2
    df_with_lof = local_outlier_factor(df, k, threshold)

if __name__ == "__main__":
    main()