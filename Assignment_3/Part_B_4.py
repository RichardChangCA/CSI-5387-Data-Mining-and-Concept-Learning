import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

df = pd.read_excel('Dataset 3/Absenteeism_at_work.xls', sheet_name='Absenteeism_at_work')
# print(df.columns)
data = df[["Age","Work load Average/day "]]
# print(data)

def plot_elbow():
    distortions = []
    K = range(1,15)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    ### inertia_: Sum of squared distances of samples to their closest cluster center.

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    plt.savefig("elbow.png")

def data_scatter_plot(data):
    # attribute scaler
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)

    kmeanModel = KMeans(n_clusters=11)
    prediction = kmeanModel.fit_predict(data)
    # print(prediction)

    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], c=prediction)
    # plt.show()
    plt.savefig("data_scatter_plot.png")


def my_AgglomerativeClustering(data):
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)

    prediction = AgglomerativeClustering(affinity='euclidean', n_clusters=11, linkage='single').fit_predict(data)
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], c=prediction)
    # plt.show()
    plt.savefig("AgglomerativeClustering_plot.png")

def my_GaussianMixture(data):
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    prediction = GaussianMixture(n_components=11).fit_predict(data)
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], c=prediction)
    # plt.show()
    plt.savefig("EM_GaussianMixture_plot.png")

plot_elbow()
#### k == 11 is determined from elbow plot
data_scatter_plot(data)
my_AgglomerativeClustering(data)
my_GaussianMixture(data)
