import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
similarity_matrix =  [1.00, 0.10, 0.41, 0.55, 0.35,
                     0.10, 1.00, 0.64, 0.47, 0.98,
                     0.41, 0.64, 1.00, 0.44, 0.85,
                     0.55, 0.47, 0.44, 1.00, 0.76,
                     0.35, 0.98, 0.85, 0.76, 1.00]
similarity_matrix = np.array(similarity_matrix)

f = open("Part_B_2_results.txt",'w+')
# print(similarity_matrix)

similarity_matrix = np.reshape(similarity_matrix,[5,5],order='C')
# print(similarity_matrix)

clustering = AgglomerativeClustering(affinity='euclidean', n_clusters=2, linkage='complete').fit(similarity_matrix)
# print(clustering.labels_)
f.write(str(clustering.labels_)+'\n')

clustering = AgglomerativeClustering(affinity='euclidean', n_clusters=3, linkage='complete').fit(similarity_matrix)
# print(clustering.labels_)
f.write(str(clustering.labels_)+'\n')

clustering = AgglomerativeClustering(affinity='euclidean', n_clusters=4, linkage='complete').fit(similarity_matrix)
# print(clustering.labels_)
f.write(str(clustering.labels_)+'\n')

clustering = AgglomerativeClustering(affinity='euclidean', n_clusters=5, linkage='complete').fit(similarity_matrix)
# print(clustering.labels_)
f.write(str(clustering.labels_)+'\n')

f.close()

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(similarity_matrix, method='single', metric='euclidean'))
# plt.show()
plt.savefig("Dendrograms_single.png")

