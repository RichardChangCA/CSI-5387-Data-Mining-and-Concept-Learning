import numpy as np
from sklearn.cluster import DBSCAN

similarity_matrix =  [1.00, 0.10, 0.41, 0.55, 0.35,
                     0.10, 1.00, 0.64, 0.47, 0.98,
                     0.41, 0.64, 1.00, 0.44, 0.85,
                     0.55, 0.47, 0.44, 1.00, 0.76,
                     0.35, 0.98, 0.85, 0.76, 1.00]
similarity_matrix = np.array(similarity_matrix)
similarity_matrix = np.reshape(similarity_matrix,[5,5],order='C')

clustering = DBSCAN(eps=0.5, min_samples=2).fit(similarity_matrix)

f = open("Part_B_3_result.txt",'w+')
f.write(str(clustering.labels_)+'\n')
f.write(str(clustering.core_sample_indices_)+'\n')
f.write("others are noise points\n")
f.write("no border point\n")
f.write("Border points are points that are (in DBSCAN) part of a cluster, but not dense themselves (i.e. every cluster member that is not a core point).\n")
f.close()

# print(clustering.labels_)
# print(clustering.core_sample_indices_)
# others are noise points. & no border point
# Border points are points that are (in DBSCAN) part of a cluster, but not dense themselves (i.e. every cluster member that is not a core point).