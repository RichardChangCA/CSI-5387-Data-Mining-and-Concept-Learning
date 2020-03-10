import numpy as np
from sklearn.cluster import KMeans
x_1 = [1,0]
x_2 = [0,1]
x_3 = [2,1]
x_4 = [3,3]

C_1 = [x_1, x_3]
C_2 = [x_2, x_4]
data = [x_1,x_2,x_3,x_4]
# print(C_1)
def get_mean(C):
    n = 0
    dim_0 = 0
    dim_1 = 0
    for item in C:
        dim_0 += item[0]
        dim_1 += item[1]
        n += 1
    return [dim_0/n, dim_1/n]

# C_1_mean = get_mean(C_1)
# C_2_mean = get_mean(C_2)
# print(C_1_mean)
# print(C_2_mean)
def square_error(C):
    C_mean = get_mean(C)
    error = 0
    for item in C:
        error += (item[0]-C_mean[0])**2
        error += (item[1]-C_mean[1])**2
    return error

sum_squared_error = square_error(C_1) + square_error(C_2)
print("sum_squared_error",sum_squared_error)

def my_kmeans(iteration):
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=iteration).fit(data)
    # print(kmeans.labels_)
    C_1 = []
    C_2 = []
    labels = kmeans.labels_
    for i in range(len(labels)):
        if(labels[i]==0):
            C_1.append(data[i])
        else:
            C_2.append(data[i])

    sum_squared_error = square_error(C_1) + square_error(C_2)
    # print("sum_squared_error",sum_squared_error)
    return C_1, C_2, sum_squared_error

f = open("Part_B_1_results.txt",'w+')
f.write("initialization:\n")
f.write("C_1:"+str(C_1)+"\n")
f.write("C_2:"+str(C_2)+"\n")
f.write("sum_squared_error:"+str(sum_squared_error)+"\n")
C_1, C_2, sum_squared_error = my_kmeans(1)
f.write("kmeans iteration 1:\n")
f.write("C_1:"+str(C_1)+"\n")
f.write("C_2:"+str(C_2)+"\n")
f.write("sum_squared_error:"+str(sum_squared_error)+"\n")
C_1, C_2, sum_squared_error = my_kmeans(2)
f.write("kmeans iteration 2:\n")
f.write("C_1:"+str(C_1)+"\n")
f.write("C_2:"+str(C_2)+"\n")
f.write("sum_squared_error:"+str(sum_squared_error)+"\n")
f.close()
