# Measure of central tendency: mean, median and mode

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
# from tensorflow import keras
# from datetime import datetime

dataset = pd.read_csv("Dataset 2/diabetes.csv")

imp = SimpleImputer(missing_values='?', strategy='most_frequent')
dataset = imp.fit_transform(dataset)

dataset = pd.DataFrame(dataset).astype('float')

scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

dataset = pd.DataFrame(dataset)

X = dataset[[0,1,2,3,4,5,6,7]]
Y = dataset[[8]]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)

clf = SVC(gamma='auto')
svm_model = clf.fit(X_train,y_train)
y_predict = clf.predict(X_test).astype('int32')
y_score = svm_model.decision_function(X_test)
y_test = np.array(y_test).astype('int32').ravel()

def cal_acc(y_predict,y_test):
    total_num = len(y_test)

    # true positive, true negative, false positive, false negative
    # 0 is majority, 1 is minority
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(total_num):
        if y_test[i] == 1:
            if y_predict[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_predict[i] == 1:
                FP += 1
            else:
                TN += 1

    # print("TP:",TP)
    # print("TN:",TN)
    # print("FP:",FP)
    # print("FN:",FN)

    accuracy = (TP+TN) / (TP+TN+FP+FN)
    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)

    # print("accuracy",accuracy)
    # print("sensitivity",sensitivity)
    # print("specificity",specificity)

    return TP,TN,FP,FN,accuracy,sensitivity,specificity

def plot_roc(model_name,y_score,y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    roc_auc = roc_auc_score(y_test, y_score)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label=model_name + ' ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # plt.savefig("roc_curve.png")
    plt.show()

plot_roc('SVM',y_score,y_test)

# logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# single layer
# model_single = keras.models.Sequential()

# model_single.add(keras.layers.Dense(units=1,input_dim=8,use_bias=True,activation='sigmoid'))

# sgd_optimizer = keras.optimizers.SGD(lr=0.1)

# model_single.compile(optimizer=sgd_optimizer,loss='sparse_categorical_crossentropy')

# history = model_single.fit(X_train,y_train,epochs=50)

# y_predict = model_single.predict_classes(X_test)

# print(y_predict)

# multi layer

# model_multiple = keras.models.Sequential()

# model_multiple.add(keras.layers.Dense(units=5,input_dim=8,use_bias=True,activation='sigmoid'))
# model_multiple.add(keras.layers.Dense(units=5,input_dim=5,use_bias=True,activation='sigmoid'))
# model_multiple.add(keras.layers.Dense(units=1,input_dim=5,use_bias=True,activation='sigmoid'))

# sgd_optimizer = keras.optimizers.SGD(lr=0.1)

# model_multiple.compile(optimizer=sgd_optimizer,loss='sparse_categorical_crossentropy')

# history = model_multiple.fit(X_train,y_train,epochs=50,callbacks=[tensorboard_callback])

# y_predict = model_multiple.predict_classes(X_test)

# print(y_predict)

