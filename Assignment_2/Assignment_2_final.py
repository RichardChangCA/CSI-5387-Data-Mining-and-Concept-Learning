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
from tensorflow import keras
from datetime import datetime

def svm(X_train,y_train,X_test):
    clf = SVC(gamma='auto')
    svm_model = clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test).astype('int32')
    y_score = svm_model.decision_function(X_test)
    return y_predict,y_score

def cal_acc(model_name,y_predict,y_test):
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

    # f = open('results/'+model_name+".txt","w+")

    # f.write("TP:"+str(TP)+'\n')
    # f.write("TN:"+str(TN)+'\n')
    # f.write("FP:"+str(FP)+'\n')
    # f.write("FN:"+str(FN)+'\n')

    accuracy = (TP+TN) / (TP+TN+FP+FN)
    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)

    # f.write("accuracy:"+str(accuracy)+'\n')
    # f.write("sensitivity:"+str(sensitivity)+'\n')
    # f.write("specificity:"+str(specificity)+'\n')

    # f.close()

    return TP,TN,FP,FN,accuracy,sensitivity,specificity

def plot_roc(color,model_name,y_score,y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    roc_auc = roc_auc_score(y_test, y_score)

    plt.plot(fpr, tpr, color=color,
            lw=2, label=model_name + ' ROC curve (area = %0.2f)' % roc_auc)

# single layer
def single_layer(X_train,y_train,X_test):
    # logdir = 'logs/single_layer'
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model_single = keras.models.Sequential()

    model_single.add(keras.layers.Dense(units=1,input_dim=8,use_bias=True,activation='sigmoid'))

    sgd_optimizer = keras.optimizers.SGD(lr=0.01)

    model_single.compile(optimizer=sgd_optimizer,loss='binary_crossentropy')

    # history = model_single.fit(X_train,y_train,epochs=100,callbacks=[tensorboard_callback])
    history = model_single.fit(X_train,y_train,epochs=100)

    model_single.save("models/single_layer_model.h5")

    y_predict = model_single.predict_classes(X_test)

    y_score = model_single.predict_proba(X_test)

    # with open('results/single_layer_model.txt','w') as fh:
    # # Pass the file handle in as a lambda function to make it callable
    #     model_single.summary(print_fn=lambda x: fh.write(x + '\n'))

    # keras.utils.plot_model(model_single, to_file='results/single_layer_model.png',show_shapes=True,expand_nested=True)

    return y_predict.ravel(),y_score.ravel()


def single_layer_prediction(model,x,f_prediction,y_actual):
    x=np.reshape(np.array(x),(-1,8))
    print(x)
    y_prediction = model.predict_classes(x)
    f_prediction.write("X="+str(x)+", prediction="+str(y_prediction)+", actual="+y_actual+"\n\n")

# multi layer

def multilayer(model_name,X_train,y_train,X_test,use_bias,activation,learning_rate,epoch):

    # logdir = 'logs/'+str(use_bias)+"_"+activation+"_"+str(learning_rate)+"_"+str(epoch)
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model_multiple = keras.models.Sequential()

    model_multiple.add(keras.layers.Dense(units=5,input_dim=8,use_bias=use_bias,activation=activation))
    model_multiple.add(keras.layers.Dense(units=5,input_dim=5,use_bias=use_bias,activation=activation))
    model_multiple.add(keras.layers.Dense(units=1,input_dim=5,use_bias=use_bias,activation='sigmoid'))

    sgd_optimizer = keras.optimizers.SGD(lr=learning_rate)

    model_multiple.compile(optimizer=sgd_optimizer,loss='binary_crossentropy')

    # history = model_multiple.fit(X_train,y_train,epochs=epoch,callbacks=[tensorboard_callback])
    history = model_multiple.fit(X_train,y_train,epochs=epoch)

    y_predict = model_multiple.predict_classes(X_test)

    y_score = model_multiple.predict_proba(X_test)

    # with open('results/'+model_name+'.txt','w') as fh:
    # # Pass the file handle in as a lambda function to make it callable
    #     model_multiple.summary(print_fn=lambda x: fh.write(x + '\n'))

    # keras.utils.plot_model(model_multiple, to_file='results/'+model_name+'.png',show_shapes=True,expand_nested=True)
    
    return y_predict.ravel(),y_score.ravel()


def main():
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
    y_test = np.array(y_test).astype('int32').ravel()

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure()

    # SVM
    y_predict,y_score = svm(X_train,y_train,X_test)
    cal_acc('SVM',y_predict,y_test)
    plot_roc(color[0],'SVM',y_score,y_test)

    # Single layer
    y_predict,y_score = single_layer(X_train,y_train,X_test)
    cal_acc('single_Layer',y_predict,y_test)
    plot_roc(color[1],'single_Layer',y_score,y_test)

    model = keras.models.load_model("models/single_layer_model.h5")
    f_prediction = open("prediction_results.txt",'w')
    for i in range(5):
        single_layer_prediction(model,X_test.iloc[i],f_prediction,str(y_test[i]))
    f_prediction.close()

    #multi layer
    # epoch_list = [50, 100, 200]
    # learning_rate_list = [0.1, 0.01, 0.001]
    # use_bias_list = [True, False]
    # activation_list = ['relu','sigmoid','tanh']
    epoch_list = [200]
    learning_rate_list = [0.1]
    use_bias_list = [True]
    activation_list = ['sigmoid']
    for epoch in epoch_list:
        for learning_rate in learning_rate_list:
            for use_bias in use_bias_list:
                for activation in activation_list:
                    model_name = "multilayer_"+str(use_bias)+"_"+activation+"_"+str(learning_rate)+"_"+str(epoch)
                    y_predict,y_score = multilayer(model_name,X_train,y_train,X_test,use_bias,activation,learning_rate,epoch)
                    cal_acc(model_name,y_predict,y_test)
                    if(epoch==200 and learning_rate==0.1 and use_bias==True and activation=='sigmoid'):
                        plot_roc(color[2],model_name,y_score,y_test)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_fine_tune.png")
    # plt.show()

if __name__== "__main__":
    main()