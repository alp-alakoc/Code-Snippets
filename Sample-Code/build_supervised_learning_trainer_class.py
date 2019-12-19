import pandas as pd
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sns


#We construct a class which will allow us to build and test multiple ML methods dedicated to a classification problem
#The class addresses decision-tree-classifiers; random forest classifiers; support vector machines; and deep-learning ANNs

#The class is instantiated with several key arguments attributing to the most impact in the respective ML models

#It is up to the user to configure their data accordingly prior to using the class (specifically the neural net which
#has a input shape of dimension 2...

class Model_Trainer(object):

    def __init__(self, max_depth, class_weight, n_estimators, gamma, kernel, neurons):
        self.clf = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
        self.rfc = RandomForestClassifier(n_estimators=n_estimators)
        self.svc = svm.SVC(gamma=gamma, kernel=kernel)
        self.NN = tf.keras.Sequential([
            tf.keras.layers.Dense(neurons, input_shape=[2]),
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense("", activation='softmax')])

    def trainer(self, train_x, train_y, test_x, test_y):
        fit_clf = self.clf.fit(train_x, train_y)
        fit_rfc = self.rfc.fit(train_x, train_y)
        fit_svc = self.svc.fit(train_x[:"SOME - Train -Size"], train_y.ravel()[:"SOME - Train -Size"])
        self.NN.compile(optimizer=tf.keras.optimizers.RMSprop(),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.history = self.NN.fit(train_x, train_y,
                                   validation_data= (test_x, test_y),
                                   epochs=2)

        mydict = dict()
        mydict.update({'Decision Tree Score': fit_clf.score(test_x, test_y)})
        mydict.update({'Random Forest Score': fit_rfc.score(test_x, test_y)})
        mydict.update({'Support Vector Machine Score': fit_svc.score(test_x, test_y)})
        mydict.update({'Neural Net Score': self.history.history})

        return mydict

    def confusion_plotter(self, test_x, test_y):
        cm1 = confusion_matrix(test_y, self.fit_clf.predict(test_x))
        cm2 = confusion_matrix(test_y, self.fit_rfc.predict(test_x))
        cm3 = confusion_matrix(test_y, self.fit_svc.predict(test_x))
        cm4 = confusion_matrix(test_y, self.NN.predict(test_x))

        heatmap1 = sns.heatmap(cm1, annot=True, fmt='d')
        heatmap2 = sns.heatmap(cm2, annot=True, fmt='d')
        heatmap3 = sns.heatmap(cm3, annot=True, fmt='d')
        heatmap4 = sns.heatmap(cm4,annot=True,fmt='d')

        p1 = plt.figure(1)
        print('Confusion Matrix: Decision Tree')
        p1.show(heatmap1)
        print('\n'+ 'Confusion Matrix: Random Forest')
        p2 = plt.figure(2)
        p2.show(heatmap2)
        print('\n' + 'Confusion Matrix: Support Vector Machine')
        p3 = plt.figure(3)
        p3.show(heatmap3)
        print('n' + 'Confusion Matrix: Neural Net')
        plt.show(heatmap4)

        return



