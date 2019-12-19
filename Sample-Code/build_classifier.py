import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import wget
import sklearn
from sklearn.model_selection import train_test_split


#We pull fannie mae mortgage data with the urls below
#Each data set represents a class label
url0 = "https://labs-fannie-data.s3.amazonaws.com/q4_l0.csv"
url1 = "https://labs-fannie-data.s3.amazonaws.com/q4_l1.csv"
url2 = "https://labs-fannie-data.s3.amazonaws.com/q4_l2.csv"
url3 = "https://labs-fannie-data.s3.amazonaws.com/q4_l3.csv"

wget.download(url0)
wget.download(url1)
wget.download(url2)
wget.download(url3)

l0 = pd.read_csv(r'q4_l0.csv', header=None)
l1 = pd.read_csv(r'q4_l1.csv', header=None)
l2 = pd.read_csv(r'q4_l2.csv', header=None)
l3 = pd.read_csv(r'q4_l3.csv', header=None)

print(l0.shape, l1.shape, l2.shape, l3.shape)

#We add the class labels to each dataframe in the form of a new column

l0['5'] = pd.DataFrame(np.array([0 for i in range(len(l0))]))
l1['5'] = pd.DataFrame(np.array([1 for i in range(len(l1))]))
l2['5'] = pd.DataFrame(np.array([2 for i in range(len(l2))]))
l3['5'] = pd.DataFrame(np.array([3 for i in range(len(l3))]))

full_dataset = pd.concat([l0, l1, l2, l3], axis=0)
print(full_dataset.shape)

full_dataset.iloc[:,:-1] = tf.keras.utils.normalize(full_dataset.iloc[:,:-1], axis=0, order=2)

#random subset of the full data set to ensure that the class labels are unaffected by normalization
print(full_dataset[40000:40007])


X = full_dataset.iloc[:,:-1]
Y = full_dataset.iloc[:,-1]
print(X.shape)
print(Y.shape)

#We randomly assort our feature and target variables making use of sklearn's train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=30)

print("X_train Shape: {}".format(X_train.shape))
print("Y_train Shape: {}".format(Y_train.shape))
print("X_test Shape: {}".format(X_test.shape))
print("Y_test Shape: {}".format(Y_test.shape))

# Our target variable is multiclass (total 5, i.e. [0,1,2,3,4]);
# as a result we will be utilizing sparse-categorical cross-entropy loss function with activation softmax.


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=[5]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)


history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=3)

#We observe the loss and accuracay of our model (train and validation) from epoch to epoch
print(history.history)
#model.save(r"Classifier.h5")


#We will now build a slightly more advanced model which will take into consideration the weighting of each class label;
# we will also add dropout layers.

print(full_dataset.iloc[:,-1].value_counts())

#Now that we have a visual representation of the disparity in each class label value count,
# we may determine appropriate class weightings.

my_dict = {3:30000, 0:25437, 1:10320, 2:1930}

class_weights = {}
for i in range(4):
  prop = my_dict[i]/len(full_dataset)
  weight = 1 - prop
  dic = {i: weight}
  class_weights.update(dic)

print(class_weights)

#Now we make use of the same model as before, but this time we add a dropout rate
# to the input layer and assign our class weights upon fitting the model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=[5]),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    class_weight = class_weights, #introduced the class weights in the training process
                    epochs=3)

#We observe how this refined model compares to its predecessor
print(history.history)
