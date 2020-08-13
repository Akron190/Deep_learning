##This shows you ho to use Gradient Descent as optimizer in tensorflow-cpu

#import required packages
import tensorflow as tf
from sklearn.datasets import load_iris

#import data
iris = load_iris()
#define data and lables
data  = iris['data']
lables = iris['target']
# define model
model = tf.keras.models.Sequential()
# add a dense layer
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3 , activation=tf.nn.softmax))
# addig loss,optimizer
model.compile(optimizer=tf.train.GradientDescentOptimizer(100),loss='categorical_crossentropy',metrics=['accuracy'])
# training our data
model.fit(data,lables,epochs=2)
