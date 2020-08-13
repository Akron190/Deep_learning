##This shows you ho to use Gradient Descent as optimizer in tensorflow-cpu

import tensorflow as tf
from sklearn.datasets import load_iris

iris = load_iris()
data  = iris['data']
lables = iris['target']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3 , activation=tf.nn.softmax))
model.compile(optimizer=tf.train.GradientDescentOptimizer(100),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(data,lables,epochs=2)
