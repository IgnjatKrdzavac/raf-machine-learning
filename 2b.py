# TODO popuniti kodom za problem 2b

%tensorflow_version 1.x

%matplotlib inline
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)
  
global data_loss
data_loss = []

def treniranje(dataX, dataY, nb_samples, nb_features, nb_epochs):

 
  data_loss.append([])
  dataX = create_feature_matrix(dataX, nb_features)

  #Model
  X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
  Y = tf.placeholder(shape=(None), dtype=tf.float32)
  w = tf.Variable(tf.zeros(nb_features))
  bias = tf.Variable(0.0)

  w_col = tf.reshape(w, (nb_features, 1))
  hyp = tf.add(tf.matmul(X, w_col), bias)

  # Korak 3: Funkcija troška i optimizacija.
  Y_col = tf.reshape(Y, (-1, 1))

 

  mse = tf.reduce_mean(tf.square(hyp - Y_col))
 

  opt_op = tf.train.AdamOptimizer().minimize(mse)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  
    # Izvršavamo 1000 epoha treninga.
    
    for epoch in range(nb_epochs):
    
      # Stochastic Gradient Descent.
      epoch_loss = 0
      for sample in range(nb_samples):
        feed = {X: dataX[sample].reshape((3, nb_features)), 
                Y: dataY[sample]}
        _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
        epoch_loss += curr_loss
        
      # U svakoj ntoj epohi ispisujemo prosečan loss.
      epoch_loss /= nb_samples
      data_loss[nb_features-1].append(epoch_loss)
      if (epoch + 1) % 1 == 0:
        print('Epoch: {}/{}| Avg loss: {:.5f}'.format(epoch+1, nb_epochs, 
                                              epoch_loss))
 
    # Ispisujemo i plotujemo finalnu vrednost parametara.
    w_val = sess.run(w)
    bias_val = sess.run(bias)
    print('w = ', w_val, 'bias = ', bias_val)
    xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
    hyp_val = sess.run(hyp, feed_dict={X: xs})  # Bez Y jer nije potrebno.
    stage = nb_features / nb_features_max
    plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=(1-stage, stage, 0))
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    



tf.reset_default_graph()

nb_features_min = 1
nb_features_max = 6

#ucitavamo fajl
filename = '2a.xls'
all_data = np.loadtxt(filename,delimiter=',',skiprows = 1, usecols=(2, 3, 4, 7))
data= dict()
data['x'] = all_data[: , :3]
data['y'] = all_data[:, 3]
print(f"ucitavanje data [x]", data['x'])


nb_epochs = 10

# Nasumično mešanje.
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]


# Normalizacija
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])


plt.figure(figsize=(20, 15))
plt.subplot(4, 1, 1)
plt.scatter(data['x'][:, 0], data['y'], linewidths = 0)
plt.xlabel('Temperature')
plt.ylabel('PM2.5_ambient')

for nb_features in range(nb_features_min, nb_features_max+1):
   treniranje(data['x'],data['y'], nb_samples, nb_features, nb_epochs)

plt.figure(figsize=(20, 15))
plt.subplot(4, 1, 2)
plt.scatter(data['x'][:, 1], data['y'], linewidths = 0)
plt.xlabel('Humidity')
plt.ylabel('PM2.5_ambient')

for nb_features in range(nb_features_min, nb_features_max+1):
    treniranje(data['x'],data['y'], nb_samples, nb_features, nb_epochs)

plt.figure(figsize=(20, 15))
plt.subplot(4, 1, 3)
plt.scatter(data['x'][:, 2], data['y'], linewidths = 0)
plt.xlabel('lowcost PM2.5')
plt.ylabel('PM2.5_ambient')

data_loss = []

for nb_features in range(nb_features_min, nb_features_max+1):
    treniranje(data['x'],data['y'], nb_samples, nb_features, nb_epochs)

#plt.figure(figsize=(20, 15))
# Grafik koji prikazuje zavisnost funkcije troska od stepena polinoma
plt.subplot(4, 1, 4)
for nb_features in range(nb_features_min, nb_features_max+1):
    stage = nb_features / nb_features_max
    plt.plot(range(1, nb_epochs+1), data_loss[nb_features-1], color=(1-stage, stage, 0))



