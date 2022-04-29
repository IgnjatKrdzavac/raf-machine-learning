# TODO popuniti kodom za problem 3c
%tensorflow_version 1.x



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

%matplotlib inline



class KNN:
  
  def __init__(self, nb_features, nb_classes, data, k, weighted = False):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.data = data
    self.k = k
    self.weight = weighted
    
    # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
    self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Q = tf.placeholder(shape=(nb_features), dtype=tf.float32)
    
    # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), 
                                  axis=1))
    _, idxs = tf.nn.top_k(-dists, self.k)  
    
    self.classes = tf.gather(self.Y, idxs)
    self.dists = tf.gather(dists, idxs)
    
    if weighted:
       self.w = 1 / self.dists  # Paziti na deljenje sa nulom.
    else:
       self.w = tf.fill([k], 1/k)
    
    # Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
    
    # Klasa sa najvise glasova je hipoteza.
    self.hyp = tf.argmax(self.scores)
  
  # Ako imamo odgovore za upit racunamo i accuracy.
  def predict(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      nb_queries = query_data['x'].shape[0]
      
      # Pokretanje na svih 10000 primera bi trajalo predugo,
      # pa pokrecemo samo prvih 100.
     # nb_queries = 100
      
      matches = 0
      for i in range(nb_queries):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'], 
                                                  self.Y: self.data['y'], 
                                                 self.Q: query_data['x'][i]})
        if query_data['y'] is not None:
          actual = query_data['y'][i]
          match = (hyp_val == actual)
          if match:
            matches += 1
          if i % 10 == 0:
            print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
                 .format(i+1, nb_queries, hyp_val, actual, match))
      
      accuracy = matches / nb_queries
      print('{} matches out of {} examples'.format(matches, nb_queries))


      return accuracy






#ucitavanje podataka
f = pd.read_csv('/content/social_network_ads.csv')
#file.head()

#X - input, sve osim kolone purchased
#y - outut, cuvamo vrednosti purchased 

#f = f.astype([('User ID', 'int64'),('Gender','int64'),('Age','int64'),('EstimatedSalary','int64'),('Purchased','int64')])

#X = f[:, 1:4]
X = f.drop(columns=['Purchased']).values
#normalizujem
#X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

for row  in X:
  if row[1] == 'Male':
    row[1] = 0;
  else:
    row[1] = 1;
print(X)

y = f['Purchased'].values




train_ratio = 0.8
nb_samples = 400

nb_train = int(train_ratio * nb_samples)
data_train = dict()
data_train['x'] = X[:nb_train]
data_train['y'] = y[:nb_train]

nb_test = nb_samples - nb_train
data_test = dict()
data_test['x'] = X[nb_train:]
data_test['y'] = y[nb_train:]


nb_features = 4
nb_classes = 2
k = 3

lista = []

for k in range(1,15): 
  knn = KNN(nb_features, nb_classes, data_train, k, weighted=False)

  accuracy = knn.predict(data_test)
  lista.append(accuracy)


print(lista)



plt.plot(range(1,15), lista)
plt.show()
