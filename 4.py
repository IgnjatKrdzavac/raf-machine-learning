# TODO popuniti kodom za problem 4


%tensorflow_version 1.x

import nltk
nltk.download()



!pip install nltk

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import csv
import sys
csv.field_size_limit(sys.maxsize)
import html
import re
import random
import json
import nltk
import math
from string import punctuation
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from string import ascii_lowercase
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')


#ucitavamo tekst iz fajla
file = '/content/fake_news.csv'

X = []
y = []

with open(file, 'r', encoding='latin1') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)
            for row in reader:
                y.append(int(row[4]))
                X.append(row[3])
nb_classes = 2


#ciscenje

X = [html.unescape(x) for x in X]

#specijalni karakteri
X = [re.sub(r'[^\w\s]|\d+', '', x) for x in X]
X = [re.sub(r'\s\s+', ' ', x) for x in X]
#mala slova
X = [x.strip().lower() for x in X]
#ascii
X = [x.encode('ascii', 'ignore').decode() for x in X]
#tokenizacija
X = [regexp_tokenize(x, '\w+') for x in X]
#stopwords (the, a...)
stop_punc = set(stopwords.words('english')).union(set(punctuation))
X = [[w for w in l if w not in stop_punc] for l in X]



print('ociscen ', X[0])


#kreiramo vokabular

vocab_set = set()
for doc in X:
  for word in doc:
    vocab_set.add(word)
vocab = list(vocab_set)



#ogranicavamo vocab na trazenu vrednost 10000(trenutno zbog brzeg ucitavanja je na 100)
freqDist = FreqDist([w for x in X for w in x])
number = 10000
vocab, _ = zip(*freqDist.most_common(number))






features = np.zeros((len(X), number), dtype=np.float32)

lr = np.zeros(number, dtype=np.float32)

#j brojac od 0 do broj reci iz vokabulara, w sama  rec
for j, w in enumerate(vocab):
      neg = 0
      pos = 0
            
      for i, x in enumerate(X):
            cnt = x.count(w)
            #matrica jedinstvenih reci
            features[i][j] = cnt
            if y[i] == 0:
               neg += cnt
            else:
                pos += cnt
      if pos >= 10 and neg >= 10:
            lr[j] = pos / neg
      if j % 100 == 0:
             print('[calculate_bow_lr] Word: {}/{}'.format(j, number))






print(vocab)
print(X[0])
for doc in X:
  print(doc)





#test i trening
x_train, x_test = np.split(features, [int(len(features)*0.8)])
y_train, y_test = np.split(y, [int(len(y)*0.8)])
nb_train = len(x_train)
nb_test = len(x_test)




class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount #alfa, iz formule


  #sluzi da istreniramo model
  def fit(self,x_train,y_data):
    nb_examples = x_train.shape[0]

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    #P(Klasa)=|Elementi trening skupa klase Klasa|/|Ceo trening skup| 
    #koliko puta nam se pojavljuje neki tekst u svim klasama
    #bincount 
    self.priors = np.bincount(y_data) / nb_examples
    #priors = np.bincount(y) / nb_examples
    print('Priors:')
    print(self.priors)

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    self.occs = np.zeros((self.nb_classes, self.nb_words), dtype=np.float32)
    for i, y in enumerate(y_train):
      for w in range(nb_words):
        self.occs[y][w] += x_train[i][w]
      if i % 100 == 0:
        print('[calculate_occurrences] Object: {}/{}'.format(i, nb_train))

    # Racunamo P(Rec_i|Klasa) - likelihoods - verovatnocu pojavljivanja reci u klasi
    #P(Reči|Klasa)=broj_pojavljivanja(Reči,Klasa)+α/ukupan_broj_reči(Klasa)+|Vocab|⋅α
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = self.occs[c][w] + self.pseudocount
        #nb_words ukupan broj reci, occs ukupan broj reci po klasi
        down = np.sum(self.occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
    print('Likelihoods:')
    print(self.like)

  def predict(self, x_test):
    # Racunamo P(Klasa|bow) za svaku klasu
    #popunjavamo verovatnocu sa nulama
    #P(Klasa|BoWvektor)∼P(Klasa)⋅∏P(Reči|Klasa)^BoW[Reči]
    probs = np.zeros(self.nb_classes,dtype=object)
    for c in range(self.nb_classes):
      #logaritmujemo
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = x_test[w]
        prob += cnt* np.log(self.like[c][w])#verovatnoca da se ta rec nalazi u klasi
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    # print('"Probabilites" for a test BoW (with log):')
    # print(probs)
    prediction = np.argmax(probs)
    return prediction

class_names = ['pouzdan', 'nepouzdan']
model = MultinomialNaiveBayes(nb_classes=2, nb_words=10000, pseudocount=1)
model.fit(x_train, y)
pogodak = 0
for i in range (len(x_test)):
  prediction = model.predict(x_test[i])
  if(prediction == y_test[i]):
    pogodak+=1

acc = pogodak/len(x_test)
print(acc)
#print('Predicted class (with log): ', class_names[prediction])
