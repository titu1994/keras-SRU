'''Trains an SRU model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- Increase depth to obtain similar performance to LSTM
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.datasets import imdb

from sru import SRU

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

depth = 1

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
ip = Input(shape=(maxlen,))
embed = Embedding(max_features, 128)(ip)

prev_input = embed
hidden_states = []

if depth > 1:
    for i in range(depth - 1):
        h, h_final, c_final = SRU(128, dropout=0.0, recurrent_dropout=0.0,
                                  return_sequences=True, return_state=True,
                                  unroll=True)(prev_input)
        prev_input = h
        hidden_states.append(c_final)

outputs = SRU(128, dropout=0.0, recurrent_dropout=0.0, unroll=True)(prev_input)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(ip, outputs)
model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
