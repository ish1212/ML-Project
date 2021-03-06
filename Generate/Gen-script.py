
# coding: utf-8

# In[1]:

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


# In[2]:

wordlist = np.load('wordsList.npy')


# In[3]:

int_to_char = np.load('int_to_char.npy')
char_to_int = np.load('char_to_int.npy')

int_to_char = int_to_char.item()
char_to_int = char_to_int.item()

sequences = np.load('sequences.npy')
next_seq = np.load('next_seq.npy')


# In[4]:

max_word = np.asarray(next_seq.max())
maxlen = 4


# In[5]:

from keras.utils import to_categorical


# In[6]:

# reshape X to be [samples, time steps, features]
X = np.reshape(sequences, (len(sequences), maxlen, 1))
# normalize
X = X / float(next_seq.max())
# one hot encode the output variable
y = to_categorical(next_seq)


# In[7]:

# build the model: a single LSTM
print('Build model...')
model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256))
model.add(Dropout(0.2))

model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[8]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = preds[1:]
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[9]:

text = np.ndarray.flatten(sequences)


# In[10]:

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        print(sentence)
        generated.join([str(wordlist[[int_to_char[value]]]) for value in sentence])
        print('----- Generating with seed: %s'%sentence)
#         sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.reshape(sentence,(1, maxlen, 1))
            x_pred = x_pred / max_word

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = wordlist[int_to_char[next_index]]

            generated.join(str(next_char))
            sentence = np.append(sentence[1:],next_index)

            sys.stdout.write(next_char)
            sys.stdout.write(" ")
            sys.stdout.flush()
        print()


# In[11]:

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

model.fit(X, y,
          batch_size=32,
          epochs=100,
          callbacks=[print_callback, checkpoint])


# In[ ]:



