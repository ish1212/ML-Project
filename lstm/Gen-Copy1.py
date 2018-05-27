from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


# In[2]:

ids = np.load('idsMatrix22.npy')
labels = np.load('final_topics2.npy')

wordlist = np.load('wordsList.npy')
wordvec = np.load('wordVectors.npy')


# In[3]:

ids = ids[:1000,:200]


# In[5]:

labels = labels[:1000]


# In[6]:

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 4
step = 1
sentences = []
next_chars = []
quote_len = ids.shape[1]

for quote in ids:
    for i in range(0, quote_len - maxlen, step):
        if quote[i] >0:
            sentences.append(quote[i: i + maxlen])
            next_chars.append(quote[i + maxlen])
print('nb sequences:', len(sentences))


# In[9]:

sentences[0:10]


# In[11]:

chars = set()

for sentence in sentences:
    chars = set(sentence).union(set(chars))

chars = sorted(list(set(chars)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


# In[13]:

len(char_to_int), char_to_int, int_to_char


# In[14]:

sequences=[]
for s,sentence in enumerate(sentences):
    sequence = []
    for w, word in enumerate(sentence):
        print(s,w, word)
        sequence.append(char_to_int[word])
    sequences.append(sequence)


# In[15]:

sequences = np.asarray(sequences)


# In[16]:

sequences.shape, sequences.max()


# In[17]:

np.save('sequences.npy',sequences)


# In[18]:

next_seq = []
for w, word in enumerate(next_chars):
    next_seq.append(char_to_int[word])


# In[19]:

next_seq = np.asarray(next_seq)


# In[20]:

np.save('next_seq.npy',next_seq)


# In[21]:

next_seq.shape, next_seq.max()


# In[70]:

max_word = np.asarray(next_seq.max())


# In[22]:

from keras.utils import to_categorical


# In[26]:

# reshape X to be [samples, time steps, features]
X = np.reshape(sequences, (len(sequences), maxlen, 1))
# normalize
X = X / float(next_seq.max())
# one hot encode the output variable
y = to_categorical(next_seq)


# In[32]:

y.shape, X.shape


# In[33]:

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))



model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[34]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[35]:

text = np.ndarray.flatten(sequences)


# In[36]:

text.shape


# In[117]:

x_pred = 0
generated = 0
sentence = 0
def on_epoch_end(epoch, logs):
    global x_pred
    global sentence 
    global generated
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        print(sentence)
        generated.join([str(wordlist[[int_to_char[value]]]).join(' ') for value in sentence])
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


# In[118]:

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(X, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])


# In[ ]:



