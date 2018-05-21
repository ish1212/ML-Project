import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
# fix random seed for reproducibility
np.random.seed()

ids = np.load('idsMatrix22.npy')
labels = np.load('final_topics2.npy')

ids = ids[:,:200]

labels_encode={}
labels_decode={}
for i, topic in enumerate(np.unique(labels)):
    labels_encode[topic] = i
    labels_decode[i] = topic


labels = np.asarray([labels_encode[topic] for topic in labels])



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(ids,labels,test_size=0.2)




embedding_vecor_length = 100
max_review_length = 200 #300

model = Sequential()
model.add(Embedding(400000, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(LSTM(1000))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# define the checkpoint
filepath="keras2class-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, y_train, epochs=10, batch_size=3, callbacks=callbacks_list)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
