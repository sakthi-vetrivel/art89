import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re

# load ascii text and covert to lowercase
filename = "../data/shakespeare.txt"
raw_text = open(filename).read()

filename = "../data/more_shakespeare.txt"
raw_text += open(filename).read()

filename = "../data/spenser.txt"
raw_text += open(filename).read()
raw_text = raw_text.lower()

# remove special characters
raw_text = re.sub('[!@#$.,:;?()-0123456789]', '', raw_text)
# create mapping of uni../datachars to integers
chars = sorted(list(set(raw_text)))
print(chars)
#create dictionaries
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 25
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 3):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(y.shape[1], activation='softmax'))

# filename = "weights-improvement-10-2.0343-complex.hdf5"
# model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-complex.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=35, batch_size=128, callbacks=callbacks_list)
