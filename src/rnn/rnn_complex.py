import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from model import sample, build_model, generate_from_model
import re

# load ascii text and covert to lowercase
filename = "../data/shakespeare.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# remove special characters
raw_text = re.sub('[!@#$.,:;?()-0123456789]', '', raw_text)
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
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

# load the network weights
filename = "weights-improvement-35-0.2121-complex.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed

# train the model, output generated text after each iteration
generate_from_model(model,int_to_char, char_to_int, n_vocab, seq_length)
