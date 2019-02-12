import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import processing

# load ascii text and covert to lowercase
filename = "../data/shakespeare.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
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
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(200))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-50-2.1945-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed

# counting new line characters to generate a 14 line poem
count = 0
emission = []
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# we want 14 lines for our sonnet
while (count < 14):
    print(count)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # normalize
    x = x / float(n_vocab)
    # predict probability of next character
    prediction = model.predict(x, verbose=0)
    # gets index by probability
    index = np.random.choice(range(48), 1, p = prediction[0])[0]
    result = int_to_char[index]
    emission.append(result)
    seq_in = [int_to_char[value] for value in pattern]
    if result == '\n':
        count += 1
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print(''.join(emission))
