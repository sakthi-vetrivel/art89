
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import tensorflow as tf
import numpy as np
import random
import json
import pickle
import re

def sample(a, temperature=1.0):
	a = np.log(a) / temperature
	dist = np.exp(a)/np.sum(np.exp(a))
	choices = range(len(a))
	return np.random.choice(choices, p=dist)


def build_model(window, len_chars):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(window, len_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len_chars))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

if __name__=='__main__':
    # load ascii text and covert to lowercase
    text = ''
    files = ['../data/shakespeare.txt', '../data/more_shakespeare.txt', '../data/spenser.txt']
    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line.isdigit():
                    text += '$' + line.lower() + '\n'
    text = re.sub('[:;,.!()?&]', '', text)
    # create mapping of unique chars to integers
    chars = sorted(list(set(text)))

    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 25 # Window size

    # Train in reverse, so we can construct lines from the back for rhyme
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i + maxlen: i: -1])
        next_chars.append(text[i])

    print('Number sequences:', len(sentences))

    # One hot encode sequences
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1

    model = build_model(maxlen, len(chars))
    model.fit(X, y, batch_size=128, nb_epoch=60)
    model.save_weights("backwards_char_rnn.h5")
    print("Saved model to disk")
