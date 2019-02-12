from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import tensorflow as tf
import numpy as np
import random
import json
import pickle
import re
import sys

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

def generate_from_model(model, int_to_char, char_to_int, n_vocab, seq_length):
	#start = np.random.randint(0, len(dataX)-1)

	# diversity in [0.2, 0.5, 1.0, 1.2]:
	for diversity in [1.5, 0.75, 0.25]:
		print('----- diversity:', diversity)
		seed = '  shall i compare thee to a summers day\n'
		sentence = []
		for c in seed:
			sentence.append(char_to_int[c])
		generated = seed
		print('----- Generating with seed: "' + seed + '"')
		sys.stdout.write(generated)

		tot_lines = 0
		new_line = False

		while True:
			if tot_lines > 14:
				break
			x = np.reshape(sentence[0:seq_length], (1, seq_length, 1))
			# normalize
			x = x / float(n_vocab)
			preds = model.predict(x, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = int_to_char[next_index]
			generated += next_char
			if next_char == '\n':
				tot_lines += 1
			sentence.append(next_index)
			sentence = sentence[1:len(sentence)]
			# for the format of the volta
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()
