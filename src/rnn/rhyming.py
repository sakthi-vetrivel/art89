import nltk
import string
import json

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict
from nltk.util import ngrams

d = cmudict.dict()

def split_lines(filename):
    # create token for each line in the file
    tokenizer = RegexpTokenizer('\w[\w|\'|-]*\w|\w')
    line_tokens = []
    f = open(filename)
    for line in f:
        line = line.strip()
        if (line.isdigit()):
            continue
        if (len(line) > 0):
            line = line.lower()
            tokens = tokenizer.tokenize(line)
            if len(tokens) > 1:
                line_tokens.append(tokens)
    return line_tokens

def parse_rhyme(word):
    k = ''
    try:
        # try the find the pronounciation in the dictionary
        pronounciation = d[word][-1]
        k = ','.join(pronounciation[-2:])

    except (KeyError):
        pass

    # if there are no rhymes, we just return the empty string
    return word, k

def parse_words(line):
    tot = 0
    # return the generator for the line
    for word in line:
        sk = parse_rhyme(word)[1]
        yield word,sk

if __name__=='__main__':
    line_tokens = []
    files = ['../data/shakespeare.txt', '../data/more_shakespeare.txt', '../data/spenser.txt']
    for filename in files:
        line_tokens.extend(split_lines(filename))
    rhyme = {}

    for line in line_tokens:
        for word,sk in parse_words(line):
            # Save meter of word
            if len(sk) > 0:
                if sk in rhyme.keys():
                    rhyme[sk].add(word)
                else:
                    rhyme[sk] = set()
                    rhyme[sk].add(word)

    for k, v in rhyme.items():
       rhyme[k] = list(v)

    inv_rhyme = {}
    for key, value in rhyme.items():
        for string in value:
            inv_rhyme.setdefault(string, []).append(key)

    with open('../json/rhyme.json', 'w') as f:
        json.dump(rhyme, f)
    with open('../json/inv_rhyme.json', 'w') as f:
        json.dump(inv_rhyme, f)
