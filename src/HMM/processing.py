def read_saved_HMM(name):
    f= open("saved/"+name, "r")
    syllableslist = f.read().splitlines()
    L = 0
    D = 0
    A = []
    O = []
    for i in range(0, len(syllableslist)):
        data = syllableslist[i].split(' ')
        if i == 0:
            L = int(data[0])
            D = int(data[1])
        else:
            floats = []
            for d in data:
                floats.append(float(d))
            if (i <= L):
                A.append(floats)
            else:
                O.append(floats)
    return (A, O)


def add_to_rhyme(worda, wordb, rhymedict):
    '''
    Arguments:
        worda: an integer that maps to one of the words of the rhyme-pair
        wordb: an integer that maps to the other word in the rhyme-pair
        rhymedict: a dictionary that maps from an integer which maps to a word
                   to a list of words that rhyme with it
    This function adds a pair of rhyming words to a dictionary of rhymes.
    Output:
        rhymedict: an updated rhyme dictionary that includes that worda and
                   wordb rhyme.
    '''
    try:
        rhymedict[worda].append(wordb)
    except KeyError:
        rhymedict[worda] = [wordb]
    try:
        rhymedict[wordb].append(worda)
    except KeyError:
        rhymedict[wordb] = [worda]
    return rhymedict



def converttoint(line, wordtonum, numtoword):
    '''
    Arguments:
        line: A string containing a single line of an input poem.
        wordtonum: A dictionary that maps words to unique integers, where
                   each integer corresponds to an "emission".
        numtoword: A dictinoary that maps integer "emissions" back to the
                   words corresponding to each integer. This is used to
                   convert emissions to an understandable state.
    This function removes quotes and fills the dictionaries wordtonum and
    numtoword so that we can convert from word to int and vice versa. It also
    converts the line of words into a list of integers, which can be used to
    train our HMM model.

    Note: while it seems more intuitive to remove the quotes in the function
    removechars(), we are doing it here to minimize the number of times
    that we have to loop over each line of the poem and improve the efficiency
    of our program.

    Output:
        converted: A list of integers containing the integers which map to the
                   words of the line using the numtoword dictionary.
    '''
    # splits the line into a list and appends the "\n" string to the end of the
    # list, because this character was removed from the end of the lines earlier
    # when reading lines. Including this character allows us to represent
    # that lines end.
    line = line.split(' ')
    line.append("\n")
    # a list that stores the integers mapping to the words (and "\n") in line.
    converted = []
    # these two variables are used for the removal of quotes in lines.
    quote = False
    q = "'"

    # processes every word in the line
    for word in line:
        # converts the word to lowercase
        word = word.lower()
        # if the first letter of the word is an apostrophe and the word is not
        # one of our recognized words, then we consider this the start of a
        # quote and remove the apostrophe and set quote to true.

        if(word[0] == q[0] and (not word.replace(q[0], '') == "tis") and \
        (not word.replace(q[0], '') == "gainst") and \
        (not word.replace(q[0], '') == "greeing") and \
        (not word.replace(q[0], '') == "scaped") and \
        (not word.replace(q[0], '') == "twixt")):
            word = word.replace(q[0], '')
            quote = True
        # if we have established that we are in a quote, then we remove the
        # next apostrophe character that we see.
        # Note: we could do this in a more sophistocated way: i.e. remove the
        # apostrophe only if it is the last character of the word and we are
        # in a quote, but it doesn't appear that issue arises here.
        # Once we have removed the second apostrophe, we are no longer in a
        # quote and set quote = False so that we stop looking for the end of
        # a quote.
        if quote:
            if q[0] in word:
                word = word.replace(q[0], '')
                quote = False
        # checks if the word is already in the dictionary. If so, it appends
        # the value it maps to to converted.
        if word in wordtonum:
            converted.append(wordtonum[word])
        # if it is not in the dictionary, word is added to both dictionaries,
        # and the value it now maps to is appende to converted
        else:
            wordtonum[word] = len(wordtonum)
            numtoword[len(numtoword)] = word
            converted.append(wordtonum[word])
    return converted

def removechars(original):
    '''
    Argument:
        original: the original line that was read without any special
                  characters removed
    This function removes special characters .,:;?() from original and returns
    the new, modified string.
    Output:
        replacement: the new, processed line, without the special characters
    '''
    # loops through each character in the line, and if the character is one
    # of the specified characters to be removed, the character is removed from
    # the line.
    replacement = original
    for char in replacement:
        if char in '.,:;?()':
            replacement = replacement.replace(char, '')
    return replacement

def readshakes():
    '''
    This function reads all of the poems stored in Shakespeare.txt and
    removes punctuation, and returns two lists: one storing each poem as a
    list of words and newline characters, and another storing a list of
    numbers corresponding to each poem.
    Output:
        poems: this is a list of lists, where each list stored is a integer
               representation of a poem input.
        poemnums: this is a list storing the number corresponding to each of
                  the poems stored in poems. It is indexed so that poems[i]
                  is poem number poemnums[i]. This list is generated so that
                  we can potentially choose which poems to exclude from training
                  after our initial processing
        wordtonum: this is a dictionary that maps each "word" in a string to an
                   integer value so that the observation can map to
                   indeces of matrices. "word" is used loosely here, as the
                   dictionary also maps "\n" to an integer as well. It really
                   maps our tokens to unique integers, and the tokens happen to
                   be words most of the time
        numtoword: this is a dictionary that maps the unique integers back to
                   the "word" or token that they represent. It is ultimately
                   necessary for generating poems and converting them back into
                   a form with words, which is interperable.
    '''
    pfile = open("data/shakespeare.txt", "r")
    pread = pfile.read().splitlines()
    # the current line number of the poem we are reading
    line = 0
    # the list corresponding to the current poem being read.
    poem = []
    poems = []
    poemnums = []
    rhymesa = ''
    rhymesb = ''
    # initalizes our dictionaries, and maps "\n" to 0 and vice versa for
    # our convenience later when generating poems.
    rhymedict = {}
    wordtonum = {"\n" : 0}
    numtoword = {0 : "\n"}
    currentpoem = 0
    # processes each line of the file being read
    for p in pread:
        # Lines starting with long strings of space characters indicate the
        # number of the poem, so we read in and store the poem's identification
        # number and set line = 1, so that we know the next p we process is the
        # first line of a poem.
        if (p[0:19] == '                   '):
            currentpoem = int(p[19:len(p)])
            #print(currentpoem)
            line = 1
        else:
            # if the length of p is zero, then we know that this is an empty
            # line between poems and we must be done reading our previous poem.
            # So, if we have a processed poem stored, we append it to poems and
            # its identification number to poemnums and then we set poem to
            # be empty again so that it is ready to store the next poem we
            # see. We also set line = 0 here so that we are about to start
            # reading a new poem
            if len(p) == 0:
                if len(poem) > 0:
                    poems.append(poem)
                    poemnums.append(currentpoem)
                    poem = []
                    line = 0

            # Note: I tried to remove these abnormal poems in a different
            # function, but that was unsuccessful so I just removed them here.
            # we might want to change this later, but for now this works and I
            # doubt we will want to train with these poems anyways, so it is
            # probably fine. I've left their processing commented out though
            # so that if we do want to remove these poems somewhere else, we
            # can still process and store them if we want to.

            # Now, we process lines by calling removechars() and converttoint()
            # and adding these values to the end of our list poem.
            else:
                if (currentpoem == 99):
                    # if (line < 14):
                    #     poem.extend(converttoint(removechars(p), wordtonum, numtoword))
                    # else:
                    #     poem.extend(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                    currentpoem = currentpoem
                else:
                    if (currentpoem == 126):
                        # if (line < 11):
                        #     poem.extend(converttoint(removechars(p), wordtonum, numtoword))
                        # else:
                        #     poem.extend(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                        currentpoem = currentpoem
                    else:
                        if (line < 13):
                            poem.extend(converttoint(removechars(p), wordtonum, numtoword))
                            if (line % 4 == 1):
                                rhymesa = poem[len(poem) -2]
                            else:
                                if (line % 4 == 2):
                                    rhymesb = poem[len(poem) -2]
                                else:
                                    if (line % 4 == 3):
                                        rhymedict = add_to_rhyme(rhymesa, poem[len(poem) -2], rhymedict)
                                    else:
                                        rhymedict = add_to_rhyme(rhymesb, poem[len(poem) -2], rhymedict)

                        # if we are reading one of the last two lines, we remove
                        # the two spaces at the beginning of the line.
                        else:
                            poem.extend(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                            if (line == 13):
                                rhymesa = poem[len(poem) -2]
                            else:
                                rhymedict = add_to_rhyme(rhymesa, poem[len(poem) -2], rhymedict)
                # we increment line to keep track of what line of the poem we
                # are reading.
                line += 1

    return (poems, poemnums, wordtonum, numtoword, rhymedict)

def readsylls(wtn):
    '''
    This reads the file storing syllable information provided for us and
    produces two dictionaries, sylls, and endsylls from it, which map from
    words to a syllable count associeted with the word.
    Note: these dictionaries use all-lowercase words.
    Outputs:
        sylls: a dictionary mapping words to an integer representing the number
               of syllables the word has when it is not at the end of the line
        endsylls: a dictionary mapping words to an integer representing the
                  number of syllables the word has when it is at the end of
                  the line. Note: for most words, the number mapped to will
                  be the same in both dictionaries.
    '''
    sfile = open("data/Syllable_dictionary.txt", "r")
    # creates a list where each item corresponds to a line in the file
    syllableslist = sfile.read().splitlines()
    # initalizes the two dictionaries to be empty
    sylls = {}
    endsylls = {}
    # goes over each line of the file
    for syllables in syllableslist:
        # for each line, we split the line into words and store these words as
        # a list.
        syllables = syllables.split(' ')

        # if the length of the list is 2, then the word corresponding to this
        # line has the same number of syllables regardless of its position in
        # a line, so the values mapped to by the word are both set to be
        # this syllable count.
        endsyl = []
        syl = []
        for i in range(1, len(syllables)):
            if syllables[i][0] == 'E':
                endsyl.append(int(syllables[i][1:]))
            else:
                syl.append(int(syllables[i]))
        # we only add a syllable to our syllable dictionary if it is in our
        # word to number dictionary. Theoretically, there should only be words
        # in the syllable text file if it is in a poem, but this is just to
        # be safe and make sure we aren't storing unnecessary words and that
        # we can convert each word being represented in the syllable dictionary
        # to the integer which it maps to in word to number dictionary.
        if (syllables[0] in wtn):
            # sylls stores the normal syllable count of words
            sylls[wtn[syllables[0]]] = syl
            # if the word has a different number of syllables at the end of a
            # line, we store those syllables in endsylls. Otherwise, we store
            # the normal syllable count here too.
            if (len(endsyl) > 0):
                endsylls[wtn[syllables[0]]] = endsyl
            else:
                endsylls[wtn[syllables[0]]] = syl

    return sylls, endsylls
