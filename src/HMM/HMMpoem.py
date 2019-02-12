########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from HMM import HiddenMarkovModel
from Utility import Utility
import processing

def generations(HMM, k, sylls, endsylls, linesylls, rhymedict, wantrhyme):
    '''
    This function generates k emissions for each HMM, where each emission
    consists of 14 lines. Then, it converts the emission from a list of
    integers into a string of words and prints it out.
    Input:
        HMM: a trained HMM
        k: the number of emissions to generate per HMM
    '''
    q = "''"
    print("{:30}".format('Generated Emissions'))
    print('#' * 70)
    # Generate k input sequences.
    for i in range(k):
        # if wantrhyme is True, then our generated emissions will each be 14
        # lines long, with each line being 10 syllables, and the resulting
        # lines will rhyme with the rhyme scheme abab cdcd efef gg.
        if (wantrhyme):
            emission, states = HMM.generate_emission_rhyme(sylls, endsylls, rhymedict)
        # if wantrhyme is False, then if linesylls is True, generated
        # emissions will be 14 lines each, and will each be 10 syllables
        # long. Otherwise, the generated emissions will simply be 14 lines,
        # but any number of syllables long. (and could possibly contain no
        # words at all)
        else:
            if (linesylls):
                emission, states = HMM.generate_emission_set_sylls(sylls, endsylls)
            else:
                emission, states = HMM.generate_emission()
        line = []
        poem = []
        count = 0
        for e in emission:
            for i in e:
                if ntw[i] == 'i':
                    line.append('I')
                else:
                    line.append(ntw[i])
            line = ' '.join([str(i) for i in line])
            # Capitalizes the first word of every line
            if (line[0] == q[0]):
                line = line[0] + line[1].upper() + line[2:]
            else:
                line = line[0].upper() + line[1:]
            # if we are in the final two lines of the poem, then we add two
            # spaces to the beginning of the line so that they are formatted
            # correctly.
            if (count >= 12):
                line = "  " + line
            poem.append(line)
            line = []
            count += 1
        poem = ''.join([str(i) for i in poem])
        print("generated poem:")
        # Print the results.
        print("{:30}".format(poem))

    print('')
    print('')


def unsupervised_generation(D, n_states, N_iters, k, sylls, endsylls, rhymedict):
    '''
    Trains an HMM using unsupervised learning on the poems and then calls
    generations() to generate k emissions for each HMM, processing the emissions
    and printing them as strings.

    Arguments:
        ps: the list of poems, where each poem is a list of integers
            representing the tokens (words) of the poem.
        D: the number of "words" contained in ps.
        n_states: number of hidden states that the HMM should have.
        N_iters: the number of iterations the HMM should train for.
        k: the number of generations for each HMM.
    '''
    # simply tells us that we are running unsupervised learning. this isn't
    # necessary, but is nice for now.
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Generating Emissions From HMM with %d States") %n_states)
    print('#' * 70)
    print('')
    print('')
    rhymefile = str(n_states)+'_'+str(N_iters)+'_1.txt'
    regularfile = str(n_states)+'_'+str(N_iters)+'_2.txt'

    A, O = processing.read_saved_HMM(rhymefile)
    HMMrhyme = HiddenMarkovModel(A, O)
    A, O = processing.read_saved_HMM(regularfile)
    HMMreg= HiddenMarkovModel(A, O)
    # generates and prints "poems"
    print("RHYMING!!")
    generations(HMMrhyme, k, sylls, endsylls, True, rhymedict, True)
    print("10 SYLLABLES!!")
    generations(HMMrhyme, k, sylls, endsylls, True, rhymedict, False)
    print("REGULAR")
    generations(HMMrhyme, k, sylls, endsylls, False, rhymedict, False)


def train_HMM(ps, D, n_states, N_iters):
    '''
    Trains an HMM using unsupervised learning on the poems and then calls
    generations() to generate k emissions for each HMM, processing the emissions
    and printing them as strings.

    Arguments:
        ps: the list of poems, where each poem is a list of integers
            representing the tokens (words) of the poem.
        D: the number of "words" contained in ps.
        n_states: number of hidden states that the HMM should have.
        N_iters: the number of iterations the HMM should train for.
        k: the number of generations for each HMM.
    '''
    # simply tells us that we are running unsupervised learning. this isn't
    # necessary, but is nice for now.
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Unsupervised Learning With %d States") %n_states)
    print('#' * 70)
    print('')
    print('')

    # Train the HMM.
    HMM = unsupervised_HMM(ps, D, n_states, N_iters)
    # Save the HMM in a text file
    HMM.save_HMM(n_states, N_iters, 2)

    for p in ps:
        p.reverse()
        p.insert(0,0)
        p.pop()

    HMM = unsupervised_HMM(ps, D, n_states, N_iters)
    HMM.save_HMM(n_states, N_iters, 1)

if __name__ == '__main__':
    # reads in the shakespeare poems.
    (ps, pns, wtn, ntw, rdict) = processing.readshakes()
    # reads and stores the information in the syllable text file
    (sylls, endsylls) = processing.readsylls(wtn)
    k = 5 # number of poems to generate for each model.
    # We have already trained and stored our models, so we don't need to keep
    # retraining them.
    training = False
    # We want to generate poems from our stored models now.
    generating = True
    n_states = [16, 14, 12, 10, 8]
    for n in n_states:
        # Trains the HMM
        if (training):
            train_HMM(ps, len(wtn), n, 400)
        # Uses a stored, trained HMM to generate different kinds of emissions
        # (Rhyming, 10 syllables per line, and simply 14 lines)
        if (generating):
            unsupervised_generation(len(wtn), n, 400, k, sylls, endsylls, rdict)
