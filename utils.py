from __future__ import unicode_literals, print_function, division
from io import open

import numpy as np
import unicodedata
import torch
import re
import random
import time
import math

import matplotlib.pyplot as plt
from lang import Lang

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    sentences = [[normalizeString(s) for s in l.split('\t')[:1]] for l in lines]
    lang = Lang(lang1)

    return lang, sentences


def filterSentence(sentence, max_length, prefix=True, min_length=None):
    if min_length is None:
        return len(sentence.split(' ')) < max_length and \
            (sentence.startswith(eng_prefixes) if prefix else True)
    else:
        return len(sentence.split(' ')) >= min_length and \
            len(sentence.split(' ')) < max_length and \
            (sentence.startswith(eng_prefixes) if prefix else True)


def filterSentences(sentences, max_length, prefix=True, min_length=None):
    return [sentence[0] for sentence in sentences if
            filterSentence(sentence[0], max_length, prefix=prefix, min_length=min_length)]


def prepareData(lang1, lang2, max_length, prefix=True, min_length=None):
    lang, sentences = readLangs(lang1, lang2)
    print("Read %s sentences" % len(sentences))
    sentences = filterSentences(sentences, max_length, prefix=prefix,
                                min_length=min_length)
    print("Trimmed to %s sentences" % len(sentences))
    print("Counting words...")
    for sentence in sentences:
        lang.addSentence(sentence)
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, sentences


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plot_loss(axs, train_loss, valid_loss, plot_freq=1, show=False, save=True, path=None):
    # Training (NLL) Loss plot
    x = (np.arange(len(train_loss)) * plot_freq).tolist()
    axs.clear()
    axs.plot(x, train_loss, color='red',
             label='Train Loss')
    axs.plot(x, valid_loss, color='blue', label='Validation Loss')
    axs.set(title='Learning Curves')
    axs.set(ylabel='NLLL')
    axs.set(xlabel='Iterations')
    axs.legend(loc='upper right')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)
