from __future__ import unicode_literals, print_function, division
from io import open

import numpy as np
import unicodedata
import torch
import re
import time
import math

import matplotlib.pyplot as plt
from lang import Lang
from char import Char

SOS_token = 0
EOS_token = 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s, ascii=False):
    if ascii:
        # Lowercase, trim, and remove non-letter characters
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    else:
        s = s.lower()
    return s


def filterWordSentence(sentence, max_length=None, prefix=True, min_length=None):
    if min_length is None:
        return len(sentence.split(' ')) < max_length
    else:
        return len(sentence.split(' ')) >= min_length and \
            len(sentence.split(' ')) < max_length


def filterCharSentence(sentence, max_length=None, prefix=True, min_length=None):
    if min_length is None:
        return len(sentence) < max_length
    else:
        return len(sentence) >= min_length and len(sentence) < max_length


def filterSentences(sentences, max_length=None, prefix=True, min_length=None, char=True):
    if char:
        return [sentence for sentence in sentences if filterCharSentence(
            sentence, max_length, prefix=prefix, min_length=min_length)]
    else:
        return [sentence for sentence in sentences if filterWordSentence(
            sentence, max_length, prefix=prefix, min_length=min_length)]


def prepareData(data_path, max_length=None, min_length=None, char=True, ascii=False):
    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    # Split every line into individual review and normalize
    sentences = [' '.join([normalizeString(s, ascii=ascii) for s in l.split(' ')[1:]])
                 for l in lines]

    review_lens = list()

    if char:
        lang = Char('eng')
        for sentence in sentences:
            review_lens.append(len(sentence))
    else:
        lang = Lang('eng')
        for sentence in sentences:
            review_lens.append(len(sentence.split(' ')))

    max_sentence_length = max(review_lens)
    min_sentence_length = min(review_lens)

    max_length = (max_sentence_length if max_length is None else max_length)

    # bins = np.arange(min_sentence_length, max_sentence_length, 1)
    # plt.xlim([min_sentence_length - 5, max_sentence_length + 5])
    # plt.hist(review_lens, bins=bins, alpha=0.5)
    # plt.show()
    # input()

    print(f"Min Sentence Length (chars): {min_sentence_length}")
    print(f"Max Sentence Length (chars): {max_sentence_length}")

    print("Read %s sentences" % len(sentences))
    sentences = filterSentences(sentences, max_length, min_length=min_length, char=char)

    print("Trimmed to %s sentences" % len(sentences))
    for sentence in sentences:
        lang.addSentence(sentence)

    if char:
        print("Counting characters...")
        print("Counted characters:")
        print(lang.name, lang.n_chars)
    else:
        print("Counting words...")
        print("Counted words:")
        print(lang.name, lang.n_words)

    return lang, sentences, max_length


def indexesFromSentence(lang, sentence, char=True):
    if char:
        return [lang.char2index[character] for character in sentence]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromText(lang, sentence, device, char=True):
    indexes = indexesFromSentence(lang, sentence, char)
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


def plot_loss(axs, train_loss, valid_loss, plot_freq=1, show=False, save=True, path=None,
              plot_text=None):
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

    if plot_text is not None:
        x_min = axs.get_xlim()[0]
        y_max = axs.get_ylim()[1]
        axs.text(x_min * 1.0, y_max * 1.01, plot_text, fontsize=14, color='Black')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

        with open(path + 'plots/train_loss.npy', 'wb') as f:
            np.save(f, np.array(train_loss))

        with open(path + 'plots/valid_loss.npy', 'wb') as f:
            np.save(f, np.array(valid_loss))

    if show:
        plt.show(block=False)
        plt.pause(0.01)
