from __future__ import unicode_literals, print_function, division
# from io import open
# import unicodedata
# import string
# import re
import random

import numpy as np
import os
import torch
import argparse
# import json

from datetime import datetime
from shutil import rmtree

# import time
# import math

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

from ae import EncoderRNN, AttnDecoderRNN, trainIters

from utils import prepareData, evaluateRandomly

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create a timestamp directory to save model, parameter and log files
    training_dir = \
        ('training/' + str(datetime.now().date()) + '_' +
         str(datetime.now().hour).zfill(2) + '-' + str(datetime.now().minute).zfill(2) +
         '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(training_dir):
        rmtree(training_dir)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(training_dir)
    os.makedirs(training_dir + 'plots')
    os.makedirs(training_dir + 'models')
    os.makedirs(training_dir + 'params')

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', args.max_length, True)
    print(random.choice(pairs))

    encoder1 = EncoderRNN(input_lang.n_words, args.hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(args.hidden_size, output_lang.n_words, args.max_length,
                                   device, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs,
               n_iters=args.n_iters, device=device, max_length=args.max_length,
               teacher_ratio=args.teacher_ratio, print_every=100, plot_show=args.plot,
               path=training_dir)

    evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, pairs, args.max_length, device, n=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expert Policy Distillation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden representation size (default: 256)')
    parser.add_argument('--max_length', type=int, default=10,
                        help='maximum sequence length (default: 10)')
    parser.add_argument('--teacher_ratio', type=float, default=0.5,
                        help='teacher forcing ratio for decoder input (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for optimizer (default: 0.01)')
    parser.add_argument('--n_iters', type=int, default=75000,
                        help='number of training iterations (default: 75000)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data file (default: None)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU (default: None)')
    parser.add_argument('--plot', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
