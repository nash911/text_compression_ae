from __future__ import unicode_literals, print_function, division
import random

import numpy as np
import os
import torch
import argparse
import json

from datetime import datetime
from shutil import rmtree

from torch import optim

from ae import EncoderRNN, DecoderRNN, AttnDecoderRNN, trainIters

from utils import prepareData


def main(args):
    if args.params_path is not None:
        with open(args.params_path) as pf:
            params = json.load(pf)

        seed = params['seed']
        gpu = params['gpu']
        data_path = params['data_path']
        max_length = params['max_length']
        min_length = params['min_length']
        char = params['char']
        ascii = params['ascii']
        hidden_1_size = params['hidden_1_size']
        hidden_2_size = params['hidden_2_size']
        encoder_dropout = params['encoder_dropout']
        decoder_dropout = params['decoder_dropout']
        attention = params['attention']
        lr = params['lr']
        n_iters = params['n_iters']
        teacher_ratio = params['teacher_ratio']
        plot = params['plot']
    else:
        seed = args.seed
        gpu = args.gpu
        data_path = args.data_path
        max_length = args.max_length
        min_length = args.min_length
        char = args.char
        ascii = args.ascii
        hidden_1_size = args.hidden_1_size
        hidden_2_size = args.hidden_2_size
        encoder_dropout = args.encoder_dropout
        decoder_dropout = args.decoder_dropout
        attention = args.attention
        lr = args.lr
        n_iters = args.n_iters
        teacher_ratio = args.teacher_ratio
        plot = args.plot

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    # Dump params to file
    with open(training_dir + 'params/params.dat', 'w') as jf:
        json.dump(vars(args), jf, indent=4)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    lang, sentences, max_length = prepareData(
        data_path, max_length, min_length=min_length, char=char, ascii=ascii)
    print(random.choice(sentences))
    if char:
        print(f"Unique characters:\n{lang.char2index}")

    encoder = EncoderRNN((lang.n_chars if char else lang.n_words), hidden_1_size,
                         hidden_2_size, device, dropout_p=encoder_dropout).to(device)

    if attention:
        decoder = AttnDecoderRNN(hidden_2_size, hidden_1_size,
                                 (lang.n_chars if char else lang.n_words),
                                 (max_length if max_length is None else max_length),
                                 device, dropout_p=decoder_dropout).to(device)
    else:
        decoder = DecoderRNN(hidden_2_size, hidden_1_size,
                             (lang.n_chars if char else lang.n_words), device).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    trainIters(
        encoder, decoder, lang, sentences, encoder_optimizer, decoder_optimizer,
        n_iters=n_iters, device=device, path=training_dir, print_plot_every=100,
        max_length=(max_length if max_length is None else max_length),
        teacher_ratio=teacher_ratio, eval_every=5000, char=char,
        plot_show=plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expert Policy Distillation')
    parser.add_argument('--char', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='seq model at individual character level (default: False)')
    parser.add_argument('--ascii', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='convert Unicode to ASCII (default: False)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--hidden_1_size', type=int, default=256,
                        help='hidden layer 1 size (default: 256)')
    parser.add_argument('--hidden_2_size', type=int, default=128,
                        help='hidden layer 2 size (default: 128)')
    parser.add_argument('--attention', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='attention mechanism for decoder (default: False)')
    parser.add_argument('--min_length', type=int, default=None,
                        help='minimum sequence length of input (default: None)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='maximum sequence length (default: None)')
    parser.add_argument('--encoder_dropout', type=float, default=0.1,
                        help='encoder dropout probability (default: 0.1)')
    parser.add_argument('--decoder_dropout', type=float, default=0.1,
                        help='decoder dropout probability (default: 0.1)')
    parser.add_argument('--teacher_ratio', type=float, default=0.5,
                        help='teacher forcing ratio for decoder input (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--n_iters', type=int, default=100000,
                        help='number of training iterations (default: 75000)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data file (default: None)')
    parser.add_argument('--params_path', type=str, default=None,
                        help='Path to file containing model/training hyperparameters ' +
                             '(default: None)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU (default: None)')
    parser.add_argument('--plot', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
