from __future__ import unicode_literals, print_function, division
import random

import numpy as np
# import os
import torch
import argparse
import json

# from datetime import datetime
# from shutil import rmtree
#
# from torch import optim

from ae import EncoderRNN, DecoderRNN, AttnDecoderRNN, evaluateRandomly

from utils import prepareData


def main(args):
    # Load env and model parameters
    with open(args.model_path + 'params/params.dat') as pf:
        params = json.load(pf)

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")

    lang, sentences, max_length = prepareData(
        params['data_path'], params['max_length'],  min_length=params['min_length'],
        char=params['char'])

    encoder = EncoderRNN(
        (lang.n_chars if params['char'] else lang.n_words), params['hidden_1_size'],
        params['hidden_2_size'], device, dropout_p=params['encoder_dropout']).to(device)

    if params['attention']:
        decoder = AttnDecoderRNN(
            params['hidden_2_size'], params['hidden_1_size'],
            (lang.n_chars if params['char'] else lang.n_words),
            (max_length if params['max_length'] is None else params['max_length']),
            device, dropout_p=params['decoder_dropout']).to(device)
    else:
        decoder = DecoderRNN(
            params['hidden_2_size'], params['hidden_1_size'],
            (lang.n_chars if params['char'] else lang.n_words), device).to(device)

    # Load saved model
    encoder.load_state_dict(torch.load(args.model_path + 'models/encoder.pth'))
    decoder.load_state_dict(torch.load(args.model_path + 'models/decoder.pth'))

    evaluateRandomly(encoder, decoder, lang, sentences, max_length, device,
                     char=params['char'], n=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expert Policy Distillation')
    parser.add_argument('--char', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='seq model at individual character level (default: False)')
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
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model file (default: None)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU (default: None)')
    parser.add_argument('--plot', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
