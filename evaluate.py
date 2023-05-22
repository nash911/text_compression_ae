from __future__ import unicode_literals, print_function, division
import random

import numpy as np
import torch
import argparse
import json


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
        char=params['char'], ascii=params['ascii'])

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
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model file (default: None)')
    args = parser.parse_args()

    main(args)
