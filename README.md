# text_compression_ae
Vanilla Auto Encoders with Attention implementation in PyTorch for compressing text documents

## Installation

[Poetry](https://python-poetry.org/) is used for dependency management. So please install poetry:

```bash
$ curl -sSL https://install.python-poetry.org | python3 -

```

To install all the dependencies, please enter the following from the project's root directory:

```bash
$ poetry install

```

Then, to enter the virtual environment:

```bash
$ poetry shell

```

To train a model:

```bash
$ python train.py --params_path PATH/TO/JSON/FILE/CONTAINING/MODEL/AND/TRAINING/HYPERPARAMETERS

```

For evaluating a trained policy:

```bash
$ python evaluate.py -model_path PATH/TO/FOLDER/CONTAINING/TRAINED/MODEL

```
