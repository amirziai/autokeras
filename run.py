from model.keras import Keras
from encode.text import TextEncoder

import config

import random

SEARCH_TYPE = config.search['type']
SEARCH_TRIALS = config.search['trials']
ENCODINGS = config.encodings
ARCHITECTURES = config.architectures

if __name__ == '__main__':
    configurations = [
        (encoding, architecture)
        for encoding in ENCODINGS
        for architecture in ARCHITECTURES
    ]
    random.shuffle(configurations)
    configurations_to_explore = configurations[len(configurations) if SEARCH_TYPE == 'random' else SEARCH_TRIALS]

    for encoding, architecture in configurations_to_explore:
        pass
