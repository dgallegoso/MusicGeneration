from midi import encode_midi, decode_midi
import numpy as np
import os


DATA_DIR = 'dev'
CACHE_DIR = 'cache'


def generate_dataset():
    dataset = []
    for filename in os.listdir(DATA_DIR):
        dataset.append(decode_midi('{}/{}'.format(DATA_DIR, filename)))
    return dataset


def generate_dataset_iterator():
    for filename in os.listdir(DATA_DIR):
        yield decode_midi('{}/{}'.format(DATA_DIR, filename))

def save_decoding(decoding, filename):
    encode_midi(decoding, filename)


def main():
    timesteps = float('inf')
    for x in generate_dataset_iterator():
        timesteps = min(timesteps, x.shape[0])
        print timesteps, x.shape[0]

if __name__== "__main__":
  main()
