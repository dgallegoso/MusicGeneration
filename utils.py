from midi import encode_midi, decode_midi
import numpy as np
import os


DATA_DIR = 'dev'
TEST_DIR = 'test'
CACHE_DIR = 'cache'


def generate_dataset(dir, timesteps):
    if dir == 'dev':
        directory = DATA_DIR
    else:
        directory = TEST_DIR
    dataset = []
    for filename in os.listdir(directory):
        decoding = decode_midi('{}/{}'.format(DATA_DIR, filename))
        for subset in range(decoding.shape[0] / timesteps):
            subDecoding = decoding[subset * timesteps:]
            if subDecoding.shape[0] < timesteps:
                continue
            dec = subDecoding[:timesteps]
            dataset.append(dec.copy().T)
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
