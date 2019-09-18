from __future__ import absolute_import, division, print_function, unicode_literals

import dataset as dataset
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1

tf.compat.v1.enable_eager_execution()

import numpy as np
import os
import time


class Shakespeare:

    sequence = DatasetV1

    def explore_dataset(self):
        path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                               'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        text = open(path_to_file, 'rb').read().decode('utf-8')
        print('Length of text: {} characters'.format(len(text)))
        print(text[:250])

        # The unique character in the file
        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])
        print('{')
        for char,_ in zip(char2idx, range(20)):
            print(' {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('   ...\n}')
        print('{} ------- characters map to int -----> {}'.format(repr(text[:13]), text_as_int[:13]))

        # The maximum length sentence we want for a single input in character
        seq_length = 100
        examples_per_epoch = len(text)//seq_length

        # Create training examples / target
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        self.sequence = char_dataset.batch(seq_length+1, True)

        for item in self.sequence.take(1):
            print(repr(''.join(idx2char[item.numpy()])))

    def split_input_target(self, chunk):
        input_text = chunk[:-1]
        target_size = chunk[1:]
        return input_text, target_size

    dataset = sequence.map(split_input_target)

    #for input_example, target_example in dataset.take(1):
    #    print('Input data: ', repr(''.join(i)))


Shakespeare().explore_dataset()
