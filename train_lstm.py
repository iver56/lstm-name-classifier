import io
import json
import os
import random

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


class Vectorizer(object):
    METADATA_FILE_PATH = os.path.join('data', 'metadata.json')
    MODEL_FILE_PATH = os.path.join('data', 'name_model.h5')

    def __init__(self, mode='training'):
        if mode == 'training':
            self.boy_names = self.read_file_lines(os.path.join('data', 'norwegian_male_names.txt'))
            self.girl_names = self.read_file_lines(os.path.join('data', 'norwegian_female_names.txt'))
            characters = set(''.join(self.boy_names)).union(set(''.join(self.girl_names)))
            ordered_characters = sorted(list(characters))
            self.character_to_index = {
                character: index for index, character in enumerate(ordered_characters)
            }
            self.max_string_length = max(len(name) for name in self.boy_names + self.girl_names)

            with open(self.METADATA_FILE_PATH, 'w') as outfile:
                json.dump(
                    {
                        'character_to_index': self.character_to_index,
                        'max_string_length': self.max_string_length
                    },
                    outfile
                )
        else:
            # mode == 'prediction'
            with open(self.METADATA_FILE_PATH) as data_file:
                metadata = json.load(data_file)
            self.character_to_index = metadata['character_to_index']
            self.max_string_length = metadata['max_string_length']

        self.num_characters = len(self.character_to_index)

    def train_model(self):
        boy_names = self.pad_strings(self.boy_names, self.max_string_length)
        girl_names = self.pad_strings(self.girl_names, self.max_string_length)

        boy_vectors = [self.vectorize_string(name) for name in boy_names]
        girl_vectors = [self.vectorize_string(name) for name in girl_names]

        x = boy_vectors + girl_vectors
        y = [0 for _ in boy_vectors] + [1 for _ in girl_vectors]

        num_examples = len(x)

        random.seed(42)

        x = np.array(x).reshape((num_examples, self.max_string_length, self.num_characters))
        y = np.array(y).reshape((num_examples, 1))

        # Create model
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(None, self.num_characters), return_sequences=True))
        model.add(Activation('relu'))
        model.add(LSTM(units=64))
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        print(model.summary())

        # Fit model
        model.fit(x, y, epochs=200, batch_size=128)

        # Final evaluation of the model
        scores = model.evaluate(x, y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        model.save(self.MODEL_FILE_PATH)

    @staticmethod
    def read_file_lines(file_path):
        with io.open(file_path, 'r', encoding='utf8') as input_file:
            lines = input_file.readlines()

        return [line.strip() for line in lines]

    @staticmethod
    def pad_strings(strings, max_length):
        # Right pad names
        return [s.ljust(max_length) for s in strings]

    def vectorize_string(self, s):
        vectors = []
        for c in s.upper():
            vector = [0] * self.num_characters
            character_index = self.character_to_index.get(c, None)
            if character_index is not None:
                vector[character_index] = 1
            vectors.append(vector)
        return vectors


if __name__ == '__main__':
    Vectorizer().train_model()
