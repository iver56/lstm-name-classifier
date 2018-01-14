import os
import random
import io

import numpy as np
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


class Vectorizer(object):
    def __init__(self):
        self.boy_names = self.read_file_lines(os.path.join('data', 'boy_names.txt'))
        self.girl_names = self.read_file_lines(os.path.join('data', 'girl_names.txt'))

        #print('{} boy names, {} girl names'.format(len(self.boy_names), len(self.girl_names)))

        self.characters = set(
            ' '.join(self.boy_names)
        ).union(
            set(' '.join(self.girl_names))
        ).union(
            {chr(3)}  # "End of text" (ETX) character
        )
        self.ordered_characters = sorted(list(self.characters))
        self.num_characters = len(self.ordered_characters)
        self.character_to_index = {
            character: index for index, character in enumerate(self.ordered_characters)
        }
        self.index_to_character = {
            index: character for index, character in enumerate(self.ordered_characters)
        }

    def train_model(self):
        string_length = 5
        boy_names = [name for name in self.boy_names if len(name) == 5]
        x = []
        y = []
        for name in boy_names:
            input_vectors, output_vectors = self.vectorize_string(name)
            x.append(input_vectors)
            y.append(output_vectors)

        num_examples = len(x)

        random.seed(42)

        x = np.array(x).reshape((num_examples, string_length - 1, self.num_characters))
        y = np.array(y).reshape((num_examples, string_length - 1, self.num_characters))

        # Create model
        model = Sequential()
        model.add(LSTM(units=64, input_dim=self.num_characters, return_sequences=True))
        model.add(Activation('relu'))
        model.add(LSTM(units=64, input_dim=self.num_characters, return_sequences=True))
        model.add(Activation('relu'))
        model.add(Dense(self.num_characters, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        print(model.summary())

        # Fit model
        model.fit(x, y, epochs=2000, batch_size=128)

        # Final evaluation of the model
        scores = model.evaluate(x, y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        model.save(os.path.join('data', 'name_generator_model.h5'))

    @staticmethod
    def read_file_lines(file_path):
        with io.open(file_path, 'r', encoding='utf8') as input_file:
            lines = input_file.readlines()

        return [line.strip() for line in lines]

    def vectorize_character(self, character):
        vector = [0] * self.num_characters
        character_index = self.character_to_index.get(character, None)
        if character_index is not None:
            vector[character_index] = 1
        return vector

    def vectorize_string(self, s):
        # used for generating training data
        input_vectors = []
        output_vectors = []
        s_upper = s.upper()
        for i in range(len(s_upper) - 1):
            character = s_upper[i]
            input_vector = self.vectorize_character(character)
            input_vectors.append(input_vector)

            next_character = s_upper[i + 1]
            output_vector = self.vectorize_character(character)
            output_vectors.append(output_vector)

        return input_vectors, output_vectors


if __name__ == '__main__':
    Vectorizer().train_model()
