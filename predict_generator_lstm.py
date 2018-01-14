import os

import numpy as np
from keras.models import load_model

from train_generator_lstm import Vectorizer

if __name__ == '__main__':
    model = load_model(os.path.join('data', 'name_generator_model.h5'))

    vectorizer = Vectorizer()

    strings_to_try = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N'
    ]

    for s in strings_to_try:
        generated_string = s
        print('initial string: "{}"'.format(s))
        for i in range(4):
            vectors = [vectorizer.vectorize_character(c) for c in generated_string]
            vectors = np.array([vectors])
            prediction = model.predict(vectors)
            prediction = prediction[0][-1]
            max_index = np.argmax(prediction)
            predicted_character = vectorizer.index_to_character.get(max_index, ' ')
            generated_string += predicted_character
        print('prediction: "{}"'.format(generated_string))
