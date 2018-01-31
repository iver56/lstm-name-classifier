import os

import numpy as np
from keras.models import load_model

from train_lstm import Vectorizer

if __name__ == '__main__':
    model = load_model(os.path.join('data', 'name_model.h5'))

    vectorizer = Vectorizer()

    strings_to_try = [
        'henrik',
        'andrea',
        'åshild',
    ]
    #strings_to_try = vectorizer.read_file_lines(os.path.join('data', 'hots_female_names.txt'))
    strings_to_try = vectorizer.preprocess_strings(strings_to_try, vectorizer.max_string_length)
    for s in strings_to_try:
        vector = vectorizer.vectorize_string(s)
        vector = np.array([vector])
        prediction = model.predict(vector)
        prediction = prediction[0][0]
        prediction_label = 'boy' if prediction < 0.5 else 'girl'
        prediction_confidence = 1 - prediction if prediction_label == 'boy' else prediction
        print(
            '{0:<20} {1:.1f}% {2}'.format(
                s, prediction_confidence * 100, prediction_label
            )
        )
