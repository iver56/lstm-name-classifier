# -*- coding: utf-8 -*-

import os

import numpy as np
from keras.models import load_model

from train_lstm import Vectorizer


def print_meter(that_string, prediction, meter_width=50, label_left='boy', label_right='girl'):
    """
    :param that_string:
    :param prediction: float between 0 and 1
    :param meter_width: int
    :param label_left:
    :param label_right

    Example output:
    Iben            boy ______________________|___________________________ girl
    """
    meter_string = list('_' * meter_width)
    meter_index = min(int(round(prediction * meter_width)), meter_width - 1)
    meter_string[meter_index] = '|'
    print(
        '{:<15} {} {} {}'.format(
            that_string, label_left, ''.join(meter_string), label_right
        )
    )


if __name__ == '__main__':
    model = load_model(Vectorizer.MODEL_FILE_PATH)

    vectorizer = Vectorizer(mode='prediction')


    def predict_and_visualize(that_string):
        that_string = vectorizer.pad_strings([that_string], vectorizer.max_string_length)[0]
        vector = vectorizer.vectorize_string(that_string)
        vectors = np.array([vector])
        prediction = model.predict(vectors)
        prediction = prediction[0][0]
        print_meter(that_string, prediction)


    strings_to_try = vectorizer.read_file_lines(os.path.join('data', 'hots_female_names.txt'))
    for s in strings_to_try:
        predict_and_visualize(s)

    while True:
        new_string = input('Type a name: ')

        if not new_string:
            break

        if len(new_string) > vectorizer.max_string_length:
            new_string = new_string[:vectorizer.max_string_length]
            print('Note: The string was truncated to {}'.format(new_string))

        predict_and_visualize(new_string)
