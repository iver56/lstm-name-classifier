import io
import os


def read_file_lines(file_path):
    with io.open(file_path, 'r', encoding='utf8') as input_file:
        lines = input_file.readlines()

    return [line.strip() for line in lines]


boy_names = read_file_lines(os.path.join('data', 'norwegian_male_names.txt'))
girl_names = read_file_lines(os.path.join('data', 'norwegian_female_names.txt'))

print('We have {} boy names and {} girl names'.format(len(boy_names), len(girl_names)))
print('Example names: "{}" and "{}"'.format(boy_names[40], girl_names[300]))

max_name_length = max(len(name) for name in boy_names + girl_names)
print('The longest name in the dataset consists of {} characters'.format(max_name_length))

characters = set(''.join(boy_names)).union(set(''.join(girl_names)))
num_characters = len(characters)
ordered_characters = sorted(list(characters))

print('We have {} unique characters in the dataset:'.format(num_characters))
print(ordered_characters)

character_to_index = {
    character: index for index, character in enumerate(ordered_characters)
}
print('Here is a mapping from character to index:')
print(character_to_index)


def vectorize_character(c):
    vector = [0] * num_characters
    character_index = character_to_index.get(c, None)
    if character_index is not None:
        vector[character_index] = 1
    return vector


print('The vector for character "B" looks like this:')
print(vectorize_character('B'))


def vectorize_string(s):
    vectors = []
    for c in s.upper():
        vector = vectorize_character(c)
        vectors.append(vector)
    return vectors


print('"ARE" vectorized looks like this:')
example_string_vectorized = vectorize_string('ARE')
for character_vector in example_string_vectorized:
    print(character_vector)


def preprocess_strings(strings):
    # Right pad names so that all names are of equal length
    return [s.ljust(max_name_length) for s in strings]


padded_boy_names = preprocess_strings(boy_names)
padded_girl_names = preprocess_strings(girl_names)

print('Padded strings look like this:')
print(padded_boy_names[40:42])

boy_vectors = [vectorize_string(name) for name in padded_boy_names]
girl_vectors = [vectorize_string(name) for name in padded_girl_names]

x = boy_vectors + girl_vectors
y = [0 for _ in boy_vectors] + [1 for _ in girl_vectors]

num_examples = len(x)

# Convert the data into numpy arrays, for efficient processing in the machine learning library
import numpy as np

x = np.array(x).reshape((num_examples, max_name_length, num_characters))
y = np.array(y).reshape((num_examples, 1))


from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

model = Sequential()
model.add(LSTM(units=64, input_shape=(None, num_characters), return_sequences=True))
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

model.save(os.path.join('data', 'name_model.h5'))

