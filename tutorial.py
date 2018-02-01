import io
import os


def read_file_lines(file_path):
    with io.open(file_path, 'r', encoding='utf8') as input_file:
        lines = input_file.readlines()

    return [line.strip() for line in lines]


boy_names = read_file_lines(os.path.join('data', 'norwegian_male_names.txt'))
girl_names = read_file_lines(os.path.join('data', 'norwegian_female_names.txt'))

print(
    'We have {} boy names and {} girl names'.format(len(boy_names), len(girl_names))
)
print('Example names: "{}" and "{}"'.format(boy_names[40], girl_names[300]))

max_name_length = max(len(name) for name in boy_names + girl_names)
print('The longest name in the dataset has {} characters'.format(max_name_length))

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


def pad_strings(strings):
    # Right pad names so that all names are of equal length
    return [s.ljust(max_name_length) for s in strings]


padded_boy_names = pad_strings(boy_names)
padded_girl_names = pad_strings(girl_names)

print('Padded strings look like this:')
print(padded_boy_names[40:42])

boy_vectors = [vectorize_string(name) for name in padded_boy_names]
girl_vectors = [vectorize_string(name) for name in padded_girl_names]

x = boy_vectors + girl_vectors
y = [0 for _ in boy_vectors] + [1 for _ in girl_vectors]

num_examples = len(x)
target_vector_size = 1  # the number of entries in each target vector

# Convert the data into numpy arrays, because that is the format Keras expects
import numpy as np

x = np.array(x).reshape((num_examples, max_name_length, num_characters))
y = np.array(y).reshape((num_examples, target_vector_size))
print('x shape: {}'.format(x.shape))
print('y shape: {}'.format(y.shape))

# Import Keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

# Set up structure of the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(None, num_characters), return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(units=64))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x, y, epochs=200, batch_size=128)

# Use the model for prediction
while True:
    new_string = input('Type a name: ')

    if not new_string:
        break

    if len(new_string) > max_name_length:
        new_string = new_string[:max_name_length]
        print('Note: The string was truncated to {}'.format(new_string))

    new_string = pad_strings([new_string])[0]
    vectors = [vectorize_string(new_string)]

    prediction = model.predict(np.array(vectors))
    prediction = prediction[0][0]
    print('Prediction: {:.2f}'.format(prediction))
