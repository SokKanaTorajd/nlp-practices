from nlp_id.stopword import StopWord
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import numpy as np


def create_dataset(file_path):
    stopword = StopWord()
    sentences = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(int(row[0]))
            text = row[1].lower()
            sentence = stopword.remove_stopword(text)
            sentences.append(sentence)

    training_size = int(0.8 * len(sentences))

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    print(f'training labels length: {len(training_labels)}')
    print(f'training sentences length: {len(training_sentences)}')
    print(f'testing labels length: {len(testing_labels)}')
    print(f'testing sentences length: {len(testing_sentences)}')

    return training_sentences, testing_sentences, training_labels, testing_labels

def create_model(x_train, x_test, y_train, y_test):

    oov_tok = "<OOV>"
    max_length = 120
    padding_type = 'post'
    trunc_type = 'post'
    embedding_dim = 16

    tokenizer = Tokenizer(num_words=1000, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    vocab_size = len(tokenizer.word_index)+1

    training_sequences = tokenizer.texts_to_sequences(x_train)
    training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                    padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(x_test)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                   padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(y_train)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(y_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=max_length),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.SpatialDropout1D(0.25),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.fit(training_padded, training_labels,
              epochs=30, verbose=2,
              validation_data=(testing_padded, testing_labels))
    return model




if __name__ == '__main__':
    dataset_dir = '/Users/fttiunjani/gemastik/practice/dataset/dataset-idsa-master/labelled-sentiment-copy.csv'
    training_sentences, testing_sentences, training_labels, testing_labels = create_dataset(dataset_dir)
    model = create_model(training_sentences, testing_sentences,
                         training_labels, testing_labels)
    model.save('sentiment-model.h5')

# print(df.info())


