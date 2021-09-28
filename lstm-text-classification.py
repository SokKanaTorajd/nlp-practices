from nlp_id.stopword import StopWord as stopword
# import tensorflow as tf
import csv


def create_dataset(file_path):
    sentences = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1].lower()
            sentence = stopword.remove_stopword(sentence)
            sentences.append(sentence)

    print(f'labels length: {len(labels)}')
    print(f'sentences length: {len(sentences)}')

    training_size = int(0.8 * len(sentences))

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    return training_sentences, testing_sentences, training_labels, testing_labels


dataset_dir = 'dataset/dataset-idsa-master/labelled-sentiment-copy.csv'

training_sentences, testing_sentences, training_labels, testing_labels = create_dataset(dataset_dir)

# print(df.info())


