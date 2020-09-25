import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pickle

import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords

stp = stopwords.words('english')

np.random.seed(0)

MAX_WORDS = 8638
MAX_SEQUENCE_LENGTH = 100
OUTPUT_DIM = 50
label_mapping = {'ham': 0, 'spam': 1}
inverse_mapping = {0: 'ham', 1: 'spam'}

with open('./models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('./models/model.h5')


def predict_label(sentence):
    # cleaning sentence
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]+', ' ', sentence)
    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(seq).round()
    prediction = np.argmax(prediction)
    return inverse_mapping[prediction]


class Pipeline:

    def __init__(self):
        self.dataset = pd.read_csv('./dataset/spam.csv', encoding="ISO-8859-1")
        self.tokenizer = Tokenizer(num_words=MAX_WORDS)

    def preprocess(self):
        print('Preprocessing Dataset\n')
        dataset = self.dataset[['v2', 'v1']]
        dataset.columns = ['features', 'target']
        dataset['features'] = dataset['features'].apply(lambda x: x.lower())
        dataset['features'] = dataset['features'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stp]))
        dataset['features'] = dataset['features'].apply(lambda x: re.sub(r'[^\w\s]+', ' ', x))
        dataset['features'] = dataset['features'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ', x))
        ham = dataset[dataset['target'] == 'ham']
        spam = dataset[dataset['target'] == 'spam'].sample(len(ham), replace=True)
        dataset = pd.concat([ham,spam])
        dataset = shuffle(dataset)
        X_train, X_test, y_train, y_test = train_test_split(dataset['features'], dataset['target'], test_size=0.2)
        return X_train, X_test, y_train, y_test

    def create_features(self):
        print('creating features\n')
        X_train, X_test, y_train, y_test = self.preprocess()
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        # saving model
        with open('./models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)

        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        y_train = np.array([label_mapping[label] for label in y_train])
        y_test = np.array([label_mapping[label] for label in y_test])

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return (X_train, y_train, X_test, y_test), self.tokenizer

    def train_and_save_model(self):
        print('Training model\n')
        (X_train, y_train, X_test, y_test), _ = self.create_features()
        model = keras.models.Sequential([
            keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=OUTPUT_DIM, input_length=MAX_SEQUENCE_LENGTH),
            keras.layers.Flatten(),
            keras.layers.Dense(units=2, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model.fit(X_train, y_train, batch_size=15, epochs=5, validation_data=(X_test, y_test),
                  callbacks=[callback])

        print('Accuracy of model = {}\n'.format(model.evaluate(X_test, y_test)[1]))
        print('saving model\n')
        model.save('./models/model.h5')
        return 'Model Trained and saved Successfully.'


if __name__ == '__main__':
    pipe = Pipeline()
    pipe.train_and_save_model()
