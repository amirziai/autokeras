import config
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

GLOVE_DIR = config.encodings_config['glove_directory']


class TextEncoder:
    def _create_embedding_index(self):
        with open(os.path.join(GLOVE_DIR), 'glove.6B.{}.d.txt'.format(self.glove_dimension)) as file_handle:
            for line in file_handle:
                values = line.split()
                word = values[0]
                coefficients = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefficients

    def _create_embedding_matrix(self):
        self.embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.embedding_dimension))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def __init__(self, max_words, max_sequence_length, glove_dimension, embedding_dimension):
        self.tokenizer = Tokenizer(nb_words=max_words)
        self.max_sequence_length = max_sequence_length
        self.glove_dimension = glove_dimension
        self.embedding_dimension = embedding_dimension
        self.data = None
        self.embeddings_index = {}
        self.embedding_matrix = None
        self.embedding_layer = None

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        self._create_embedding_index()
        self._create_embedding_matrix()
        self.embedding_layer = Embedding(
            len(self.tokenizer.word_index) + 1,
            self.embedding_dimension,
            weights=[self.embedding_matrix],
            input_length=self.max_sequence_length,
            trainable=False
        )
