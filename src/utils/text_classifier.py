"""
Model dispatcher
"""
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from typing import List, Dict
import keras.backend.tensorflow_backend as K
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras import initializers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import (Dense, CuDNNLSTM, Bidirectional, Dropout, Input,
                          SpatialDropout1D, GlobalAveragePooling1D,
                          GlobalMaxPooling1D, MaxPooling1D, concatenate)
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils import multi_gpu_model, plot_model
import numpy as np
import matplotlib.pyplot as plt
import constants


class TextClassifier(object):

    def save(self, model_path: str = None):
        if self._history is not None:
            model_path = self._get_model_path(self._history, model_path)
            os.makedirs(model_path, exist_ok=True)
            print('Saving model statistics')
            self._save_training_stats(self._history, model_path)
        else:
            return
        if self._model is not None:
            print('Saving model')
            self._model.save_weights(
                os.path.join(model_path, 'model_weights.h5'))
        pass

    def load(self, model_path: str):
        self._model = load_model(model_path)

    def fit(self,
            X_train,
            y_train,
            glove_path,
            embedding_dim,
            num_words,
            sequence_length,
            validation_data=None,
            epochs=1,
            batch_size=None) -> None:
        self._tokenizer = Tokenizer(num_words, oov_token=True)
        self._tokenizer.fit_on_texts(X_train)
        X_train = self._text2vec(X_train, num_words, sequence_length)
        parallel_model, self._model = self._lstm(num_words, sequence_length,
                                                 glove_path, embedding_dim)
        print('Training model')
        print(self._model.summary())
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0, patience=5),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              min_lr=0.000001)
        ]
        self._history = parallel_model.fit(X_train,
                                           y_train,
                                           validation_data=validation_data,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           callbacks=callbacks)

    def predict_proba(self, x, batch_size=None) -> np.ndarray:
        return self._model.predict(x, batch_size=batch_size)

    def evaluate(self, x, y, batch_size=None):
        """
        Returns the loss value & metrics values for the model in test mode.
        """
        return self._model.evaluate(x, y, batch_size)

    @staticmethod
    def _get_model_path(history, model_path: str = None) -> str:
        model_path = os.path.join(
            constants.MODELS_PATH, 'lstm_{:.4f}_{:.4f}'.format(
                history.history['val_loss'][-1] if 'val_loss' in history.history
                else history.history['loss'][-1],
                history.history['val_acc'][-1] if 'val_acc' in history.history
                else history.history['acc'][-1]))
        return model_path

    @staticmethod
    def _get_gpus(gpus: str) -> List[int]:
        """
        Returns a list of integers (numbers of gpus)
        """
        return list(map(int, gpus.split(',')))

    @staticmethod
    def _load_txt_model(model_path) -> Dict:
        """
        Returns pretrained serialized model saved in text format
        where numbers are separated with spaces
        """
        pickled_model = os.path.join(
            constants.PICKLES_PATH,
            '{}.pickle'.format(os.path.basename(model_path)))
        try:
            # load ready text model
            with open(pickled_model, 'rb') as model:
                return pickle.load(model)
        except:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # form text model
            with open(model_path, 'r') as file:
                model = {}
                for line in file:
                    splitLine = line.split()
                    # pull word
                    word = splitLine[0]
                    # pull features
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                with open(pickled_model, 'wb') as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                return model

    @staticmethod
    def _plot_loss_acc(history, model_path):
        """
        Saves into files accuracy and loss plots
        """
        plt.gcf().clear()
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(model_path, 'accuracy.png'))
        plt.gcf().clear()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(model_path, 'loss.png'))
        plt.gcf().clear()

    def _save_training_stats(self, history, model_path):
        plot_model(self._model,
                   os.path.join(model_path, 'model.png'),
                   show_shapes=True)
        self._plot_loss_acc(history, model_path)

    def _text2vec(self, X_train, num_words, max_comment_length):
        X_train_seq = []
        for seq in self._tokenizer.texts_to_sequences_generator(X_train):
            X_train_seq.append(seq)
        # Truncate and pad input sequences
        X_train_seq = sequence.pad_sequences(
            self._tokenizer.texts_to_sequences(X_train),
            maxlen=max_comment_length)
        return X_train_seq

    def _lstm(self, top_words: int, sequence_length: int, glove_path: str,
              embedding_dim: int):
        """
        Returns compiled keras lstm model ready for training
        Params:
        - top_words - load the dataset but only keep the top n words, zero the rest
        """
        units = 256
        inputs = Input(shape=(sequence_length,), dtype='int32')
        x = self._get_pretrained_embedding(top_words, sequence_length,
                                           glove_path, embedding_dim)(inputs)
        x = SpatialDropout1D(0.2)(x)
        # For mor detais about kernel_constraint - see chapter 5.1
        # in http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
        x = Bidirectional(CuDNNLSTM(
            units,
            kernel_initializer=initializers.he_uniform(),
            recurrent_regularizer=regularizers.l2(),
            return_sequences=True),
                          merge_mode='concat')(x)
        x = Bidirectional(CuDNNLSTM(
            units,
            kernel_initializer=initializers.he_uniform(),
            recurrent_regularizer=regularizers.l2(),
            return_sequences=True),
                          merge_mode='concat')(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        output = Dense(6, activation='sigmoid')(x)
        gpus = self._get_gpus(os.environ['CUDA_VISIBLE_DEVICES'])
        if len(gpus) == 1:
            with K.tf.device('/gpu:{}'.format(gpus[0])):
                model = Model(inputs, output)
                parallel_model = model
        else:
            with K.tf.device('/cpu:0'):
                # creates a model that includes
                model = Model(inputs, output)
            parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(loss='binary_crossentropy',
                               optimizer=Adam(lr=1e-3),
                               metrics=['accuracy'])
        return parallel_model, model

    def _get_pretrained_embedding(self, top_words: int, sequence_length: int,
                                  glove_path: str, embedding_dim: int):
        """
        Returns Embedding layer with pretrained word2vec weights
        """
        word_vectors = {}
        if glove_path is not None:
            word_vectors = self._load_txt_model(glove_path)
        else:
            return Embedding(input_dim=top_words,
                             output_dim=embedding_dim,
                             input_length=sequence_length,
                             trainable=False,
                             mask_zero=False)

        embedding_matrix = np.zeros((top_words, embedding_dim))
        for word, i in self._tokenizer.word_index.items():
            if i >= top_words:
                continue
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25),
                                                       embedding_dim)

        return Embedding(input_dim=top_words,
                         output_dim=embedding_dim,
                         input_length=sequence_length,
                         weights=[embedding_matrix],
                         trainable=False,
                         mask_zero=False)
