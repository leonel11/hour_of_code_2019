import sys
import fire
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.data_handler import DataHandler
from utils.text_classifier import TextClassifier

NUM_WORDS = 300000
MAX_COMMENT_LENGTH = 220


class Nlp(object):
    """
    Class for demonstration of
    """

    def prepare_data(self,
                     raw_data: str = r'./data',
                     stop_words: str = r'./data/stop_words.txt',
                     processed_data: str = r'./processed_data') -> None:
        """
        It cleans data and prepares datasets ready for training a model 
        and evaluating it.
        """
        print('Preparing data')
        dh = DataHandler(raw_data, stop_words)
        dh.prepare_data()
        print(f'Saving data into {processed_data}')
        dh.save(processed_data)

    def train(self,
              data_path: str,
              model_path: str = None,
              glove_path: str = r'./data/glove.840B.300d.txt',
              embedding_dim: int = 300,
              num_words: int = NUM_WORDS,
              max_comment_length: int = MAX_COMMENT_LENGTH,
              epochs: int = 10,
              batch_size: int = 512) -> None:
        """
        It trains a model
        """
        print(f'Loading data from {data_path}')
        dh = DataHandler(data_path)
        tc = TextClassifier()
        tc.fit(dh.X_train,
               dh.y_train,
               num_words=num_words,
               glove_path=glove_path,
               embedding_dim=embedding_dim,
               sequence_length=max_comment_length,
               validation_data=(dh.X_val, dh.y_val),
               epochs=epochs,
               batch_size=batch_size)
        tc.save(model_path)
        preds = tc.predict_proba(dh.X_test,
                                 num_words=num_words,
                                 sequence_length=max_comment_length,
                                 batch_size=batch_size)
        print('ROC_AUC score for test data:', roc_auc_score(dh.y_test, preds))

    def test(self,
             model_path: str,
             num_words: int = NUM_WORDS,
             max_comment_length: int = MAX_COMMENT_LENGTH) -> None:
        """
        It tests entered comments for abusiveness
        """
        tc = TextClassifier()
        print(f'Loading model from {model_path}')
        tc.load(model_path)
        print(
            'Enter a message and I will tell you whether it is abusive or not')
        print('To exit, please, press Ctrl+C')
        while True:
            comment = input('--> ')
            print(comment)
            print(
                'Prediction:',
                tc.predict_proba(np.array([comment]),
                                    num_words=num_words,
                                    sequence_length=max_comment_length,
                                    batch_size=1))


def main():
    fire.Fire(Nlp)


if __name__ == '__main__':
    main()
