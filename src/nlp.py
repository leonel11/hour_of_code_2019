import sys
import fire
from utils.data_handler import DataHandler
from utils.text_classifier import TextClassifier


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
              glove_path: str = r'./data/glove.840B.300d',
              embedding_dim: int = 300,
              num_words: int = 180000,
              max_comment_length: int = 150,
              epochs: int = 10,
              batch_size: int = 256) -> None:
        """
        It trains a model
        """
        tc = TextClassifier()
        dh = DataHandler(data_path)
        tc.fit(dh.X_train,
               dh.y_train,
               num_words=num_words,
               glove_path=glove_path,
               embedding_dim=embedding_dim,
               sequence_length=max_comment_length,
               validation_data=(dh.X_val, dh.y_val),
               epochs=epochs,
               batch_size=batch_size)
        print(tc.evaluate(dh.X_test, dh.y_test, batch_size))
        tc.save(model_path)

    def test(self, model_path: str) -> None:
        tc = TextClassifier()
        tc.load(model_path)
        print('Enter a message and I will tell you whether it is offensive or not')
        print('To exit, please, press Ctrl+C')
        while True:
            for comment in sys.stdin.readline():
                print(tc.predict_proba(comment, 1))


def main():
    fire.Fire(Nlp)


if __name__ == '__main__':
    main()
