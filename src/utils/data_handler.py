try:
    from HTMLParser import HTMLParser
except ImportError:
    from html.parser import HTMLParser
import string
import itertools
import re
import os
import multiprocessing
import joblib
import numpy as np
import pandas as pd


class DataHandler():

    def __init__(self, data_path: str = None, stop_words_path: str = None):
        if data_path is not None:
            self._X_train = pd.read_csv(f'{data_path}/X_train.csv')
            self._X_val = pd.read_csv(f'{data_path}/X_val.csv')
            self._X_test = pd.read_csv(f'{data_path}/X_test.csv')
            self._y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None)
            self._y_val = pd.read_csv(f'{data_path}/y_val.csv', header=None)
            self._y_test = pd.read_csv(f'{data_path}/y_test.csv', header=None)
        if stop_words_path is not None:
            self._stop_words = set(open(stop_words_path).read().split())
        else:
            self._stop_words = set()
        self._url_regexp = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
        self._emoji_pattern = re.compile(
            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
            u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
            "+",
            flags=re.UNICODE)
        self._all_punct_pattern = re.compile(
            '[%s]' % re.escape(string.punctuation.replace('\'', '')))
        self._punctuation = set(
            string.punctuation)  # string of ASCII punctuation
        self._exclamation_token = ' exclmrk '
        self._question_token = ' qstmrk '
        self._dot_token = ' eosmkr '
        self._hash_token = ' hshmrk '

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train['comment_text']

    @property
    def X_val(self) -> pd.DataFrame:
        return self._X_val['comment_text']

    @property
    def X_test(self) -> pd.DataFrame:
        return self._X_test['comment_text']

    @property
    def y_train(self) -> pd.DataFrame:
        return self._y_train

    @property
    def y_val(self) -> pd.DataFrame:
        return self._y_val

    @property
    def y_test(self) -> pd.DataFrame:
        return self._y_test

    def prepare_data(self, cpus: int = None):

        def process(series):
            return u'\"{}\"'.format(self.clean_comment(series['comment_text']))

        def call_process(df):
            return df.apply(process, axis=1)

        def clean(data: pd.DataFrame, cpus: int) -> None:
            pool_results = joblib.Parallel(n_jobs=cpus, backend='threading')(
                joblib.delayed(call_process)(df)
                for df in np.array_split(data, cpus))
            data['comment_text'] = pd.concat(pool_results)

        if cpus is None:
            cpus = multiprocessing.cpu_count()
        print('Preparing train data')
        clean(self._X_train, cpus)
        print('Preparing  test data')
        clean(self._X_val, cpus)
        print('Preparing  validation data')
        clean(self._X_test, cpus)

    def save(self, data_path: str) -> None:
        if not os.path.exists(data_path):
            # It may raise winerror 123
            os.makedirs(data_path)
        self._X_train.to_csv(f'{data_path}/X_train.csv', index=False)
        self._X_val.to_csv(f'{data_path}/X_val.csv', index=False)
        self._X_test.to_csv(f'{data_path}/X_test.csv', index=False)
        self._y_train.to_csv(f'{data_path}/y_train.csv',
                             index=False,
                             header=False)
        self._y_val.to_csv(f'{data_path}/y_val.csv', index=False, header=False)
        self._y_test.to_csv(f'{data_path}/y_test.csv',
                            index=False,
                            header=False)

    def clean_comment(self, comment: str) -> str:
        """
        Cleans comment
        Returns clean comment string in utf-8.
        Original raw `comment` is not transformed
        """
        clean_comment = self._remove_code_sequencies(comment,
                                                     html=True,
                                                     wiki_templates=False)
        clean_comment = self._remove_urls(comment)
        clean_comment = self._remove_emojis(clean_comment)
        clean_comment = self._standardize_words(clean_comment)
        clean_comment = self._remove_punctuation(clean_comment)
        clean_comment = self._remove_digits(clean_comment)
        # clean_comment = self._remove_stop_words(clean_comment)
        clean_comment = self._replace_marks_with_tokens(clean_comment)
        return clean_comment

    @staticmethod
    def _remove_code_sequencies(comment, html=True,
                                wiki_templates=False) -> str:
        """
        Removes html code and wiki templates
        Returns clean comment
        """
        clean_comment = comment
        if html:
            clean_comment = HTMLParser().unescape(comment)
        if wiki_templates:
            # TODO Clean wiki templates
            pass
        return clean_comment

    @staticmethod
    def _remove_punctuation(comment: str) -> str:
        """
        Removes all punctuations except !?,.'.
        Returns clean comment
        """
        comment = comment.replace("$", "s")
        comment = comment.replace("@", "a")
        # removes other punctuation
        new_comment = re.sub(r"[^\w\s!?.'$#@]", ' ', comment)
        # removes duplicate punctuation
        new_comment = re.sub(r"([\s,.'])\1+", r'\1', new_comment)
        # remove spaces before punctuation
        new_comment = re.sub(r"[\s]+(?=[!.,'])", '', new_comment)
        # add space after punctuation
        new_comment = re.sub(r"([?!.,])(?=[\w\d])", r'\1 ', new_comment)
        clean_comment = new_comment.strip()
        return clean_comment

    def _remove_emojis(self, comment: str) -> str:
        """
        Removes emojis from the comment
        Returns clean comment
        """
        return self._emoji_pattern.sub(' ', comment)

    @staticmethod
    def _split_attached_words(comment: str) -> str:
        """
        HelloToEveryone -> Hello To Everyone
        Returns clean comment
        """
        return ' '.join(re.findall('[A-Z][^A-Z]*', comment))

    def _remove_urls(self, comment: str) -> str:
        """
        Removes URLs
        Returns clean comment
        """
        return re.sub(self._url_regexp, '', comment)

    @staticmethod
    def _standardize_words(comment: str) -> str:
        """
        happpy -> happy :-), looooove -> loove :-(
        Returns clean comment
        """
        return ''.join(''.join(s)[:2] for _, s in itertools.groupby(comment))

    @staticmethod
    def _remove_digits(comment: str) -> str:
        """
        Removes digits
        Returns clean comment
        """
        new_comment = []
        for word in comment.split():
            # Filter out punctuation and stop words
            if (not word.lstrip('-').replace('.', '', 1).isdigit()):
                new_comment.append(word)
        clean_comment = ' '.join(new_comment)
        return clean_comment

    def _remove_stop_words(self, comment: str) -> str:
        """
        Removes all stop-words and separate numbers from comment
        Returns clean comment
        """
        new_comment = []
        for word in comment.split():
            # It filters out punctuation and stop words
            if (word not in self._punctuation and self._all_punct_pattern.sub(
                    '', word.lower()) not in self._stop_words):
                new_comment.append(word)
        return ' '.join(new_comment)

    def _replace_marks_with_tokens(self,
                                   comment: str,
                                   exclamation=True,
                                   question=True,
                                   hash_mrk=True,
                                   eos_mrk=True) -> str:
        """
        '!' -> 'exclmrk', '?' -> 'qstmrk'
        Returns clean comment
        """
        new_comment = comment
        if exclamation:
            new_comment = new_comment.replace('!', self._exclamation_token)
        if question:
            new_comment = new_comment.replace('?', self._question_token)
        if eos_mrk:
            new_comment = new_comment.replace('.', self._dot_token)
        if hash_mrk:
            new_comment = new_comment.replace('#', self._hash_token)
        return new_comment
