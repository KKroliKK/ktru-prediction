import pandas as pd
import natasha
from nltk.corpus import stopwords
import re
import numpy as np
from wikipedia2vec import Wikipedia2Vec
import fasttext
from sklearn.model_selection import train_test_split


def add_partial_ktru_codes(
        df: pd.DataFrame,
        ktru_column: str='ktru_code',
        inplace: bool=False
    ):
    if not inplace:
        df = df.copy()

    prefixes = [2, 5, 8, 12]

    for prefix in prefixes:
        df[f'ktru_{prefix}'] = df[ktru_column].apply(lambda x: x[:prefix], convert_dtype=False)

    if not inplace:
        return df


class NatashaTokenizer():
    stop_words = stopwords.words('russian')
    stop_words += ['.', ',', '"', '!', "''", '%', '«', '»', '“', '”', ':', '№', '=',
                '?', '(', ')', '-', '``', '@', '#', "'", '—', '/', '+', '&', '*',
                ':', ';', '_', '\\', '...', '\n', '$', '[', ']', '>', '<', '..']

    stop_tags = ['PUNCT', 'NUM']

    def __init__(self):
        self.segmenter = natasha.Segmenter()
        nat_emb = natasha.NewsEmbedding()
        self.morph_tagger = natasha.NewsMorphTagger(nat_emb)
        self.morph_vocab = natasha.MorphVocab()

    def tokenize(self, text, remove_numbers=True):
        try:
            doc = natasha.Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)

            def check(token):
                if token.pos in self.stop_tags:
                    return False
                if token.lemma in self.stop_words:
                    return False
                if len(token.lemma) < 3:
                    return False
                if re.search('\d', token.lemma) and remove_numbers:
                    return False
                return True

            tokens = [token.lemma for token in doc.tokens if check(token)]
        
        except:
            tokens = []

        return tokens


class Embedder():
    def __init__(self, use_wiki2vec=False) -> None:
        self.use_wiki2vec = use_wiki2vec
        self.tokenizer = NatashaTokenizer()

        self.ft = fasttext.load_model('cc.ru.300.bin')
        if use_wiki2vec:
            self.wiki2vec = Wikipedia2Vec.load('ruwiki_20180420_300d.pkl')

    def vectorize(self, token):
        try:
            fast_text_vector = self.ft.get_word_vector(token)
        except KeyError:
            fast_text_vector = np.zeros((self.ft.get_dimension()))

        if self.use_wiki2vec:
            try:
                word2vec_vector = self.wiki2vec.get_word_vector(token)
            except KeyError:
                word2vec_vector = np.zeros((len(self.wiki2vec.get_word_vector('word'))))
            return np.concatenate([word2vec_vector, fast_text_vector])

        else:
            return fast_text_vector


    def get_sent_emb(self, sentence, remove_numbers=True):
        tokens = self.tokenizer.tokenize(sentence, remove_numbers)
        sent_emb = []
        for token in tokens:
            sent_emb.append(self.vectorize(token))
            
        return np.mean(sent_emb, axis=0)


def separate_train_test(
        df: pd.DataFrame,
        full_ktru_col: str,
        
    ):
    pass