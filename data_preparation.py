import pandas as pd
import natasha
from nltk.corpus import stopwords
import re
import numpy as np
from wikipedia2vec import Wikipedia2Vec
import fasttext
from sklearn.model_selection import train_test_split
from tqdm import tqdm


SEED = 42


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


def map_from_exact_to_enlarged(
        ktru_code: str,
        mapping_table: pd.DataFrame
    ):
    '''
    mapping_table: | code | parent_code | is_template |
    '''
    try:
        ktru_record = mapping_table[mapping_table['code'] == ktru_code]
        is_template = ktru_record['is_template'].array[0]

        if is_template == True:
            return ktru_code
        else:
            parent_code = ktru_record['parent_code'].array[0]
            return parent_code

    except IndexError:
        return ktru_code


def remove_rare_codes(
        df: pd.DataFrame,
        column: str='ktru_12',
        min_freq: int=2,
        group_by: str='product_name'
    ) -> pd.DataFrame:

    df = df.copy()
    frequent = df.groupby(column).count()[group_by] >= min_freq
    frequent = frequent.index[frequent]
    indexes = df[column].isin(frequent)
    df = df[indexes]
    return df


def separate_train_test(
        df: pd.DataFrame,
        test_size=0.2,
        stratify_col='ktru_code'    
    ):

    df = df.copy()

    y_train, y_test = train_test_split(
                        df.index, 
                        test_size=test_size, 
                        stratify=df[stratify_col], 
                        random_state=SEED
                    )

    df['train'] = df.index.isin(y_train)

    return df


def get_embeddings(
        df: pd.DataFrame,
        name: str='product_name',
        descr: str='product_descr'
    ):
    embedder = Embedder()
    embeddings = []

    for title, description, index in zip(tqdm(df[name]), df[descr], df.index):
        try:
            title_emb = embedder.get_sent_emb(title)
            description_emb = embedder.get_sent_emb(description)
            embedding = np.concatenate([title_emb, description_emb])
            embeddings.append(embedding)
        except:
            print(title)
            print(index)
            break
    
    return embeddings