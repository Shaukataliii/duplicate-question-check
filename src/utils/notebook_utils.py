import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import re, string
from gensim.models import Word2Vec
import pickle

QUESTION_COLS = ['question1', 'question2']
UNUSED_COLS = ['id', 'qid1', 'qid2']
TARGET_COL = ['is_duplicate']
STOPWORDS = stopwords.words('english')
PUNCTUATION_REPLACEMENTS = {
    '!': ' exclamation ',
    '?': ' question ',
    "'": ' apostrophe '
}
PUNCTUATIONS = string.punctuation
SPECIAL_CHARS = {
    '%': ' percent',
    '$': ' dollar ',
    '₹': ' rupee ',
    '€': ' euro ',
    '@': ' at ',
    '[math]': ''
}
ABBREVIATIONS = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }


class FeatureEngineer():
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.do_basic_preprocessing()

        self.ques1_colname = QUESTION_COLS[0]
        self.ques2_colname = QUESTION_COLS[1]
        self.ques1_len_colname = 'ques1_len'
        self.ques2_len_colname = 'ques2_len'
        self.ques1_wordcount_colname = 'ques1_word_count'
        self.ques2_wordcount_colname = 'ques2_word_count'
        self.ques1_stopwordcount_colname = 'ques1_stopwords_count'
        self.ques2_stopwordcount_colname = 'ques2_stopwords_count'
        self.ques1_stopwords_ratio_colname = 'ques1_stopwords_ratio'
        self.ques2_stopwords_ratio_colname = 'ques2_stopwords_ratio'
        
        self.common_wordcount_colname = 'common_words_count'
        self.total_words_colname = 'ques1+ques2_word_count'
        self.word_share_ratio_colname = 'word_share_ratio'

        self.advance_feature_colnames = [
            'cmn_tokens_cnt_min',
            'cmn_tokens_cnt_max',
            'cmn_words_cnt_min',
            'cmn_words_cnt_max',
            'cmn_stopwords_cnt_min',
            'cmn_stopwords_cnt_max',
            'last_word_equal',
            'first_word_equal'
        ]
        
    def do_basic_preprocessing(self):
        self.dataframe = self.dataframe.drop(UNUSED_COLS, axis=1)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.drop_duplicates()

    def add_features(self):
        self.add_intuitive_features()
        self.add_advance_features()

        return self.dataframe
    
    def add_intuitive_features(self):
        self.dataframe[self.ques1_len_colname] = self.dataframe[self.ques1_colname].apply(self._compute_length)
        self.dataframe[self.ques2_len_colname] = self.dataframe[self.ques2_colname].apply(self._compute_length)

        self.dataframe[self.ques1_wordcount_colname] = self.dataframe[self.ques1_colname].apply(self._count_words)
        self.dataframe[self.ques2_wordcount_colname] = self.dataframe[self.ques2_colname].apply(self._count_words)

        self.dataframe[self.common_wordcount_colname] = self.dataframe[QUESTION_COLS].apply(self._count_common_words, axis=1)

        self.dataframe[self.total_words_colname] = self.dataframe[QUESTION_COLS].apply(self._count_total_words, axis=1)

        self.dataframe[self.word_share_ratio_colname] = self.dataframe[[self.common_wordcount_colname, self.total_words_colname]].apply(self._compute_word_share_ratio, axis=1)

        self.dataframe[self.ques1_stopwordcount_colname] = self.dataframe[self.ques1_colname].apply(self._count_stop_words)
        self.dataframe[self.ques2_stopwordcount_colname] = self.dataframe[self.ques2_colname].apply(self._count_stop_words)

        self.dataframe[self.ques1_stopwords_ratio_colname] = self.dataframe[[self.ques1_stopwordcount_colname, self.ques1_wordcount_colname]].apply(self._compute_stopwords_ratio, axis=1)
        self.dataframe[self.ques2_stopwords_ratio_colname] = self.dataframe[[self.ques2_stopwordcount_colname, self.ques2_wordcount_colname]].apply(self._compute_stopwords_ratio, axis=1)

    def _compute_length(self, question: str):
        return len(question)

    def _count_words(self, question: str):
        return len(question.split())

    def _count_common_words(self, questions_odj):
        question1 = set(questions_odj[self.ques1_colname].split())
        question2 = set(questions_odj[self.ques2_colname].split())
        return len(question1.intersection(question2))

    def _count_total_words(self, questions_obj):
        question1 = questions_obj[self.ques1_colname].split()
        question2 = questions_obj[self.ques2_colname].split()
        return len(question1 + question2)
        
    def _compute_word_share_ratio(self, common_wordcount_and_total_wordcount_obj):
        common_words = common_wordcount_and_total_wordcount_obj[self.common_wordcount_colname]
        total_words = common_wordcount_and_total_wordcount_obj[self.total_words_colname]
        return np.round(common_words/total_words, 2) if total_words > 0 else 0

    def _count_stop_words(self, question: str):
        question = set(question.split())
        return len(question.intersection(set(STOPWORDS)))
    
    def _compute_stopwords_ratio(self, common_stopwordcount_and_total_wordcount_obj):
        if self.ques1_stopwordcount_colname in common_stopwordcount_and_total_wordcount_obj:
            stopwords_count = common_stopwordcount_and_total_wordcount_obj[self.ques1_stopwordcount_colname]
            total_words_count = common_stopwordcount_and_total_wordcount_obj[self.ques1_wordcount_colname]

        else:
            stopwords_count = common_stopwordcount_and_total_wordcount_obj[self.ques2_stopwordcount_colname]
            total_words_count = common_stopwordcount_and_total_wordcount_obj[self.ques2_wordcount_colname]

        return np.round(stopwords_count / total_words_count, 2)
    
    def add_advance_features(self):
        advance_features = self.dataframe.apply(self.compute_advance_features, axis=1)
        self.dataframe[self.advance_feature_colnames] = advance_features
        
    def compute_advance_features(self, row):
        features = {}
        SENSITIVITY = 0.0001
        ques1 = row[QUESTION_COLS[0]]
        ques2 = row[QUESTION_COLS[1]]

        ques1_tokens = ques1.split()
        ques2_tokens = ques2.split()
        if len(ques1_tokens) == 0 or len(ques2_tokens) == 0:
            return features
        
        ques1_words = [word for word in ques1_tokens if word not in STOPWORDS]
        ques1_stopwords = [word for word in ques1_tokens if word in STOPWORDS]
        ques2_words = [word for word in ques2_tokens if word not in STOPWORDS]
        ques2_stopwords = [word for word in ques2_tokens if word in STOPWORDS]        

        common_tokens_count = len(set(ques1_tokens).intersection(set(ques2_tokens)))
        common_words_count = len(set(ques1_words).intersection(set(ques2_words)))
        common_stopwords_count = len(set(ques1_stopwords).intersection(set(ques2_stopwords)))

        ctc_min = common_tokens_count / (min(len(ques1_tokens), len(ques2_tokens)) + SENSITIVITY)
        ctc_max = common_tokens_count / (max(len(ques1_tokens), len(ques2_tokens)) + SENSITIVITY)
        cwc_max = common_words_count / (max(len(ques1_words), len(ques2_words)) + SENSITIVITY)
        cwc_min = common_words_count / (min(len(ques1_words), len(ques2_words)) + SENSITIVITY)
        csc_min = common_stopwords_count / (min(len(ques1_stopwords), len(ques2_stopwords)) + SENSITIVITY)
        csc_max = common_stopwords_count / (max(len(ques1_stopwords), len(ques2_stopwords)) + SENSITIVITY)
        first_word_eq = self.is_firstword_equal(ques1_words, ques2_words)
        last_word_eq = self.is_lastword_equal(ques1_words, ques2_words)

        return pd.Series([ctc_min, ctc_max, cwc_min, cwc_max, csc_min, csc_max, first_word_eq, last_word_eq])

    def is_firstword_equal(self, list1: str, list2: str):
        if (not list1) or (not list2):
            return int(0)
        return int(list1[0] == list2[0])
    
    def is_lastword_equal(self, list1: str, list2: str):
        if (not list1) or (not list2):
            return int(0)
        return int(list1[-1] == list2[-1])

class PreProcessor:
    def apply_preprocessing(self, df: pd.DataFrame):
        df[QUESTION_COLS[0]] = df[QUESTION_COLS[0]].apply(self.apply_transformations)
        df[QUESTION_COLS[1]] = df[QUESTION_COLS[1]].apply(self.apply_transformations)
        return df
    
    def apply_transformations(self, text: str):
        text = self.lowercase(text)
        text = self.expand_abbreviations_in_question(text)
        text = self.replace_special_chars(text)
        text = self.replace_numbers(text)
        text = self.replace_punctuations(text)
        text = self.remove_stopwords(text)
        return text

    def lowercase(self, text: str):
        return text.lower()
    
    def expand_abbreviations_in_question(self, text: str):
        text = text.split()
        expanded_text = [self.expand_word(word) for word in text]
        return ' '.join(expanded_text)
    
    def expand_word(self, word: str):
        return ABBREVIATIONS.get(word, word)
        
    def replace_special_chars(self, text: str):
        for char, replacement in SPECIAL_CHARS.items():
            text = text.replace(char, replacement)
        return text

    def replace_numbers(self, text: str):
        text = re.sub(r'(\d+)\s*000000000', r'\1b', text)
        text = re.sub(r'(\d+)\s*000000', r'\1m', text)
        text = re.sub(r'(\d+)\s*000', r'\1k', text)
        return text
    
    def replace_punctuations(self, text: str):
        pattern = re.compile('|'.join(re.escape(key) for key in PUNCTUATION_REPLACEMENTS.keys()))
        text = pattern.sub(lambda m: PUNCTUATION_REPLACEMENTS[m.group()], text)
        text = text.translate(str.maketrans('', '', PUNCTUATIONS))
        return text
    
    def remove_stopwords(self, text: str):
        words = text.split()
        filtered_words = [word for word in words if word not in STOPWORDS]
        return ' '.join(filtered_words)

class Embeddings:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.vector_size = 100

    def encode_questions(self):
        all_questions = list(self.dataframe[QUESTION_COLS[0]]) + list(self.dataframe[QUESTION_COLS[1]])
        self.model = self.build_model(all_questions)
        
        first_questions_encoded = self.dataframe[QUESTION_COLS[0]].apply(self.get_average_vector).tolist()
        second_questions_encoded = self.dataframe[QUESTION_COLS[1]].apply(self.get_average_vector).tolist()
        
        first_questions_df = pd.DataFrame(first_questions_encoded, index=self.dataframe.index)
        second_questions_df = pd.DataFrame(second_questions_encoded, index=self.dataframe.index)
        
        first_questions_df.columns = [f'{QUESTION_COLS[0]}_vec_{i}' for i in range(self.vector_size)]
        second_questions_df.columns = [f'{QUESTION_COLS[1]}_vec_{i}' for i in range(self.vector_size)]
        
        self.dataframe = pd.concat([self.dataframe, first_questions_df, second_questions_df], axis=1)
        self.dataframe = self.dataframe.drop(QUESTION_COLS, axis=1)

        return self.dataframe
    
    def build_model(self, questions: list):
        tokenized_questions = [question.split() for question in questions]
        model = Word2Vec(tokenized_questions, vector_size=self.vector_size, window=5, min_count=1)
        print("Total Examples:", model.corpus_count)

        model.train(questions, total_examples=model.corpus_count, epochs=25)
        return model    
    
    def get_average_vector(self, question: str):
        encoded_words = []
        question_words = question.split()
        # creating a matrix of vectors representing each word
        encoded_words = [self.model.wv[word] for word in question_words if word in self.model.wv]
        if not encoded_words:
            return np.zeros(self.model.vector_size)

        return np.mean(encoded_words, axis=0).tolist()
    
class ModelBuilder:
    def build_evaluate_models(self, dataframe: pd.DataFrame):
        # splitting data
        dataframe.columns = dataframe.columns.astype(str)
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(TARGET_COL, axis=1), dataframe[TARGET_COL], test_size=0.2, random_state=20)

        models = [LogisticRegression(), RandomForestClassifier()]
        results = {}

        for model in models:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)

            results[type(model).__name__] = accuracy
            print(f"Algorithm evaluated: {type(model).__name__}")

        return results
    
    def train_save_best(self, dataframe: pd.DataFrame):
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(TARGET_COL, axis=1), dataframe[TARGET_COL], test_size=0.2, random_state=20)
        model = RandomForestClassifier(max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=150)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        train_accuracy = accuracy_score(model.predict(x_train), y_train)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Train accuracy: {train_accuracy} \n Test Accuracy: {accuracy}")
        
        self.save_model(model)
        return (model, accuracy, report)

    def save_model(self, model):
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)