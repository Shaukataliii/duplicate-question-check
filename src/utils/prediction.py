import pickle, os, keras
import numpy as np
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

@st.cache_resource
def load_and_cache_resources():
    """Uses pre-defined DL model and tokenizer paths, loads them and returns predictor module object. We can use that object for inference."""
    cwd = os.getcwd()
    model_filepath = r'{}\src\models\dl_model.h5'.format(cwd)
    tokenizer_filepath = r'{}\src\tokenizers\dl_tokenizer.pickle'.format(cwd)
    predictor = Prediction(model_filepath, tokenizer_filepath)
    return predictor


class Prediction:
  def __init__(self, model_filepath, tokenizer_filepath):
    self.model = self.load_keras_model(model_filepath)
    self.tokenizer = self.load_pkl_object(tokenizer_filepath)
    self.MAX_SEQUENCE_LEN = 30

  def load_keras_model(self, model_filepath):
    if self.check_file_existance(model_filepath):
      return load_model(model_filepath)

  def check_file_existance(self, filepath):
    if os.path.exists(filepath):
      return True
    else:
      raise FileNotFoundError(f"File not found at {filepath}")

  def load_pkl_object(self, filepath):
    if self.check_file_existance(filepath):
      with open(filepath, 'rb') as file:
        return pickle.load(file)

  def predict_on_questions(self, question1: str, question2: str):
    transformed_input = self.transform_input(question1, question2)
    return self.predict(transformed_input)

  def transform_input(self, question1: str, question2: str):
    sequences_1, sequences_2 = self.tokenize_questions(question1, question2)
    length_feats = self.create_length_features(sequences_1, sequences_2)
    return [sequences_1, sequences_2, length_feats]

  def tokenize_questions(self, question1: str, question2: str):
    sequences_1 = self.tokenizer.texts_to_sequences([question1])
    sequences_2 = self.tokenizer.texts_to_sequences([question2])
    sequences_1 = pad_sequences(sequences_1, maxlen=self.MAX_SEQUENCE_LEN)
    sequences_2 = pad_sequences(sequences_2, maxlen=self.MAX_SEQUENCE_LEN)
    return [sequences_1, sequences_2]

  def create_length_features(self, questions_1: list, questions_2: list):
    """The inputs needs to be a list of lists. We create three features i.e. length of unique words in q1 and same for q2 and len of common words."""
    length_features = [[len(set(question1)), len(set(question2)), len(set(question1).intersection(set(question2)))] for question1, question2 in zip(questions_1, questions_2)]
    return np.array(length_features, dtype = 'float32')

  def predict(self, transformed_input: list):
    prediction = self.model.predict(transformed_input)
    return self.get_prediction_label(prediction)

  def get_prediction_label(self, model_prediction):
    prediction_result = np.zeros_like(model_prediction)
    prediction_result = prediction_result[0]
    index_max_proba = model_prediction.argmax(axis=1)
    # prediction index 0 refers to not duplicate
    return "Duplicate" if index_max_proba == 1 else "Not Duplicate"
