import numpy as np
import pandas as pd
import spacy
import re
import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as ms
from sklearn.model_selection import cross_validate

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Initialize spacy ‘en’ model, keeping only component needed for lemmatization and creating an engine:
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

#reading in data
x_train_df = pd.read_csv('data_reviews/x_train.csv')
y_train_df = pd.read_csv('data_reviews/y_train.csv')


#tokenization + vectorization
tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_sentence)
x = tfidf_vectorizer.fit_transform(x_train_df['text'])
pd.DataFrame(x.toarray(), columns=tfidf_vectorizer.get_feature_names())


x_train = x.toarray()
y_train = y_train_df.to_numpy().ravel()

network = MLPClassifier()

try_params = {
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': 10.0 ** -np.arange(-2,2),
    'max_iter': np.linspace(100,1000,5).astype(int)
}

gscv = GridSearchCV(network, try_params, cv=3, return_train_score=True)
gscv.fit(x_train, y_train)
print('Best parameters found:\n', gscv.best_params_)