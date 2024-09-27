# import libraries
import pandas as pd
import sys
import re
import os
import numpy as np
import pickle
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import xgboost as xgb
from xgboost import XGBClassifier
from joblib import parallel_backend

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.model_selection import learning_curve

def load_data(database_filepath):
    """
    Loads the data from the SQLite or PostgreSQL database and returns the feature matrix (X), 
    target matrix (y), and the category names.
    
    Args:
        database_filepath (str): The file path of the SQLite database or the URL of the PostgreSQL database.
        
    Returns:
        X (np.array): The feature matrix.
        y (np.array): The target matrix.
        category_names (list): The list of category names.
    """
    
    # Check if we're using PostgreSQL (Heroku) or SQLite (local)
    if 'DATABASE_URL' in os.environ:
        # Running on Heroku (PostgreSQL)
        engine = create_engine(os.getenv('DATABASE_URL'))
    else:
        # Running locally (SQLite)
        engine = create_engine(f'sqlite:///{database_filepath}')

    # Load data from the table
    table_name = 'DisasterResponse'
    df = pd.read_sql_table(table_name, engine)
    
    # Separate the feature and target variables
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
    
    return X, y, category_names

def tokenize(text):
    """
    Tokenizes the input text by removing URLs, stripping non-alphabetic characters, 
    tokenizing the text into words, lemmatizing the words, and removing stop words.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        list: The list of cleaned tokens.
    """
    
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    rest_regex = r"[^a-zA-Z0-9]"
    
    # Stripping messages of all URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Stripping the messages of all symbols
    stripped = re.sub(rest_regex, " ", text)
    
    # Tokenize the sentences into words
    tokens = word_tokenize(stripped)
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        # Remove stop words
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    Builds a machine learning pipeline and uses GridSearchCV to optimize hyperparameters.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", XGBClassifier(n_jobs=-1, objective='binary:logistic'))
    ])

    # Parameters for Grid Search
    parameters = {
        'vect__max_df': [0.8],
        'vect__max_features': [5000],
        'tfidf__use_idf': [True],
        'clf__n_estimators': [200],
        'clf__max_depth': [7],
        'clf__learning_rate': [0.2]
    }

    # GridSearchCV with XGBoost for tuning
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=1, verbose=3, scoring='f1_macro', error_score='raise')

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates the trained model on the test set and prints the classification report for each category.
    """
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Generate classification reports for each category
    for i, category in enumerate(category_names):
        print(f"Classification report for {category}:")
        print(classification_report(y_test[:, i], y_pred[:, i]))
        print("\n")

def save_model(model, model_filepath):
    """
    Saves the trained model to a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    Main function that loads the data, builds and trains the model, evaluates the model, and saves the trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()