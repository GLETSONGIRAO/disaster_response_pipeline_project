import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Function for load the data
    
    Args:
        database_filepath (string): path to SQLite db
    Returns:
        X (Series): Series of messages column in loaded dataframe
        Y (dataframe): Labels DataFrame
        category_names (list): list of Y dataframe columns
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
     """Function to tokenize and Lemmatize text string
    
    Args:
        text (string): String containing message for processing.
       
    Returns:
        List of processed tokens.
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word).lower().strip() for word in tokens]


def build_model():
     """Build model.

    Returns:
        cv (GridSearchCV object): Contains a sklearn estimator.
    """
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    
    parameters = {'clf__estimator__max_depth': [5, 15],
              'clf__estimator__min_samples_leaf': [2, 5],
              'clf__estimator__min_samples_split': [3, 7],
              'clf__estimator__n_estimators': [10, 25],
               'clf__estimator__max_features':[0.3,0.65]}
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    return cv

    
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """Function to evaluate the model

    args:
        model (sklearn.model_selection.GridSearchCV): Sklearn estimator "cv" on last function
        X_test (numpy.ndarray): Disaster messages.
        Y_test (numpy.ndarray): Disaster categories for each messages
        category_names (list): Disaster category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))



def save_model(model, model_filepath):
    '''
    function to save the model as a pkl file

    Args:
        model (sklearn model): Results of grid serach best parameters for the model
        model_filepath (string): The path which we want to save the file
    '''
    import joblib
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=13)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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