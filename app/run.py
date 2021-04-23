# import libraries
import joblib
import pandas as pd
from sqlalchemy import create_engine
import plotly

import re
import nltk
import string
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import json
import plotly

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar




app = Flask(__name__)

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word).lower().strip() for word in tokens]

# load data
engine = create_engine('sqlite:///../data/disaster-response.db')

df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    original_counts = [df.original.notnull().sum(), df.original.isnull().sum()]
    original_names = ['Translated', 'Not Translated']

    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=original_names,
                    y=original_counts
                )
            ],

            'layout': {
                'title': 'Count of translated messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Translated"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()