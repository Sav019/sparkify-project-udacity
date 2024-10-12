# Download nltk packages
nltk.download('punkt')
nltk.download('punkt_tab')

import json
import re
import plotly
import pandas as pd

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Heatmap
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

from tokenize_mod import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    """
    Renders the index webpage, which displays visualizations of the data.

    This function extracts the necessary data from the DataFrame and creates Plotly graphs to visualize the distribution of message genres, categories and a corrlation matrix.

    The function then encodes the Plotly graphs in JSON format and passes them to the 'master.html' template, along with the IDs of the graphs.

    Returns:
        Rendered HTML template for the index page, with the Plotly graphs.
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Number of data points per column
    numeric_cols = df.select_dtypes(include='number').columns
    column_sums = df[numeric_cols].sum(axis=0).drop("id")
    column_names = list(column_sums.index)
    
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
                    x=column_names,
                    y=column_sums
                )
            ],
            'layout': {
                'title': 'Number of Data Points per Column',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Column"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=numeric_cols,
                    y=numeric_cols,
                    z=df[numeric_cols].corr(),
                    colorscale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
            ],
            'layout': {
                'title': 'Correlation Matrix',
                'xaxis': {
                    'title': 'Categories',
                    'tickangle': 90
                },
                'yaxis': {
                    'title': 'Categories'
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

    """
    Renders the 'go' webpage, which displays the classification results for a user's input.

    This function retrieves the user's input query from the request arguments, uses the trained model to predict the classification labels, and then renders the 'go.html' template with the query and classification results.

    Returns:
        Rendered HTML template for the 'go' page, with the user's input query and the classification results.
    """
    
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

    """
    The main function that runs the Flask app.

    This function starts the Flask app and runs it on the specified host and port with debug mode enabled.

    Usage:
        python run.py
    """
    
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()