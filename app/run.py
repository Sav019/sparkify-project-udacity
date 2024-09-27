import json
import re
import plotly
import pandas as pd
import os
import shutil

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Heatmap
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    # Tokenization logic (same as before)
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    rest_regex = r"[^a-zA-Z0-9]"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    stripped = re.sub(rest_regex, " ", text)
    tokens = word_tokenize(stripped)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stopwords.words("english")]
    return clean_tokens

# ---- New Section for Database Handling ----
# Check if running on Heroku (PostgreSQL) or local (SQLite)
if 'DATABASE_URL' in os.environ:
    # Running on Heroku, use PostgreSQL
    engine = create_engine(os.getenv('DATABASE_URL'))
else:
    # Running locally or in a Heroku dyno, use SQLite and copy it to /tmp
    db_path = '/tmp/DisasterResponse.db'
    bundled_db_path = '../data/DisasterResponse.db'

    # Copy bundled DB to /tmp if not already present
    if not os.path.exists(db_path):
        shutil.copyfile(bundled_db_path, db_path)

    # Create SQLite engine using the /tmp directory
    engine = create_engine(f'sqlite:///{db_path}')
# ---- End of New Section ----

# load data from the SQLite/PostgreSQL database
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = load("../models/classifier.pkl")

# The rest of the app logic (routes and functions) remains the same
@app.route('/')
@app.route('/index')
def index():
    # Function to render visuals (same as before)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    numeric_cols = df.select_dtypes(include='number').columns
    column_sums = df[numeric_cols].sum(axis=0).drop("id")
    column_names = list(column_sums.index)
    
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [Bar(x=column_names, y=column_sums)],
            'layout': {
                'title': 'Number of Data Points per Column',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Column"}
            }
        },
        {
            'data': [Heatmap(x=numeric_cols, y=numeric_cols, z=df[numeric_cols].corr(), colorscale='RdBu_r', zmin=-1, zmax=1)],
            'layout': {
                'title': 'Correlation Matrix',
                'xaxis': {'title': 'Categories', 'tickangle': 90},
                'yaxis': {'title': 'Categories'}
            }
        }
    ]
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '') 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()