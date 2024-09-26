# Disaster Response Pipeline Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Descriptions](#file-descriptions)
3. [Instructions](#instructions)
4. [Acknowledgements](#acknowledgements)

## Project Overview
This project is part of the Udacity Data Science Nanodegree program. The goal is to build a machine learning pipeline to categorize emergency messages based on the needs communicated in the text. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The dataset was provided by Figure Eight.

The main components of this project are:

1. **ETL Pipeline**: Loads the messages and categories datasets, merges them, and saves them in a SQLite database.
2. **ML Pipeline**: Builds a text processing and machine learning pipeline to classify the messages.
3. **Flask Web App**: Displays the results of the classification on a web app.

## File Descriptions
The project includes the following files and folders:

- "app/": This folder contains the files for the Flask web app.
  - "run.py": Starts the Flask web app and displays the visualizations and classification results.
  - "templates/": This folder contains the HTML templates for the web app, including "master.html" and "go.html".
- "data/": This folder contains the raw data files and the ETL pipeline.
  - "disaster_categories.csv": The raw data file containing the message categories.
  - "disaster_messages.csv": The raw data file containing the disaster messages.
  - "process_data.py": Runs the ETL pipeline that cleans the data and stores it in a SQLite database.
  - "DisasterResponse.db": The SQLite database created by the ETL pipeline.
- "models/": This folder contains the machine learning pipeline.
  - "train_classifier.py": Runs the ML pipeline that trains the classifier and saves the model as a pickle file.
  - "classifier.pkl": The trained machine learning model.
- "requirements.txt": The Python package requirements for the project.
- "runtime.txt": The runtime environment for the project.
- "Procfile": The Heroku configuration file.
- "README.md": This file, which provides an overview of the project.

## Instructions
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database:
        "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
    - To run ML ## Instructions (continued)

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database:
        "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
    - To run ML pipeline that trains classifier and saves:
        "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"

2. Run the web app:
    "python app/run.py"

3. Go to "http://0.0.0.0:3000/" to view the web app.

## Acknowledgements
- [Udacity](https://www.udacity.com/) for providing the starter code and data for this project.
- [Figure Eight](https://www.figure-eight.com/) for providing the dataset used in this project.