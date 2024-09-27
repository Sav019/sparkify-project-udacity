import sys

# import libraries
import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Loads and merges the messages and categories datasets.
    
    Args:
        messages_filepath (str): The file path of the messages dataset.
        categories_filepath (str): The file path of the categories dataset.
        
    Returns:
        pd.DataFrame: The merged dataset.
    """
    
    # Read in files
    messages = pd.read_csv(messages_filepath, encoding='utf-8')
    categories = pd.read_csv(categories_filepath, encoding='utf-8')

    # Merge datasets on ids
    df = pd.merge(messages, categories, on="id", how="inner")
      
    return df 
    
    pass


def clean_data(df):

    """
    Cleans the merged dataset by splitting the categories column, renaming the new columns, 
    converting the values to numeric, dropping rows with invalid values, and dropping duplicates.
    
    Args:
        df (pd.DataFrame): The merged dataset.
        
    Returns:
        pd.DataFrame: The cleaned dataset.
    """

    # Split the categories column
    categories = df["categories"].str.split(pat=";", expand=True)
    
    # Rename column names of new categories data set
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
    
    # Set values to numeric values only (0 or 1)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Drop rows that have values that are not exactly 0 or 1
    categories = categories[(categories.map(lambda x: x in [0, 1])).all(axis=1)]

    #remove columns that only have 0 as values
    categories = categories.loc[:, (categories != 0).any(axis=0)]
    
    # Drop the original categories column from df
    df = df.drop(columns='categories')

    # Concatenate the original df dataframe (minus the "old" categories column) with the new "categories" dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    
    # Drop duplicate values
    df = df.drop_duplicates()
    
    return df
    
    pass


def save_data(df, database_filename):

    """
    Saves the cleaned dataset to a SQLite database.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
        database_filename (str): The file path of the SQLite database.
    """
    
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    
    pass  


def main():

    """
    The main function that processes the data and saves it to a SQLite database.
    
    Usage:
        python process_data.py <messages_filepath> <categories_filepath> <database_filepath>
    
    Args:
        sys.argv[1] (str): The file path of the messages dataset.
        sys.argv[2] (str): The file path of the categories dataset.
        sys.argv[3] (str): The file path of the SQLite database.
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()