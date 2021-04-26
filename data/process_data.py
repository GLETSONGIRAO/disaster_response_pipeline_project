import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
     """
    Load and merge messages and categories datasets
    
    Args:
        messages_filepath (string): Filepath for csv file containing messages dataset with id as unique identifier.
        categories_filepath (string): Filepath for csv file containing categories dataset with id as unique identifier.
       
    Returns:
        df (dataframe): Dataframe containing merged content of messages and categories datasets.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(categories, messages, on='id')
    
def clean_data(df):
     """
    Function for cleaning data: making categories 0 or 1 value, dropping duplicates and dropping bad data category.
    
    Args:
        df (dataframe): dataframe outputed at last function
    
    Returns: 
        df(dataframe): Cleaned dataframe.
    """
    
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[[0]]
    category_colnames = [x.split('-')[0] for x in row.values[0]]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype('str').str[-1]
        categories[column] = categories[column].astype(int)

    categories.drop('child_alone', axis = 1, inplace = True)
    
    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    categories['related'] = categories['related'].astype('int')
        
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    '''
    Function for saving the database in an .db file
    Args:   
        df (dataframe): DataFrame from last function
        database_filename (string): The path which we want to save the file
    Returns: 
        Nothing. The function saves the database in an sql file
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False,if_exists='replace')  


def main():
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