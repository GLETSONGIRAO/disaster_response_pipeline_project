# Disaster Response Pipeline Project

Project Overview

In this project we going to apply learned skills on Software development and pipeline for ELT, ML, NLP, to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. 

The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disaster_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Required packages:

- flask
- joblib
- jupyter # If you want to view the notebooks
- pandas
- plot.ly
- numpy
- scikit-learn
- sqlalchemy

### File structure of project:

1.  ../Notebooks - folder for project Notebooks

    ../Notebooks/ETL Pipeline Preparation.ipynb - notebook for ETL pipeline steps
    
    ../templates/ML Pipeline Preparation.ipynb - notebook for ML pipeline steps

    

2.  ../data - folder for files for the datasets

    ../data/disaster_categories.csv - raw file containing the categories
    
    ../data/disaster_messages.csv - raw file containing the messages
    
    ../data/process_data.py
    
    ../data/disaster_response.db - database created when running `python process_data.py`
    
    ../data/DisasterResponse.db - database for the clean data

    

3.  ../models - folder for the classifier model and pickle file

    ../models/train_classifier.py - model training script
    
    ../models/classifier.pkl - saved model when running `python train_classifier.py`
    


4.  ../app - folder for web app

    ../app/run.py - flask web app
    
    ../templates - .html templates


    

5.  README.md

