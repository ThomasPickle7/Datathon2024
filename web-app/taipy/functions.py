from taipy.gui import Gui, notify
import numpy as np
import pandas as pd 
import webbrowser
import datetime

def filter_by_date_range(dataset, start_date, end_date):
    """
    Function to filter data by custom start and end date
    """
    mask = (dataset['Date'] > start_date) & (dataset['Date'] <= end_date)
    return dataset.loc[mask]

def get_data(path: str):
    """
    Function to read a csv file into a pandas dataset.
    path: local path to .csv file
    output:
    dataset: a dataset with date column formatted to dash-seperated format
    """
    dataset = pd.read_csv(path)
    dataset["Date"] = pd.to_datetime(dataset["Date"]).dt.date
    return dataset

def np_csv(path):
    return

