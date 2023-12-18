"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Demenia dataset into DataFrame.
    df = pd.read_csv('EEG_data.csv')

    # Perform feature and target split
    X = df[["EEG_AF3","EEG_F7","EEG_F3","EEG_FC5","EEG_T7","EEG_P7","EEG_O1","EEG_O2","EEG_P8","EEG_T8","EEG_FC6","EEG_F4","EEG_F8","EEG_AF4"]]
    y = df['outcome']

    return df, X, y

@st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1, 
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)

    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score