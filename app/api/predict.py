import logging
import random
import pickle

import numpy as np
import sklearn
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator, confloat
from typing import Dict

log = logging.getLogger(__name__)
router = APIRouter()
dill = open('app\\api\kickstarter.pkl', 'rb')
model = pickle.load(dill)


class Campaign(BaseModel):
    """Use this data model to parse the request body JSON."""

    name: str = Field(..., example='The Songs of Adelaide & Abullah')
    blurb: str = Field(..., example='Phasellus viverra libero eget placerat')
    category: str = Field(..., example='Publishing')
    country: str = Field(..., example='GB')
    deadline: str = Field(..., example='2015-10-09')
    goal: confloat(ge=0) = Field(..., example=1000.00)
    launched: str = Field(..., example='2015-08-11')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""

        df = pd.DataFrame([dict(self)])
        df['deadline'] = pd.to_datetime(df['deadline'], format='%Y/%m/%d')
        df['launched'] = pd.to_datetime(df['launched'], format='%Y/%m/%d')
        df = df.drop(['category', 'name', 'blurb', 'country'], axis=1)
        df['campaign_length'] = df['deadline'] - df['launched']
        df['campaign_length'] = (df['campaign_length'] / np.timedelta64(1, 'D')).astype(int)
        df = df.drop(['deadline', 'launched'], axis=1)
        return df

@validator('goal')
def goal_must_be_positive(cls, value):
    """Validate that goal is a positive number."""
    assert value >= 0, f'goal == {value}, must be >= 0'
    return value

@validator('name')
def name_must_be_string(cls, value):
    """Validate that name is a string"""
    assert type(value) == str, f'name == {value}, must be a string'
    assert value

@validator('category')
def category_must_be_string(cls, value):
    """Validates that category is a string"""
    assert type(value) == str, f'category == {value}, must be a string'
    assert value

@validator('deadline')
def deadline_must_be_string(cls, value):
    """Validate that deadline is a string of length 10"""
    assert type(value) == str, f'deadline == {value}, must be a string'
    assert len(value) == 10, f'length of deadline == {len(value)}, must be == 10'
    assert value

@validator('launched')
def launched_must_be_string(cls, value):
    """Validate that launch is a string of length 10"""
    assert type(value) == str, f'deadline == {value}, must be a string'
    assert len(value) == 10, f'length of deadline == {len(value)}, must be == 10'
    assert value

 
@validator('country')
def country_validator(cls, value):
    """"Validates that the country is a string"""
    assert type(value) == str, f'country == {value}, must be a string'
    assert len(value) == 2, f'length of country == {len(value)}, must be == 2'
    assert value


@router.post('/predict')
def predict(campaign: Campaign):
    """

    ### Request Body
    - `name`: The name of the Kickstarter campaign
    - `blurb`: The description of the Kickstarter campaign
    - `category`: The main category of the Kickstarter campaign
    - `country`: The ISO 3166-1 alpha 2 code of the country the Kickstarter is based in
    - `deadline`: The end date of the Kickstarter campaign
    - `goal`: The funding goal of the Kickstarter campaign
    - `launched`: The start date of the Kickstarter campaign


    ### Response
    - `prediction`: boolean, 1 for success and 0 for failure
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    """


    X = campaign.to_df()
    log.info(X)
    y_pred = model.predict(X)
    return {
        'prediction': y_pred[0]
    }
