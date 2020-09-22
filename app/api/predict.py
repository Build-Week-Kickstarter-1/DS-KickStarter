import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator, confloat

log = logging.getLogger(__name__)
router = APIRouter()


class Campaign(BaseModel):
    """Use this data model to parse the request body JSON."""

    name: str = Field(..., example='The Songs of Adelaide & Abullah')
    category: str = Field(..., example='Publishing')
    currency: str = Field(..., example='GBP')
    deadline: str = Field(..., example='2015-10-09')
    goal: confloat(ge=0) = Field(..., example=1000.00)
    launched: str = Field(..., example='2015-08-11')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""

        df = pd.DataFrame([dict(self)])
        df['deadline'] = pd.to_datetime(df['deadline'])
        df['launched'] = pd.to_datetime(df['launched'])
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


@router.post('/predict')
async def predict(campaign: Campaign):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `name`: The name of the Kickstarter campaign
    - `category`: The main category of the Kickstarter campaign
    - `currency`: The three letter abbreviation of the currency the Kickstarter is based in
    - `deadline`: The end date of the Kickstarter campaign
    - `goal`: The funding goal of the Kickstarter campaign
    - `launched`: The start date of the Kickstarter campaign

    ### Other Categories to be added soon

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    X_new = campaign.to_df()
    log.info(X_new)
    y_pred = random.choice([True, False])
    return {
        'prediction': y_pred
    }
